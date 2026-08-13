import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

# Optional tensorflow support
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, Concatenate
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

class HybridCropYieldModel:
    """
    Hybrid Machine Learning + Deep Learning Model Architecture:
    Input Features -> Best ML Model (e.g. LightGBM / XGBoost) + Tabular DNN
                   -> Predictions
                   -> Fusion Layer (Dense 32 -> Dense 16 -> Dense 1)
                   -> Final Yield Prediction
    """
    def __init__(self, ml_model_type='auto', epochs=100, batch_size=32):
        self.ml_model_type = ml_model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.ml_model = None
        self.dl_model = None
        self.fusion_model = None
        self.selected_ml_name = None

    def _select_best_ml_model(self, X_train, y_train):
        candidates = {
            'lightgbm': lgb.LGBMRegressor(random_state=42, n_estimators=100, verbosity=-1),
            'xgboost': xgb.XGBRegressor(random_state=42, n_estimators=100),
            'random_forest': RandomForestRegressor(random_state=42, n_estimators=100)
        }
        
        best_name = 'lightgbm'
        best_score = -float('inf')
        best_model = None
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for name, model in candidates.items():
            scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                score = r2_score(y_val, preds)
                scores.append(score)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
                best_model = model
                
        best_model.fit(X_train, y_train)
        return best_model, best_name

    def _build_dnn(self, input_dim):
        model = Sequential([
            Dense(128, input_dim=input_dim),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),
            Dense(64),
            ReLU(),
            Dropout(0.2),
            Dense(32),
            ReLU(),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='huber', optimizer=optimizer, metrics=['mae'])
        return model

    def _build_fusion_network(self, input_dim):
        model = Sequential([
            Dense(32, input_dim=input_dim, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='huber', optimizer=optimizer, metrics=['mae'])
        return model

    def fit(self, X_train, y_train):
        import re
        X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train.copy()
        X_train_df.columns = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in X_train_df.columns]
        y_train_sr = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train

        # 1. Select & Train Best Base ML Model
        if self.ml_model_type == 'auto':
            self.ml_model, self.selected_ml_name = self._select_best_ml_model(X_train_df, y_train_sr)
        elif self.ml_model_type == 'xgboost':
            self.ml_model = xgb.XGBRegressor(random_state=42, n_estimators=100)
            self.ml_model.fit(X_train_df, y_train_sr)
            self.selected_ml_name = 'xgboost'
        elif self.ml_model_type == 'lightgbm':
            self.ml_model = lgb.LGBMRegressor(random_state=42, n_estimators=100, verbosity=-1)
            self.ml_model.fit(X_train_df, y_train_sr)
            self.selected_ml_name = 'lightgbm'
        else:
            self.ml_model = RandomForestRegressor(random_state=42, n_estimators=100)
            self.ml_model.fit(X_train_df, y_train_sr)
            self.selected_ml_name = 'random_forest'

        # 2. Out-of-Fold Predictions for ML & DNN to avoid data leakage
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof_ml_preds = np.zeros(len(X_train_df))
        oof_dl_preds = np.zeros(len(X_train_df))

        X_np = X_train_df.values
        y_np = y_train_sr.values

        for train_idx, val_idx in kf.split(X_np):
            X_tr, X_val = X_np[train_idx], X_np[val_idx]
            y_tr, y_val = y_np[train_idx], y_np[val_idx]

            # Fold ML Model
            if self.selected_ml_name == 'xgboost':
                fold_ml = xgb.XGBRegressor(random_state=42, n_estimators=80, n_jobs=-1)
            elif self.selected_ml_name == 'lightgbm':
                fold_ml = lgb.LGBMRegressor(random_state=42, n_estimators=80, n_jobs=-1, verbosity=-1)
            else:
                fold_ml = RandomForestRegressor(random_state=42, n_estimators=80, n_jobs=-1)
            fold_ml.fit(X_tr, y_tr)
            oof_ml_preds[val_idx] = fold_ml.predict(X_val)

            # Fold DL Model
            if HAS_TENSORFLOW:
                fold_dl = self._build_dnn(X_tr.shape[1])
                fold_dl.fit(
                    X_tr, y_tr,
                    epochs=min(self.epochs, 25),
                    batch_size=self.batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                    verbose=0
                )
                oof_dl_preds[val_idx] = fold_dl.predict(X_val, verbose=0).flatten()
            else:
                from sklearn.neural_network import MLPRegressor
                fold_dl = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=25, random_state=42)
                fold_dl.fit(X_tr, y_tr)
                oof_dl_preds[val_idx] = fold_dl.predict(X_val)

        # Train Full DNN on complete training set
        if HAS_TENSORFLOW:
            self.dl_model = self._build_dnn(X_np.shape[1])
            self.dl_model.fit(
                X_np, y_np,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
                ],
                verbose=0
            )
        else:
            from sklearn.neural_network import MLPRegressor
            self.dl_model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=self.epochs, random_state=42)
            self.dl_model.fit(X_np, y_np)

        # 3. Train Fusion Model using OOF predictions + top original features
        fusion_inputs_oof = np.column_stack([oof_ml_preds, oof_dl_preds, X_np])
        
        if HAS_TENSORFLOW:
            self.fusion_model = self._build_fusion_network(fusion_inputs_oof.shape[1])
            self.fusion_model.fit(
                fusion_inputs_oof, y_np,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                verbose=0
            )
        else:
            self.fusion_model = Ridge(alpha=1.0)
            self.fusion_model.fit(fusion_inputs_oof, y_np)

        return self

    def predict(self, X):
        import re
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_df.columns = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in X_df.columns]
        X_np = X_df.values

        # ML predictions
        ml_preds = self.ml_model.predict(X_df)

        # DL predictions
        if HAS_TENSORFLOW:
            dl_preds = self.dl_model.predict(X_np, verbose=0).flatten()
        else:
            dl_preds = self.dl_model.predict(X_np)

        # Fusion input
        fusion_inputs = np.column_stack([ml_preds, dl_preds, X_np])

        if HAS_TENSORFLOW:
            final_preds = self.fusion_model.predict(fusion_inputs, verbose=0).flatten()
        else:
            final_preds = self.fusion_model.predict(fusion_inputs)

        return final_preds

    def save(self, directory='models', prefix='hybrid_model'):
        os.makedirs(directory, exist_ok=True)
        meta = {
            'selected_ml_name': self.selected_ml_name,
            'has_tensorflow': HAS_TENSORFLOW
        }
        with open(os.path.join(directory, f"{prefix}_ml.pkl"), 'wb') as f:
            pickle.dump(self.ml_model, f)
        with open(os.path.join(directory, f"{prefix}_meta.pkl"), 'wb') as f:
            pickle.dump(meta, f)

        if HAS_TENSORFLOW:
            self.dl_model.save(os.path.join(directory, f"{prefix}_dl.h5"))
            self.fusion_model.save(os.path.join(directory, f"{prefix}_fusion.h5"))
        else:
            with open(os.path.join(directory, f"{prefix}_dl.pkl"), 'wb') as f:
                pickle.dump(self.dl_model, f)
            with open(os.path.join(directory, f"{prefix}_fusion.pkl"), 'wb') as f:
                pickle.dump(self.fusion_model, f)

    @classmethod
    def load(cls, directory='models', prefix='hybrid_model'):
        model_obj = cls()
        meta_path = os.path.join(directory, f"{prefix}_meta.pkl")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        model_obj.selected_ml_name = meta['selected_ml_name']
        
        with open(os.path.join(directory, f"{prefix}_ml.pkl"), 'rb') as f:
            model_obj.ml_model = pickle.load(f)

        if HAS_TENSORFLOW:
            model_obj.dl_model = tf.keras.models.load_model(os.path.join(directory, f"{prefix}_dl.h5"))
            model_obj.fusion_model = tf.keras.models.load_model(os.path.join(directory, f"{prefix}_fusion.h5"))
        else:
            with open(os.path.join(directory, f"{prefix}_dl.pkl"), 'rb') as f:
                model_obj.dl_model = pickle.load(f)
            with open(os.path.join(directory, f"{prefix}_fusion.pkl"), 'rb') as f:
                model_obj.fusion_model = pickle.load(f)

        return model_obj
