import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Optional tensorflow support
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Create model save directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def train_linear_regression(X_train, y_train, cv=5):
    """
    Train a Linear Regression model
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series or array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    Trained model and cross-validation scores
    """
    model = LinearRegression()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    
    # Train the model on the full training data
    model.fit(X_train, y_train)
    
    return model, cv_scores

def train_ridge_regression(X_train, y_train, cv=5, alphas=None):
    """
    Train a Ridge Regression model with cross-validation
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series or array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
    alphas : list, default=None
        List of alpha values to try
        
    Returns:
    --------
    Trained model and cross-validation scores
    """
    alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Create parameter grid
    param_grid = {'alpha': alphas}
    
    # Create grid search object
    grid_search = GridSearchCV(
        Ridge(), 
        param_grid, 
        cv=cv, 
        scoring='r2', 
        return_train_score=True
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_score_

def train_random_forest(X_train, y_train, cv=5, params=None):
    """
    Train a Random Forest model with grid search
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series or array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
    params : dict, default=None
        Parameters for grid search
        
    Returns:
    --------
    Trained model and best score
    """
    if params is None:
        params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    # Create grid search object
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42), 
        params, 
        cv=cv, 
        scoring='r2', 
        return_train_score=True,
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_score_

def train_xgboost(X_train, y_train, cv=5, params=None):
    """
    Train an XGBoost model
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series or array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
    params : dict, default=None
        Parameters for grid search
        
    Returns:
    --------
    Trained model and best score
    """
    if params is None:
        params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    
    # Create grid search object
    grid_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42), 
        params, 
        cv=cv, 
        scoring='r2', 
        return_train_score=True,
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_score_

def train_lightgbm(X_train, y_train, cv=5, params=None):
    """
    Train a LightGBM model
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series or array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
    params : dict, default=None
        Parameters for grid search
        
    Returns:
    --------
    Trained model and best score
    """
    if params is None:
        params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 63, 127]
        }
    
    # Create grid search object
    grid_search = GridSearchCV(
        lgb.LGBMRegressor(random_state=42), 
        params, 
        cv=cv, 
        scoring='r2', 
        return_train_score=True,
        n_jobs=-1  # Use all available CPUs
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_score_

def train_neural_network(X_train, y_train, hidden_layers=None, epochs=100, batch_size=32):
    """
    Train a Deep Neural Network (DNN) model for crop yield prediction.
    Architecture:
    Input -> Dense(128) -> BatchNormalization -> ReLU -> Dropout(0.2)
          -> Dense(64)  -> ReLU -> Dropout(0.2)
          -> Dense(32)  -> ReLU
          -> Dense(1) -> Yield
    """
    if HAS_TENSORFLOW:
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        
        model = Sequential([
            Dense(128, input_dim=X_train_np.shape[1]),
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
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5
        )
        
        history = model.fit(
            X_train_np, y_train_np,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        return model, history
    else:
        # Fallback to Scikit-Learn MLPRegressor if TensorFlow is unavailable
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=epochs,
            early_stopping=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model, None

def evaluate_model(model, X_test, y_test, is_neural_network=False):
    """
    Evaluate a trained model
    
    Parameters:
    -----------
    model : trained model
        The trained model to evaluate
    X_test : pandas DataFrame
        Test features
    y_test : pandas Series or array-like
        Test target
    is_neural_network : bool, default=False
        Whether the model is a neural network
        
    Returns:
    --------
    Dictionary of evaluation metrics
    """
    # Convert inputs to numpy arrays for neural networks
    X_test_np = X_test.values if hasattr(X_test, 'values') and is_neural_network else X_test
    y_test_np = y_test.values if hasattr(y_test, 'values') and is_neural_network else y_test
    
    # Make predictions
    if is_neural_network:
        y_pred = model.predict(X_test_np).flatten()
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return metrics

def save_model(model, model_name, is_neural_network=False):
    """
    Save a trained model
    
    Parameters:
    -----------
    model : trained model
        The trained model to save
    model_name : str
        Name for the saved model
    is_neural_network : bool, default=False
        Whether the model is a neural network
        
    Returns:
    --------
    Path to the saved model
    """
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    if is_neural_network:
        model_path = os.path.join(model_dir, f"{model_name}")
        if hasattr(model, 'save'):
            model.save(model_path)
        else:
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    else:
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    return model_path

def load_model(model_path, is_neural_network=False):
    """
    Load a trained model
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    is_neural_network : bool, default=False
        Whether the model is a neural network
        
    Returns:
    --------
    Loaded model
    """
    if is_neural_network:
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is not installed. Neural network models cannot be loaded.")
        model = tf.keras.models.load_model(model_path)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    return model

def get_feature_importance(model, feature_names, model_type):
    """
    Get feature importance from the model
    
    Parameters:
    -----------
    model : trained model
        The trained model
    feature_names : list
        List of feature names
    model_type : str
        Type of the model ('linear', 'tree', 'xgboost', 'lightgbm')
        
    Returns:
    --------
    DataFrame with feature importances
    """
    if model_type == 'linear':
        # For linear models
        importance = np.abs(model.coef_)
    elif model_type in ['tree', 'random_forest']:
        # For tree-based models
        importance = model.feature_importances_
    elif model_type == 'xgboost':
        # For XGBoost
        importance = model.feature_importances_
    elif model_type == 'lightgbm':
        # For LightGBM
        importance = model.feature_importances_
    else:
        return None
    
    # Create a DataFrame of feature importances
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    return feature_importance 