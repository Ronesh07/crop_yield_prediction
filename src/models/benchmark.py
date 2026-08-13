import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

from src.data import data_loader
from src.models import model_trainer
from src.models.hybrid_model import HybridCropYieldModel

# Try importing shap cleanly if available
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

def run_full_benchmark_and_evaluation():
    """
    Train & evaluate all models on the project dataset (yield_df.csv):
    - Linear Regression
    - Ridge Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - Tabular DNN
    - Hybrid ML-DL Model
    
    Saves metrics to models/model_metrics.json
    """
    df = data_loader.load_yield_data()
    if df is None:
        raise ValueError("Could not load yield dataset yield_df.csv")

    # Sample for fast, responsive training
    if len(df) > 6000:
        df = df.sample(6000, random_state=42).reset_index(drop=True)

    # One-hot encode categorical features (Area, Item)
    categorical_cols = ['Area', 'Item']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Sanitize feature column names for LightGBM/XGBoost
    import re
    df_encoded.columns = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in df_encoded.columns]

    feature_cols = [c for c in df_encoded.columns if c not in ['hg_ha_yield', 'Unnamed__0', 'Year']]
    X = df_encoded[feature_cols]
    y = df_encoded['hg_ha_yield']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    eval_results = []

    # 1. Linear Regression
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    eval_results.append({
        "Model": "Linear Regression",
        "R2": r2_score(y_test, lr_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, lr_preds)),
        "MAE": mean_absolute_error(y_test, lr_preds)
    })

    # 2. Ridge Regression
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    eval_results.append({
        "Model": "Ridge Regression",
        "R2": r2_score(y_test, ridge_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, ridge_preds)),
        "MAE": mean_absolute_error(y_test, ridge_preds)
    })

    # 3. Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    eval_results.append({
        "Model": "Random Forest",
        "R2": r2_score(y_test, rf_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "MAE": mean_absolute_error(y_test, rf_preds)
    })

    # 4. XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    eval_results.append({
        "Model": "XGBoost",
        "R2": r2_score(y_test, xgb_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, xgb_preds)),
        "MAE": mean_absolute_error(y_test, xgb_preds)
    })

    # 5. LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    eval_results.append({
        "Model": "LightGBM",
        "R2": r2_score(y_test, lgb_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, lgb_preds)),
        "MAE": mean_absolute_error(y_test, lgb_preds)
    })

    # 6. Tabular DNN
    print("Training Deep Neural Network (DNN)...")
    dnn_model, _ = model_trainer.train_neural_network(X_train, y_train, epochs=40)
    dnn_metrics = model_trainer.evaluate_model(dnn_model, X_test, y_test, is_neural_network=True)
    eval_results.append({
        "Model": "DNN",
        "R2": dnn_metrics['r2'],
        "RMSE": dnn_metrics['rmse'],
        "MAE": dnn_metrics['mae']
    })

    # 7. Hybrid ML-DL Model
    print("Training Hybrid ML-DL Model...")
    hybrid = HybridCropYieldModel(epochs=40)
    hybrid.fit(X_train, y_train)
    hybrid_preds = hybrid.predict(X_test)
    eval_results.append({
        "Model": "Hybrid ML-DL",
        "R2": r2_score(y_test, hybrid_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, hybrid_preds)),
        "MAE": mean_absolute_error(y_test, hybrid_preds)
    })

    # Save models
    model_trainer.save_model(rf, 'random_forest')
    model_trainer.save_model(xgb_model, 'xgboost')
    model_trainer.save_model(lgb_model, 'lightgbm')
    model_trainer.save_model(dnn_model, 'dnn', is_neural_network=True)
    hybrid.save('models', 'hybrid_model')

    # Save metrics JSON
    metrics_file = os.path.join('models', 'model_comparison_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(eval_results, f, indent=4)

    results_df = pd.DataFrame(eval_results)
    print("\n=== Model Benchmark Summary ===")
    print(results_df.to_string(index=False))

    return results_df, lgb_model, feature_cols

if __name__ == "__main__":
    run_full_benchmark_and_evaluation()
