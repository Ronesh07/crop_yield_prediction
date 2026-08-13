import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.hybrid_model import HybridCropYieldModel

def train_and_evaluate_crop_specific_models(df, item_col='Item', yield_col='hg/ha_yield', feature_cols=None, min_samples=200):
    """
    Train and evaluate Crop-Specific Hybrid ML-DL models against the Global Hybrid ML-DL model.
    """
    if feature_cols is None:
        feature_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

    if len(df) > 4000:
        df = df.sample(4000, random_state=42).reset_index(drop=True)

    # Filter out crops with insufficient samples
    item_counts = df[item_col].value_counts()
    eligible_crops = item_counts[item_counts >= min_samples].index.tolist()[:5]

    # Preprocess categorical features if present
    df_encoded = df.copy()
    if 'Area' in df_encoded.columns and 'Area' not in feature_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=['Area'], drop_first=True)

    numeric_features = [col for col in df_encoded.columns if col not in [item_col, yield_col, 'Unnamed: 0', 'Year']]

    # Split Global Train / Test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)

    X_train_global = train_df[numeric_features]
    y_train_global = train_df[yield_col]
    X_test_global = test_df[numeric_features]
    y_test_global = test_df[yield_col]

    print("Training Global Hybrid ML-DL Model...")
    global_hybrid = HybridCropYieldModel(epochs=15)
    global_hybrid.fit(X_train_global, y_train_global)
    global_preds = global_hybrid.predict(X_test_global)

    global_r2 = r2_score(y_test_global, global_preds)
    global_rmse = np.sqrt(mean_squared_error(y_test_global, global_preds))
    global_mae = mean_absolute_error(y_test_global, global_preds)

    crop_results = []

    for crop in eligible_crops:
        print(f"Training Crop-Specific Hybrid Model for: {crop}...")
        crop_train = train_df[train_df[item_col] == crop]
        crop_test = test_df[test_df[item_col] == crop]

        if len(crop_train) < 50 or len(crop_test) < 20:
            continue

        X_train_crop = crop_train[numeric_features]
        y_train_crop = crop_train[yield_col]
        X_test_crop = crop_test[numeric_features]
        y_test_crop = crop_test[yield_col]

        # Evaluate Global Hybrid on this specific crop subset
        g_crop_preds = global_hybrid.predict(X_test_crop)
        g_crop_r2 = r2_score(y_test_crop, g_crop_preds)
        g_crop_rmse = np.sqrt(mean_squared_error(y_test_crop, g_crop_preds))
        g_crop_mae = mean_absolute_error(y_test_crop, g_crop_preds)

        # Train Crop-Specific Hybrid
        crop_hybrid = HybridCropYieldModel(epochs=15)
        crop_hybrid.fit(X_train_crop, y_train_crop)
        c_crop_preds = crop_hybrid.predict(X_test_crop)

        c_crop_r2 = r2_score(y_test_crop, c_crop_preds)
        c_crop_rmse = np.sqrt(mean_squared_error(y_test_crop, c_crop_preds))
        c_crop_mae = mean_absolute_error(y_test_crop, c_crop_preds)

        crop_results.append({
            'Crop': crop,
            'Global Hybrid R2': float(g_crop_r2),
            'Crop-Specific Hybrid R2': float(c_crop_r2),
            'Global Hybrid RMSE': float(g_crop_rmse),
            'Crop-Specific Hybrid RMSE': float(c_crop_rmse),
            'Global Hybrid MAE': float(g_crop_mae),
            'Crop-Specific Hybrid MAE': float(c_crop_mae)
        })

    results_path = os.path.join('models', 'crop_specific_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(crop_results, f, indent=4)

    return pd.DataFrame(crop_results)
