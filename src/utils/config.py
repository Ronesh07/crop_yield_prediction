"""
Configuration settings for the Crop Yield Prediction application.
"""

# Data paths
DATA_PATH = "data"
MODELS_PATH = "models"
VISUALIZATIONS_PATH = "visualizations"

# Model training parameters
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5

# Default model parameters
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
DEFAULT_XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
DEFAULT_LIGHTGBM_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'num_leaves': 31
}
DEFAULT_NN_PARAMS = {
    'hidden_layers': [64, 32, 16],
    'epochs': 100,
    'batch_size': 32
}

# Feature engineering settings
DEFAULT_WINDOW_SIZES = [3, 5, 7]
DEFAULT_LAG_FEATURES = ['temperature', 'rainfall', 'humidity']

# Visualization settings
VIZ_COLORMAP = 'viridis'
VIZ_FIGSIZE_DEFAULT = (10, 6) 