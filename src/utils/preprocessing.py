import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

def handle_missing_values(df, strategy='mean', categorical_cols=None):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    strategy : str, default='mean'
        Strategy for imputation ('mean', 'median', 'most_frequent')
    categorical_cols : list, default=None
        List of categorical columns
        
    Returns:
    --------
    pandas DataFrame with imputed values
    """
    if df is None:
        return None
    
    df_copy = df.copy()
    
    # Handle numerical columns
    numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col not in (categorical_cols or [])]
    
    if len(numerical_cols) > 0:
        imputer = SimpleImputer(strategy=strategy)
        df_copy[numerical_cols] = imputer.fit_transform(df_copy[numerical_cols])
    
    # Handle categorical columns
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_copy[categorical_cols] = cat_imputer.fit_transform(df_copy[categorical_cols])
    
    return df_copy

def encode_categorical_features(df, categorical_cols):
    """
    Encode categorical features using One-Hot Encoding
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    categorical_cols : list
        List of categorical columns
        
    Returns:
    --------
    pandas DataFrame with encoded features and the encoder object
    """
    if df is None or not categorical_cols:
        return df, None
    
    df_copy = df.copy()
    
    # Check if the categorical columns exist in the dataframe
    valid_cat_cols = [col for col in categorical_cols if col in df_copy.columns]
    
    if not valid_cat_cols:
        return df_copy, None
    
    # One-hot encode the categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(df_copy[valid_cat_cols])
    
    # Create a dataframe with the encoded features
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(valid_cat_cols),
        index=df_copy.index
    )
    
    # Drop the original categorical columns and concatenate the encoded features
    df_copy = df_copy.drop(columns=valid_cat_cols)
    df_copy = pd.concat([df_copy, encoded_df], axis=1)
    
    return df_copy, encoder

def scale_features(df, method='standard', exclude_cols=None):
    """
    Scale numerical features
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    method : str, default='standard'
        Scaling method ('standard', 'minmax')
    exclude_cols : list, default=None
        Columns to exclude from scaling (like target variable)
        
    Returns:
    --------
    pandas DataFrame with scaled features and the scaler object
    """
    if df is None:
        return None, None
    
    df_copy = df.copy()
    exclude_cols = exclude_cols or []
    
    # Get numerical columns except excluded ones
    numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    if len(numerical_cols) > 0:
        # Scale the features
        if method == 'standard':
            scaler = StandardScaler()
        else:  # minmax
            scaler = MinMaxScaler()
        
        df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
        
        return df_copy, scaler
    
    return df_copy, None

def select_features(X, y, k=10, method='f_regression'):
    """
    Select top k features based on statistical tests
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : pandas Series or array-like
        Target variable
    k : int, default=10
        Number of top features to select
    method : str, default='f_regression'
        Feature selection method ('f_regression', 'mutual_info')
        
    Returns:
    --------
    X with selected features, indices of selected features
    """
    if X is None or y is None:
        return None, None
    
    # Choose the feature selection method
    if method == 'f_regression':
        selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
    else:  # mutual_info
        selector = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))
    
    # Fit and transform
    X_new = selector.fit_transform(X, y)
    
    # Get selected feature indices
    selected_features = selector.get_support(indices=True)
    
    # Create a new dataframe with selected features
    X_selected = pd.DataFrame(
        X_new,
        columns=[X.columns[i] for i in selected_features],
        index=X.index
    )
    
    return X_selected, selected_features

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create train-test split
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : pandas Series or array-like
        Target variable
    test_size : float, default=0.2
        Size of the test set
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    if X is None or y is None:
        return None, None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def feature_engineering(df, window_sizes=None, lag_features=None, target_col=None):
    """
    Create additional features from the existing ones
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    window_sizes : list, default=None
        List of window sizes for rolling statistics
    lag_features : list, default=None
        List of columns to create lag features from
    target_col : str, default=None
        Target column name
        
    Returns:
    --------
    pandas DataFrame with additional features
    """
    if df is None:
        return None
    
    df_copy = df.copy()
    
    # Create rolling statistics features if time series
    if window_sizes and target_col and target_col in df_copy.columns:
        for window in window_sizes:
            df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).mean()
            df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).std().fillna(0)
    
    # Create lag features
    if lag_features:
        for col in lag_features:
            if col in df_copy.columns:
                for lag in range(1, 4):  # Create lags 1, 2, and 3
                    df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag).fillna(df_copy[col].mean())
    
    # Create interaction features for numerical columns
    numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != target_col]
    
    if len(numerical_cols) > 1:
        # Create some interaction features (limited to prevent explosion of features)
        for i in range(min(len(numerical_cols), 3)):
            for j in range(i+1, min(len(numerical_cols), 4)):
                col1, col2 = numerical_cols[i], numerical_cols[j]
                df_copy[f'{col1}_{col2}_interaction'] = df_copy[col1] * df_copy[col2]
    
    return df_copy 