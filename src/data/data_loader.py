import pandas as pd
import numpy as np
import os

# Get the absolute path to the data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_crop_recommendation_data():
    """
    Load the crop recommendation dataset
    """
    try:
        file_path = os.path.join(DATA_DIR, "Crop_recommendation.csv")
        print(f"Loading from: {file_path}")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading crop recommendation data: {e}")
        return None

def load_rainfall_data():
    """
    Load the rainfall dataset
    """
    try:
        file_path = os.path.join(DATA_DIR, "rainfall.csv")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading rainfall data: {e}")
        return None

def load_temperature_data():
    """
    Load the temperature dataset
    """
    try:
        file_path = os.path.join(DATA_DIR, "temp.csv")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading temperature data: {e}")
        return None

def load_yield_data():
    """
    Load the crop yield dataset
    """
    try:
        file_path = os.path.join(DATA_DIR, "yield_df.csv")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading yield data: {e}")
        return None

def load_pesticides_data():
    """
    Load the pesticides dataset
    """
    try:
        file_path = os.path.join(DATA_DIR, "pesticides.csv")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading pesticides data: {e}")
        return None

def get_sample_data(df, n=5):
    """
    Get a sample of the data
    """
    if df is not None:
        return df.sample(min(n, len(df)))
    return None

def get_dataset_info(df):
    """
    Get basic information about the dataset
    """
    if df is not None:
        info = {
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        return info
    return None

def get_dataset_summary(df):
    """
    Get statistical summary of the dataset
    """
    if df is not None:
        return df.describe()
    return None

def merge_datasets(df1, df2, on=None, how='inner'):
    """
    Merge two datasets based on common columns
    """
    if df1 is not None and df2 is not None:
        if on is None:
            # Try to find common columns
            common_cols = list(set(df1.columns) & set(df2.columns))
            if common_cols:
                return pd.merge(df1, df2, on=common_cols, how=how)
            else:
                print("No common columns found for merging")
                return None
        else:
            return pd.merge(df1, df2, on=on, how=how)
    return None 