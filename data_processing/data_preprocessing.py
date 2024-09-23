# data_processing/data_preprocessing.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple
import logging
from config import Config
config = Config()

# Logging
logging.basicConfig(
    filename=os.path.join(config.logs_dir, 'data_preprocessing.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_features(
    file_paths: List[str],
    feature_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    dtype: Optional[dict] = None  # Optional dtype specification
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load features from CSV files and extract specified feature columns.

    Args:
        file_paths (List[str]): List of file paths to CSV files.
        feature_columns (Optional[List[str]]): List of feature columns to extract.
            If None, all columns except those in exclude_columns are used.
        exclude_columns (Optional[List[str]]): List of columns to exclude from features.
        dtype (Optional[dict]): Optional dictionary specifying data types for certain columns.

    Returns:
        Tuple[List[np.ndarray], List[str]]: List of feature arrays and list of feature names.
    """
    features_list = []
    feature_names = []

    for file_path in tqdm(file_paths, desc="Loading Features"):
        try:
            # Reading CSV with low_memory=False to avoid dtype guessing issues for large files
            df = pd.read_csv(file_path, low_memory=False, dtype=dtype)
            
            # If feature_columns is not provided, generate it based on exclude_columns
            if feature_columns is None:
                if exclude_columns is None:
                    exclude_columns = []
                # Dynamically set the feature columns by excluding the specified ones
                feature_columns = [col for col in df.columns if col not in exclude_columns]
                feature_names = feature_columns
            else:
                feature_names = feature_columns

            # Ensure all feature columns exist in the dataframe
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                logging.warning(f"Missing columns {missing_cols} in {file_path}. Skipping this file.")
                continue

            # Extract the feature values from the dataframe
            features = df[feature_columns].values
            features_list.append(features)

        except pd.errors.EmptyDataError:
            logging.error(f"Empty data error while reading {file_path}. Skipping.")
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}. Skipping.")
        except Exception as e:
            logging.error(f"Unexpected error loading features from {file_path}: {e}")
            continue

    return features_list, feature_names
    
# Handle missing values
def handle_missing_values(features_list: List[np.ndarray], strategy: str = 'interpolate') -> List[np.ndarray]:
    processed_features = []
    for features in features_list:
        df = pd.DataFrame(features)
        if strategy == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both', axis=0)
        elif strategy == 'zero':
            df = df.fillna(0)
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        processed_features.append(df.values)
    return processed_features

# Normalize features
def normalize_features(features_list: List[np.ndarray], scaler: Optional[StandardScaler] = None) -> Tuple[List[np.ndarray], StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
        all_features = np.concatenate(features_list, axis=0)
        scaler.fit(all_features)
    normalized_features = [scaler.transform(features) for features in features_list]
    return normalized_features, scaler

# Preprocess data pipeline
def preprocess_data(file_paths: List[str], exclude_columns: List[str], missing_value_strategy: str = 'interpolate', scaler: Optional[StandardScaler] = None) -> Tuple[List[np.ndarray], StandardScaler, List[str]]:
    features_list, feature_names = load_features(file_paths, exclude_columns=exclude_columns)

    features_list = handle_missing_values(features_list, strategy=missing_value_strategy)

    features_list, scaler = normalize_features(features_list, scaler)

    return features_list, scaler, feature_names

def save_preprocessed_data(
    features_list: List[np.ndarray],
    labels: List[int],
    save_dir: str,
    file_names: List[str]
):
    """
    Save preprocessed features and labels to disk.

    Args:
        features_list (List[np.ndarray]): List of feature arrays.
        labels (List[int]): List of labels.
        save_dir (str): Directory to save the preprocessed data.
        file_names (List[str]): List of file names corresponding to each feature array.
    """
    os.makedirs(save_dir, exist_ok=True)
    for features, label, file_name in zip(features_list, labels, file_names):
        save_path = os.path.join(save_dir, file_name)
        np.savez(save_path, features=features, label=label)
        logging.info(f"Saved preprocessed data to {save_path}")
