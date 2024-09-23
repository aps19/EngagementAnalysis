# data_processing/data_loading.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Optional
from collections import Counter
from config import Config as config
from .data_preprocessing import preprocess_data
# Set up logging
log_dir = 'log_dir'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'data_pipeline.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Data Preprocessing Module
def load_features(
    file_paths: List[str],
    feature_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[List[np.ndarray], List[str]]:
    features_list = []
    feature_names = []
    for file_path in tqdm(file_paths, desc="Loading Features"):
        try:
            df = pd.read_csv(file_path)
            if feature_columns is None:
                if exclude_columns is None:
                    exclude_columns = []
                feature_columns = [col for col in df.columns if col not in exclude_columns]
                feature_names = feature_columns
            else:
                feature_names = feature_columns

            if not all(col in df.columns for col in feature_columns):
                missing_cols = [col for col in feature_columns if col not in df.columns]
                logging.warning(f"Missing columns {missing_cols} in {file_path}")
                continue

            features = df[feature_columns].values
            features_list.append(features)
        except Exception as e:
            logging.error(f"Error loading features from {file_path}: {e}")
            continue

    return features_list, feature_names

def handle_missing_values(
    features_list: List[np.ndarray],
    strategy: str = 'interpolate'
) -> List[np.ndarray]:
    processed_features = []
    for features in features_list:
        df = pd.DataFrame(features)
        if strategy == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both', axis=0)
        elif strategy == 'zero':
            df = df.fillna(0)
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        else:
            logging.error(f"Unknown missing value handling strategy: {strategy}")
            raise ValueError(f"Unknown strategy: {strategy}")

        processed_features.append(df.values)
    return processed_features

def normalize_features(
    features_list: List[np.ndarray],
    scaler: Optional[StandardScaler] = None
) -> Tuple[List[np.ndarray], StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
        all_features = np.concatenate(features_list, axis=0)
        scaler.fit(all_features)
        logging.info("Fitted new StandardScaler.")
    else:
        logging.info("Using provided StandardScaler.")

    normalized_features = [scaler.transform(features) for features in features_list]
    return normalized_features, scaler

# Data Augmentation Module
def segment_features(
    features: np.ndarray,
    segment_length: int
) -> List[np.ndarray]:
    T = features.shape[0]
    segments = []
    for start in range(0, T, segment_length):
        end = start + segment_length
        if end <= T:
            segment = features[start:end]
            segments.append(segment)
    return segments

def recombine_segments(
    segments: List[np.ndarray],
    num_segments: int
) -> np.ndarray:
    selected_indices = np.random.choice(len(segments), num_segments, replace=True)
    new_sequence = np.vstack([segments[i] for i in selected_indices])
    return new_sequence

def augment_features(
    features: np.ndarray,
    segment_length: int,
    num_augmented_samples: int
) -> List[np.ndarray]:
    segments = segment_features(features, segment_length)
    num_segments = len(segments)
    augmented_features = []
    for _ in range(num_augmented_samples):
        new_sequence = recombine_segments(segments, num_segments)
        augmented_features.append(new_sequence)

    return augmented_features

def augment_dataset(
    features_list: List[np.ndarray],
    labels: List[int],
    segment_length: int,
    num_augmented_samples: int
) -> Tuple[List[np.ndarray], List[int]]:
    augmented_features_list = []
    augmented_labels = []
    for features, label in zip(features_list, labels):
        augmented_sequences = augment_features(
            features, segment_length, num_augmented_samples
        )
        augmented_features_list.extend([features] + augmented_sequences)
        augmented_labels.extend([label] * (1 + num_augmented_samples))

    logging.info(f"Augmented dataset size: {len(augmented_features_list)} samples")
    return augmented_features_list, augmented_labels


# EngagementDataset class for handling the dataset
class EngagementDataset(Dataset):
    def __init__(self, features_list: List[np.ndarray], labels: List[int], feature_names: List[str], mode: str = 'train'):
        self.features_list = features_list
        self.labels = labels
        self.feature_names = feature_names
        self.mode = mode
        self.num_samples = len(self.features_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        features = self.features_list[idx]
        label = self.labels[idx]

        if self.mode == 'train':
            features = augment_features(features, segment_length=50, num_augmented_samples=1)[0]

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

# Custom collate function for padding sequences
def collate_fn(batch):
    data, labels = zip(*batch)
    sequence_lengths = [seq.shape[0] for seq in data]
    data_padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    labels = torch.stack(labels)
    return data_padded, labels, sequence_lengths

def prepare_data(config, logger):
    """
    Prepare the data loaders and datasets based on the configuration.
    """
    logger.info("Collecting File Paths and Labels...")
    file_paths, labels, feature_columns = load_csv_data(
        folder_path=config.data_dir,
        label_file=config.label_file,
        label_column=config.label_column,
        exclude_columns=config.exclude_columns
    )

    # Preprocess data
    logger.info("Preprocessing data...")
    features_list, scaler, feature_names = preprocess_data(
        file_paths, exclude_columns=config.exclude_columns,
        missing_value_strategy=config.missing_value_strategy
    )

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features_list, labels, test_size=config.test_size,
        stratify=labels, random_state=config.random_seed
    )

    # Data augmentation (if applicable)
    X_train_augmented, y_train_augmented = augment_dataset(
        X_train, y_train, segment_length=config.segment_length,
        num_augmented_samples=config.num_augmented_samples
    )

    # Create datasets and data loaders
    logger.info("Creating DataLoader objects...")
    train_dataset = EngagementDataset(X_train_augmented, y_train_augmented, feature_names, mode='train')
    val_dataset = EngagementDataset(X_val, y_val, feature_names, mode='val')

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset

# Load CSV data and labels
def load_csv_data(folder_path: str, label_file: str, label_column: str, exclude_columns: List[str]) -> Tuple[List[str], List[int], List[str]]:
    label_df = pd.read_excel(label_file)
    file_paths = []
    labels = []
    skipped_files = []
    feature_columns = None

    for filename in tqdm(os.listdir(folder_path), desc="Collecting File Paths and Labels"):
        if filename.endswith(".csv"):
            subject_id = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            label_row = label_df[label_df['chunk'].str.contains(subject_id)]
            if not label_row.empty:
                label = label_row[label_column].values[0]
                file_paths.append(file_path)
                labels.append(label)
                if feature_columns is None:
                    df = pd.read_csv(file_path, nrows=1)
                    feature_columns = [col for col in df.columns if col not in exclude_columns]
                    logging.info(f"Feature columns determined: {feature_columns}")
            else:
                logging.warning(f"No label found for subject {subject_id}")
                skipped_files.append(subject_id)

    return file_paths, labels, feature_columns


