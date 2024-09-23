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
from config import Config
from .data_preprocessing import preprocess_data

config = Config()

# Set up logging
logging.basicConfig(
    filename=os.path.join(config.logs_dir, 'data_pipeline.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def prepare_data(config, logger):
    """
    Prepare the data loaders and datasets based on the configuration.
    Loads only a specified fraction of the dataset.
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
        file_paths,
        exclude_columns=config.exclude_columns,
        missing_value_strategy=config.missing_value_strategy
    )

    # Ensure that features_list and labels are of the same length
    assert len(features_list) == len(labels), "Features and labels must have the same length."

    # Set seed for reproducibility
    torch.manual_seed(config.random_state)

    # Calculate the number of samples to use (e.g., 20% of the total dataset)
    num_total = len(features_list)
    num_subset = int(num_total * config.dataset_fraction)
    logger.info(f"Sampling {config.dataset_fraction * 100}% of the dataset: {num_subset} samples out of {num_total}")

    # Generate random indices for sampling
    indices = torch.randperm(num_total)[:num_subset].tolist()

    # Create a subset of the data using the sampled indices
    X_subset = [features_list[i] for i in indices]
    y_subset = [labels[i] for i in indices]

    # Split the subset into training and validation sets
    logger.info("Splitting subset into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_subset,
        y_subset,
        test_size=config.test_size,
        stratify=y_subset if config.test_size > 0 else None,
        random_state=config.random_state
    )

    logger.info(f"Training samples before augmentation: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Data augmentation (if applicable)
    if config.num_augmented_samples > 0:
        logger.info("Augmenting training data...")
        X_train_augmented, y_train_augmented = augment_dataset(
            X_train,
            y_train,
            segment_length=config.segment_length,
            num_augmented_samples=config.num_augmented_samples
        )
        logger.info(f"Training samples after augmentation: {len(X_train_augmented)}")
    else:
        X_train_augmented, y_train_augmented = X_train, y_train

    # Create EngagementDataset instances
    logger.info("Creating EngagementDataset instances...")
    train_dataset = EngagementDataset(X_train_augmented, y_train_augmented, feature_names, mode='train')
    val_dataset = EngagementDataset(X_val, y_val, feature_names, mode='val')

    # Create DataLoader objects
    logger.info("Creating DataLoader objects...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Log the number of samples
    logger.info(f"Total samples used: {len(train_dataset) + len(val_dataset)}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader, train_dataset, val_dataset


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


