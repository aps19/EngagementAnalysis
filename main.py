import os
import torch
import logging
from config import Config  # Configurations (learning rate, batch size, etc.)
from training.train import train_model  # Core training function
from data_processing.data_loading import prepare_data  # DataLoader setup
from models.model import FusionNet  # The TCCT-Net model combining two streams
from utils.logger import setup_logging  # Logger setup
from utils.seed import set_seed  # Setting random seeds for reproducibility
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Progress bar
import argparse

# Check for GPU availability
def check_device(config):
    print(f"Using device: {config.device}")

def update_config_from_args(config, args):
    """Update the config object with arguments from argparse."""
    for key, value in vars(args).items():
        setattr(config, key, value)
    return config

if __name__ == "__main__":
    # Set up argparse for command line arguments
    parser = argparse.ArgumentParser(description="Prepare data for Engagement Prediction Model")
    
    parser.add_argument('--data_dir', type=str, default='data/processed_dataset', help='Folder containing the dataset')
    parser.add_argument('--dataset_fraction', type=float, default=0.01, help='Fraction of the dataset to use')
    parser.add_argument('--label_file', type=str, default='data/train_engagement_labels.xlsx', help='Path to the label file')
    parser.add_argument('--label_column', type=str, default='label', help='Column name for labels')
    parser.add_argument('--exclude_columns', type=list, default=['frame', 'timestamp'], help='Columns to exclude from features')
    parser.add_argument('--missing_value_strategy', type=str, default='interpolate', help='Strategy to handle missing values')
    parser.add_argument('--segment_length', type=int, default=50, help='Segment length for data augmentation')
    parser.add_argument('--num_augmented_samples', type=int, default=1, help='Number of augmented samples to generate')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for validation')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')

    # Parse the arguments
    args = parser.parse_args()

    # Instantiate the Config object
    config = Config()

    # Update config with command line arguments
    config = update_config_from_args(config, args)

    # Set up logging
    logger = setup_logging()

    # Check and log device usage
    check_device(config)
    
    logger.info("Preparing data with the following configuration:")
    for key, value in vars(config).items():
        logger.info(f"{key}: {value}")

    # Set seed for reproducibility
    set_seed(config.random_seed)

    # Prepare data
    train_loader, val_loader, train_dataset, val_dataset = prepare_data(config, logger)
    
    logger.info(f"Total samples used: {len(train_dataset) + len(val_dataset)}")
    logger.info(f"Training data: {len(train_loader.dataset)} samples")
    logger.info(f"Validation data: {len(val_loader.dataset)} samples")
    
    # Display progress in loading batches
    for batch_idx, (data, labels, seq_lengths) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Loading Training Data"):
        logger.info(f"Batch {batch_idx + 1}: Data shape: {data.shape}, Labels shape: {labels.shape}")

    # Insert your training logic here
    try:
        logger.info("Starting training process...")
        train_model(train_loader, val_loader, config)
        logger.info("Training process completed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
        sys.exit(1)
