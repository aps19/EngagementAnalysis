import os
import torch
import logging
from config import Config  # Import the custom config class
from training.train import train_model  # Core training function
from data_processing.data_loading import prepare_data  # DataLoader setup
from models.model import FusionNet  # The TCCT-Net model combining two streams
from utils.logger import setup_logging  # Logger setup
from utils.seed import set_seed  # Setting random seeds for reproducibility
from tqdm import tqdm  # Progress bar

# Set up DDP environment (if applicable, for multi-GPU support)
def setup_ddp_environment(local_rank):
    """Setup the process group for distributed training."""
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

# Clean up DDP after training
def cleanup_ddp():
    dist.destroy_process_group()

def main():
    # Initialize the Config object
    config = Config()

    # Set seed for reproducibility
    set_seed(config.random_seed)

    # Set up logging
    logger = setup_logging()

    # Log device info and configuration settings
    logger.info(f"Using device: {config.device}")
    logger.info("Configuration settings:")
    for key, value in vars(config).items():
        logger.info(f"{key}: {value}")

    # Prepare data (preprocessing, augmentation, etc.)
    try:
        logger.info("Preparing data...")
        train_loader, val_loader, train_dataset, val_dataset = prepare_data(config, logger)
        logger.info(f"Total samples: {len(train_dataset) + len(val_dataset)}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}", exc_info=True)
        return

    # Initialize model
    try:
        logger.info("Initializing model...")
        model = FusionNet(config).to(config.device)
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    except Exception as e:
        logger.error(f"Error initializing the model: {str(e)}", exc_info=True)
        return

    # Display progress in loading batches
    try:
        for batch_idx, (data, labels, seq_lengths) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Loading Training Data"):
            logger.info(f"Batch {batch_idx + 1}: Data shape: {data.shape}, Labels shape: {labels.shape}")
    except Exception as e:
        logger.error(f"Error during batch loading: {str(e)}", exc_info=True)
        return

    # Train the model
    try:
        logger.info("Starting training process...")
        train_model(train_loader, val_loader, config)
        logger.info("Training process completed successfully.")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        cleanup_ddp()  # Clean up DDP environment in case of error
        return

    # Cleanup DDP after training (if applicable)
    cleanup_ddp()

if __name__ == "__main__":
    main()
