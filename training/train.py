import os
import sys
import torch
import logging
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from config import Config()
from models.model import FusionNet
from utils.logger import setup_logging
from utils.seed import set_seed
from data_processing.data_loading import prepare_data
from training.trainer import Trainer
from argparse import ArgumentParser


def parse_arguments():
    """
    Parse command-line arguments for customizing training options.
    """
    parser = ArgumentParser(description="Train the Engagement Prediction Model.")
    
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file")
    parser.add_argument('--epochs', type=int, default=None, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=None, help="Learning rate")
    parser.add_argument('--config', type=str, default=None, help="Path to custom config file (not implemented)")
    
    return parser.parse_args()


def train_model():
    """
    Main function to run the training process.
    """
    try:
        # ----------------------------
        # Parse Command-Line Arguments
        # ----------------------------
        args = parse_arguments()

        # ----------------------------
        # Initialize Configuration
        # ----------------------------
        config = Config()
        set_seed(config.random_seed)  # Set seed for reproducibility

        # Override config with command-line arguments if provided
        if args.resume:
            config.resume_training = True
            logging.info(f"Resuming training from checkpoint: {args.checkpoint}")
            if args.checkpoint:
                config.checkpoint_path = args.checkpoint

        if args.epochs:
            config.num_epochs = args.epochs
            logging.info(f"Training for {config.num_epochs} epochs.")

        if args.batch_size:
            config.batch_size = args.batch_size
            logging.info(f"Using batch size: {config.batch_size}")

        if args.learning_rate:
            config.learning_rate = args.learning_rate
            logging.info(f"Using learning rate: {config.learning_rate}")

        # ----------------------------
        # Setup Logging
        # ----------------------------
        os.makedirs(config.logs_dir, exist_ok=True)
        log_file = os.path.join(config.logs_dir, 'training.log')
        setup_logging(log_file)
        logging.info("Starting training process.")

        # ----------------------------
        # Data Preparation
        # ----------------------------
        logging.info("Preparing data...")
        train_loader, val_loader, train_dataset, val_dataset = prepare_data(
            folder_path=config.data_dir,
            label_file=config.label_file,
            label_column=config.label_column,
            exclude_columns=config.exclude_columns,
            missing_value_strategy=config.missing_value_strategy,
            segment_length=config.segment_length,
            num_augmented_samples=config.num_augmented_samples,
            test_size=config.test_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            random_state=config.random_seed
        )
        logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        # ----------------------------
        # Model Initialization
        # ----------------------------
        logging.info("Initializing model...")
        model = FusionNet(config).to(config.device)
        logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

        # ----------------------------
        # Optimizer, Criterion, and Trainer Initialization
        # ----------------------------
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        criterion = CrossEntropyLoss()

        trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, config)
        trainer.train()

        logging.info("Training process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}", exc_info=True)
        sys.exit(1)
