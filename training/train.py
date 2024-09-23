import os
import sys
import torch
import logging
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from config import Config
from models.model import FusionNet
from utils.logger import setup_logging
from utils.seed import set_seed
from data_processing.data_loading import prepare_data
from training.trainer import Trainer
from argparse import ArgumentParser
# libraries for multi-gpu setup
import torch.distributed as dist
import torch.multiprocessing as mp

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

def setup_ddp_environment(local_rank):
    """Setup the process group for distributed training."""
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_model():
    try:
        args = parse_arguments()
        config = Config()
        set_seed(config.random_seed)

        # Setup logging
        os.makedirs(config.logs_dir, exist_ok=True)
        log_file = os.path.join(config.logs_dir, 'training.log')
        setup_logging(log_file)
        logging.info("Starting training process.")

        # Setup Distributed Data Parallel (DDP) environment
        setup_ddp_environment(args.local_rank)

        # Prepare data
        logging.info("Preparing data...")
        train_loader, val_loader, train_dataset, val_dataset = prepare_data(config)

        # Model Initialization
        logging.info("Initializing model...")
        model = FusionNet(config)

        # Move model to the appropriate device
        model = model.to(args.local_rank)

        # Wrap the model with DDP
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

        logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

        # Optimizer, Criterion, and Trainer Initialization
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        criterion = CrossEntropyLoss()

        trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, config)
        trainer.train()

        logging.info("Training process completed successfully.")

        cleanup_ddp()  # Cleanup DDP environment after training
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}", exc_info=True)
        sys.exit(1)
        

