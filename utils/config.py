import torch
import os
import random
import numpy as np

class Config:
    """
    Configuration class to hold all hyperparameters and settings for the project.
    Adjust the values as needed to customize the behavior of the data loading,
    preprocessing, model architecture, and training process.
    """

    def __init__(self):
        # ----------------------------
        # General Settings
        # ----------------------------
        self.random_seed = 42  # Seed for reproducibility

        # Device Configuration
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # To force CPU usage, uncomment the following line:
        self.device = torch.device('cuda')

        # ----------------------------
        # Data Paths
        # ----------------------------
        self.data_dir = 'data/processed_dataset'  # Path to the folder containing CSV files
        self.label_file = 'data/train_engagement_labels.xlsx'  # Path to the Excel file with labels
        self.label_column = 'label'  # Column name in the label file for labels

        # ----------------------------
        # Data Preprocessing Parameters
        # ----------------------------
        self.exclude_columns = ['frame', 'timestamp']  # Columns to exclude from features
        self.missing_value_strategy = 'interpolate'  # Strategy: 'interpolate', 'zero', 'mean'
        self.max_sequence_length = 512  # Max sequence length after padding/truncating

        # ----------------------------
        # Data Augmentation Parameters
        # ----------------------------
        self.use_data_augmentation = True  # Whether to apply data augmentation
        self.segment_length = 50  # Length of each segment for S&R augmentation
        self.num_augmented_samples = 1  # Number of augmented samples per original sample

        # ----------------------------
        # Data Splitting Parameters
        # ----------------------------
        self.test_size = 0.2  # Proportion of data for validation set
        self.stratify = True  # Whether to stratify the split based on labels

        # ----------------------------
        # DataLoader Parameters
        # ----------------------------
        self.batch_size = 1  # Number of samples per batch
        self.num_workers = 2  # Number of subprocesses for data loading
        self.pin_memory = True  # Whether to copy tensors into CUDA pinned memory

        # ----------------------------
        # Model Hyperparameters
        # ----------------------------
        # Input/Output Dimensions
        self.input_dim = 68  # Number of input features (adjust based on your dataset)
        self.num_classes = 4  # Number of output classes (adjust based on your labels)

        # CT Stream Parameters
        self.dim_model = 128  # Model dimension for the Conformer blocks
        self.num_heads = 4  # Number of attention heads in Multi-Head Attention
        self.num_layers = 2  # Number of Conformer blocks in the CT Stream

        # TC Stream Parameters
        self.scales = 128  # Number of scales for the Continuous Wavelet Transform (CWT)

        # ----------------------------
        # Training Hyperparameters
        # ----------------------------
        self.num_epochs = 50  # Number of training epochs
        self.learning_rate = 1e-4  # Learning rate for the optimizer
        self.weight_decay = 1e-5  # Weight decay (L2 regularization)
        self.gradient_clip_val = 1.0  # Max norm for gradient clipping

        # Optimizer Settings
        self.optimizer_type = 'Adam'  # Type of optimizer ('Adam', 'SGD', etc.)

        # Learning Rate Scheduler (optional)
        self.use_scheduler = False  # Whether to use a learning rate scheduler
        self.scheduler_step_size = 10  # Step size for StepLR scheduler
        self.scheduler_gamma = 0.1  # Gamma (decay rate) for StepLR scheduler

        # Mixed Precision Training
        self.use_amp = True  # Use Automatic Mixed Precision (requires compatible hardware)
        self.gradient_clip_val = 1.0  # Gradient clipping value
        self.checkpoint_dir = 'checkpoints/'  # Directory for saving checkpoints


        # Early Stopping
        self.use_early_stopping = True  # Enable early stopping
        self.early_stopping_patience = 10  # Epochs to wait before stopping

        # Resume Training
        self.resume_training = False  # Resume training from a checkpoint
        self.checkpoint_path = 'checkpoints/best_model.pth'  # Path to the checkpoint file
        self.start_epoch = 1  # Starting epoch (useful when resuming)

        # ----------------------------
        # Logging and Checkpointing
        # ----------------------------
        self.logs_dir = 'log_dir/'  # Directory to save log files
        self.tensorboard_log_dir = 'log_dir/tensorboard/'  # TensorBoard log directory
        self.log_interval = 5  # How often to log training status (in batches)
        self.checkpoint_dir = 'checkpoints/'  # Directory to save model checkpoints
        self.save_best_only = True  # Whether to save only the best model based on validation loss

        # ----------------------------
        # Miscellaneous Settings
        # ----------------------------
        # For CuDNN determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_seed(self, seed: int = 42):
        """
        Sets the random seed for reproducibility.

        Args:
            seed (int): The seed value to set.
        """
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For CuDNN determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
