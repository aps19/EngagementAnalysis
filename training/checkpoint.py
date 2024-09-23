# training/checkpoint.py

import os
import torch
import logging

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, filename='checkpoint.pth'):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer whose state to save.
        epoch (int): The current epoch number.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    # Prepare the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, config):
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        model (nn.Module): The model to load the state into.
        optimizer (Optimizer): The optimizer to load the state into.
        config (Config): Configuration object containing the checkpoint path.

    Returns:
        int: The epoch number to resume training from.
    """
    checkpoint_path = config.checkpoint_path
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        logging.info(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {start_epoch}")
        return start_epoch
    else:
        logging.error(f"Checkpoint file not found at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
