import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from training.evaluator import Evaluator
from utils.metrics import calculate_accuracy, calculate_f1_score
from training.checkpoint import save_checkpoint, load_checkpoint

class Trainer:
    """
    Trainer class to handle the training loop.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        config (Config): Configuration object with training parameters.
    """

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler (optional)
        if self.config.use_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        else:
            self.scheduler = None

        # Evaluator for validation
        self.evaluator = Evaluator(self.model, self.val_loader, self.config, self)

        # Best validation loss
        self.best_val_loss = float('inf')

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.config.tensorboard_log_dir)

        # Early stopping counter
        self.early_stopping_counter = 0

        # Mixed precision scaler (optional)
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Gradient accumulation setting
        self.accumulation_steps = 4  # Adjust this based on your GPU memory

        # Load checkpoint if resume training is enabled
        if self.config.resume_training and os.path.exists(self.config.checkpoint_path):
            load_checkpoint(self.model, self.optimizer, self.config)
            logging.info(f"Resumed training from checkpoint: {self.config.checkpoint_path}")

    def train(self):
        """
        Runs the training loop over the specified number of epochs.
        """
        for epoch in range(self.config.start_epoch, self.config.num_epochs + 1):
            logging.info(f"Epoch {epoch}/{self.config.num_epochs}")
            self.model.train()
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", ncols=80)

            for batch_idx, (inputs, labels, seq_lengths) in enumerate(progress_bar):
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)
                seq_lengths = seq_lengths.to(self.config.device)

                # Zero gradients only after accumulation steps
                if batch_idx % self.accumulation_steps == 0:
                    self.optimizer.zero_grad()

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    outputs = self.model(inputs, seq_lengths)
                    loss = self.criterion(outputs, labels)

                # Normalize loss for accumulation
                loss = loss / self.accumulation_steps

                # Backward pass and optimization
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_val
                    )
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_val
                    )
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()

                # Update statistics
                batch_loss = loss.item() * self.accumulation_steps
                epoch_loss += batch_loss * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                batch_accuracy = 100.0 * correct_predictions / total_samples

                # Update progress bar
                progress_bar.set_postfix(
                    loss=batch_loss,
                    acc=batch_accuracy,
                )

                # Log to TensorBoard
                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Batch_Loss', batch_loss, global_step)
                self.writer.add_scalar('Train/Batch_Accuracy', batch_accuracy, global_step)

                # Log at specified intervals
                if (batch_idx + 1) % self.config.log_interval == 0:
                    logging.info(
                        f"Epoch [{epoch}], Batch [{batch_idx + 1}/{len(self.train_loader)}], "
                        f"Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.2f}%"
                    )

            # Compute epoch metrics
            epoch_loss /= len(self.train_loader.dataset)
            epoch_accuracy = 100.0 * correct_predictions / total_samples

            logging.info(
                f"Epoch {epoch} Training Loss: {epoch_loss:.4f}, "
                f"Training Accuracy: {epoch_accuracy:.2f}%"
            )

            # Log epoch metrics to TensorBoard
            self.writer.add_scalar('Train/Epoch_Loss', epoch_loss, epoch)
            self.writer.add_scalar('Train/Epoch_Accuracy', epoch_accuracy, epoch)

            # Validation
            val_loss, val_accuracy, val_f1_score = self.evaluator.evaluate(epoch)
            logging.info(
                f"Epoch {epoch} Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {val_accuracy:.2f}%, "
                f"Validation F1 Score: {val_f1_score:.4f}"
            )

            # Early stopping check
            if self.config.use_early_stopping:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    # Save best model
                    save_checkpoint(
                        self.model, self.optimizer, epoch, self.config.checkpoint_dir, 'best_model.pth'
                    )
                    logging.info("Best model saved.")
                else:
                    self.early_stopping_counter += 1
                    logging.info(f"No improvement in validation loss for {self.early_stopping_counter} epoch(s).")
                    if self.early_stopping_counter >= self.config.early_stopping_patience:
                        logging.info("Early stopping triggered.")
                        break
            else:
                # Save checkpoint every epoch or based on validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_checkpoint(
                        self.model, self.optimizer, epoch, self.config.checkpoint_dir, 'best_model.pth'
                    )
                    logging.info("Best model saved.")

            # Step the scheduler if in use
            if self.scheduler:
                self.scheduler.step()

        self.writer.close()
        logging.info("Training completed.")
