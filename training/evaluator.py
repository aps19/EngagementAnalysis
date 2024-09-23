# training/evaluator.py

import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from utils.metrics import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
)

class Evaluator:
    """
    Evaluator class to handle the evaluation of the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation data.
        config (Config): Configuration object with evaluation parameters.
        trainer (Trainer): Trainer instance to access shared resources like the TensorBoard writer.
    """

    def __init__(self, model, val_loader, config, trainer):
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.trainer = trainer  # Access to Trainer instance
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, epoch):
        """
        Evaluates the model on the validation set.

        Args:
            epoch (int): Current epoch number (used for logging).

        Returns:
            Tuple[float, float, float]: Validation loss, accuracy, and F1 score.
        """
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation", ncols=80)
            for inputs, labels, seq_lengths in progress_bar:
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)
                seq_lengths = seq_lengths.to(self.config.device)

                # Forward pass
                outputs = self.model(inputs, seq_lengths)

                # Compute loss
                loss = self.criterion(outputs, labels)

                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # Collect labels and predictions for metrics
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Compute average loss and accuracy
        val_loss /= len(self.val_loader.dataset)
        val_accuracy = 100.0 * correct_predictions / total_samples

        # Calculate additional metrics
        val_f1_score = f1_score(all_labels, all_predictions, average='weighted')
        val_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)

        # Log metrics to TensorBoard
        self.trainer.writer.add_scalar('Validation/Loss', val_loss, epoch)
        self.trainer.writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
        self.trainer.writer.add_scalar('Validation/F1_Score', val_f1_score, epoch)
        self.trainer.writer.add_scalar('Validation/Precision', val_precision, epoch)
        self.trainer.writer.add_scalar('Validation/Recall', val_recall, epoch)

        # Automatically determine class names from labels
        unique_labels = sorted(set(all_labels))
        class_names = [str(label) for label in unique_labels]

        # Generate classification report
        class_report = classification_report(
            all_labels,
            all_predictions,
            labels=unique_labels,
            target_names=class_names,
            zero_division=0,
        )
        logging.info("Classification Report:\n" + class_report)

        return val_loss, val_accuracy, val_f1_score
