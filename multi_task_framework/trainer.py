"""Training utilities for multi-task models"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import logging


class MultiTaskTrainer:
    """Trainer for multi-task learning models"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_balancer,
        device: str = "cpu",
        logger: Optional[logging.Logger] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_balancer = loss_balancer
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.train_history = {
            "total_loss": [],
            "task_losses": {}
        }

    def train_step(
        self,
        batch: Tuple[Dict, Dict],
        balance_method: str = "static"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Single training step

        Args:
            batch: Tuple of (inputs, targets) dicts
            balance_method: Loss balancing method

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        inputs, targets = batch

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        targets = {k: v.to(self.device) for k, v in targets.items()}

        # Forward pass
        self.optimizer.zero_grad()
        predictions = self.model(inputs)

        # Compute task losses
        task_losses = self.model.compute_losses(predictions, targets)

        # Balance and combine losses
        total_loss, weighted_losses = self.loss_balancer(
            task_losses, method=balance_method
        )

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), weighted_losses

    def train_epoch(
        self,
        train_loader: DataLoader,
        balance_method: str = "static"
    ) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            balance_method: Loss balancing method

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        task_losses = {task: 0.0 for task in self.model.active_tasks}

        for batch in train_loader:
            loss, weighted = self.train_step(batch, balance_method)
            total_loss += loss
            num_batches += 1

            for task, task_loss in weighted.items():
                task_losses[task] += task_loss

        # Average metrics
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {
            task: loss / num_batches for task, loss in task_losses.items()
        }

        # Log metrics
        self.train_history["total_loss"].append(avg_total_loss)
        for task, loss in avg_task_losses.items():
            if task not in self.train_history["task_losses"]:
                self.train_history["task_losses"][task] = []
            self.train_history["task_losses"][task].append(loss)

        return {"total_loss": avg_total_loss, **avg_task_losses}

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        balance_method: str = "static"
    ) -> Dict[str, float]:
        """
        Evaluate model

        Args:
            eval_loader: Evaluation data loader
            balance_method: Loss balancing method

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        task_losses = {task: 0.0 for task in self.model.active_tasks}

        for batch in eval_loader:
            inputs, targets = batch

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}

            # Forward pass
            predictions = self.model(inputs)

            # Compute losses
            losses = self.model.compute_losses(predictions, targets)
            combined_loss, weighted = self.loss_balancer(
                losses, method=balance_method
            )

            total_loss += combined_loss.item()
            num_batches += 1

            for task, task_loss in weighted.items():
                task_losses[task] += task_loss

        # Average metrics
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {
            task: loss / num_batches for task, loss in task_losses.items()
        }

        return {"total_loss": avg_total_loss, **avg_task_losses}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        balance_method: str = "static",
        early_stopping_patience: int = 5
    ):
        """
        Train model for multiple epochs

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            balance_method: Loss balancing method
            early_stopping_patience: Early stopping patience
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, balance_method)

            # Validation
            if val_loader:
                val_metrics = self.evaluate(val_loader, balance_method)
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f} - "
                    f"Val Loss: {val_metrics['total_loss']:.4f}"
                )

                # Early stopping
                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info(
                            f"Early stopping at epoch {epoch + 1}"
                        )
                        break
            else:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}"
                )

    def get_history(self) -> Dict:
        """Get training history"""
        return self.train_history

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.train_history
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.train_history = checkpoint["history"]
