#!/usr/bin/env python3
"""
Training Utilities for Kelly MIDI Companion ML Training
========================================================
Provides early stopping, metrics tracking, checkpoint management, and evaluation.
"""

import torch
import torch.nn as nn
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better) or 'max' for accuracy (higher is better)
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (loss or accuracy)
            model: Model to save weights from

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self._restore_checkpoint(model)

        return self.early_stop

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:  # mode == 'max'
            return current > (best + self.min_delta)

    def _save_checkpoint(self, model: nn.Module):
        """Save model weights."""
        self.best_weights = model.state_dict().copy()

    def _restore_checkpoint(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class TrainingMetrics:
    """Track training and validation metrics."""

    def __init__(self):
        self.history = defaultdict(list)
        self.current_epoch = 0

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metric: Optional[float] = None,
        val_metric: Optional[float] = None,
        **kwargs
    ):
        """Update metrics for an epoch."""
        self.current_epoch = epoch
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)

        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if train_metric is not None:
            self.history['train_metric'].append(train_metric)
        if val_metric is not None:
            self.history['val_metric'].append(val_metric)

        # Store any additional metrics
        for key, value in kwargs.items():
            self.history[key].append(value)

    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """Get epoch with best metric value."""
        if metric not in self.history:
            return 0

        values = self.history[metric]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        return self.history['epoch'][best_idx]

    def save_json(self, filepath: Path):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(dict(self.history), f, indent=2)

    def save_csv(self, filepath: Path):
        """Save metrics to CSV file."""
        # Get all keys
        keys = list(self.history.keys())
        if not keys:
            return

        # Get max length
        max_len = max(len(self.history[k]) for k in keys)

        # Pad shorter lists with None
        rows = []
        for i in range(max_len):
            row = {}
            for key in keys:
                if i < len(self.history[key]):
                    row[key] = self.history[key][i]
                else:
                    row[key] = None
            rows.append(row)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def plot_curves(self, output_dir: Path, model_name: str):
        """Plot training curves."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot loss curves
        if 'train_loss' in self.history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['epoch'], self.history['train_loss'],
                    label='Train Loss', marker='o')
            if 'val_loss' in self.history:
                plt.plot(self.history['epoch'], self.history['val_loss'],
                        label='Validation Loss', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{model_name} - Loss Curves')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_dir / f'{model_name.lower()}_loss.png', dpi=150)
            plt.close()

        # Plot metric curves if available
        if 'train_metric' in self.history or 'val_metric' in self.history:
            plt.figure(figsize=(10, 6))
            if 'train_metric' in self.history:
                plt.plot(self.history['epoch'], self.history['train_metric'],
                        label='Train Metric', marker='o')
            if 'val_metric' in self.history:
                plt.plot(self.history['epoch'], self.history['val_metric'],
                        label='Validation Metric', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.title(f'{model_name} - Metric Curves')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_dir / f'{model_name.lower()}_metric.png', dpi=150)
            plt.close()


class CheckpointManager:
    """Manage model checkpoints and resume training."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: TrainingMetrics,
        model_name: str,
        is_best: bool = False
    ):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': dict(metrics.history),
            'model_name': model_name
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'{model_name.lower()}_latest.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f'{model_name.lower()}_best.pt'
            torch.save(checkpoint, best_path)

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        model_name: str,
        resume_from: str = 'latest'
    ) -> Tuple[int, TrainingMetrics]:
        """
        Load checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            model_name: Name of the model
            resume_from: 'latest' or 'best'

        Returns:
            Tuple of (epoch, metrics)
        """
        checkpoint_path = self.checkpoint_dir / f'{model_name.lower()}_{resume_from}.pt'

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        metrics = TrainingMetrics()
        metrics.history = defaultdict(list, checkpoint.get('metrics', {}))

        epoch = checkpoint['epoch']

        return epoch, metrics


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cpu',
    metric_fn: Optional[callable] = None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to run on
        metric_fn: Optional function to compute additional metrics

    Returns:
        Dictionary with 'loss' and optionally 'metric'
    """
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            # Get inputs and targets (adapt based on model type)
            if 'mel_features' in batch:
                inputs = batch['mel_features'].to(device)
                targets = batch['emotion'].to(device)
            elif 'emotion' in batch:
                inputs = batch['emotion'].to(device)
                targets = batch['notes'].to(device)
            elif 'context' in batch:
                inputs = batch['context'].to(device)
                targets = batch['chords'].to(device)
            else:
                # Generic fallback
                inputs = batch[list(batch.keys())[0]].to(device)
                targets = batch[list(batch.keys())[1]].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            if metric_fn is not None:
                metric = metric_fn(outputs, targets)
                total_metric += metric

            num_batches += 1

    results = {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0
    }

    if metric_fn is not None:
        results['metric'] = total_metric / num_batches if num_batches > 0 else 0.0

    return results


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute accuracy for binary/multi-label classification."""
    if outputs.dim() > 1 and outputs.size(1) > 1:
        # Multi-class: use argmax
        preds = outputs.argmax(dim=1)
        correct = (preds == targets.argmax(dim=1) if targets.dim() > 1 else preds == targets).float()
    else:
        # Binary: use threshold
        preds = (outputs > threshold).float()
        correct = (preds == targets).float()

    return correct.mean().item()


def compute_cosine_similarity(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute cosine similarity between outputs and targets."""
    # Flatten if needed
    outputs_flat = outputs.view(outputs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    # Normalize
    outputs_norm = torch.nn.functional.normalize(outputs_flat, p=2, dim=1)
    targets_norm = torch.nn.functional.normalize(targets_flat, p=2, dim=1)

    # Cosine similarity
    similarity = (outputs_norm * targets_norm).sum(dim=1).mean()

    return similarity.item()
