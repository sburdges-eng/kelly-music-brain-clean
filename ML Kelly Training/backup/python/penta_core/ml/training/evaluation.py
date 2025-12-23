"""
Evaluation and Validation Algorithms for Kelly ML.

Provides comprehensive evaluation tools:
- Music-specific metrics (emotion accuracy, groove consistency)
- Cross-validation utilities
- Confusion matrix analysis
- Calibration metrics
- A/B testing framework

Usage:
    from python.penta_core.ml.training.evaluation import evaluate_model, MusicMetrics
    
    metrics = MusicMetrics()
    results = evaluate_model(model, test_loader, metrics)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Metric Classes
# =============================================================================

@dataclass
class MetricResult:
    """Container for metric results."""
    name: str
    value: float
    std: Optional[float] = None
    per_class: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseMetrics:
    """Base class for metric computation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self._predictions = []
        self._targets = []
        self._probabilities = []
    
    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ):
        """Add batch of predictions."""
        self._predictions.append(predictions)
        self._targets.append(targets)
        if probabilities is not None:
            self._probabilities.append(probabilities)
    
    def compute(self) -> Dict[str, MetricResult]:
        """Compute all metrics. Override in subclasses."""
        raise NotImplementedError


class MusicMetrics(BaseMetrics):
    """
    Comprehensive metrics for music ML models.
    
    Includes:
    - Classification metrics (accuracy, F1, precision, recall)
    - Regression metrics (MSE, MAE, R²)
    - Music-specific metrics (emotional distance, harmonic accuracy)
    """
    
    def __init__(
        self,
        task: str = "classification",
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.class_names = class_names
    
    def compute(self) -> Dict[str, MetricResult]:
        """Compute all metrics."""
        predictions = np.concatenate(self._predictions, axis=0)
        targets = np.concatenate(self._targets, axis=0)
        
        results = {}
        
        if self.task == "classification":
            results.update(self._compute_classification_metrics(predictions, targets))
            
            if self._probabilities:
                probs = np.concatenate(self._probabilities, axis=0)
                results.update(self._compute_calibration_metrics(probs, targets))
        else:
            results.update(self._compute_regression_metrics(predictions, targets))
        
        return results
    
    def _compute_classification_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, MetricResult]:
        """Compute classification metrics."""
        results = {}
        
        # Accuracy
        accuracy = np.mean(predictions == targets)
        results["accuracy"] = MetricResult("accuracy", accuracy)
        
        # Per-class metrics
        if self.num_classes:
            per_class_acc = {}
            per_class_f1 = {}
            
            for c in range(self.num_classes):
                mask = targets == c
                if mask.sum() > 0:
                    class_name = self.class_names[c] if self.class_names else str(c)
                    
                    # Class accuracy
                    class_acc = np.mean(predictions[mask] == targets[mask])
                    per_class_acc[class_name] = class_acc
                    
                    # F1 score
                    tp = np.sum((predictions == c) & (targets == c))
                    fp = np.sum((predictions == c) & (targets != c))
                    fn = np.sum((predictions != c) & (targets == c))
                    
                    precision = tp / (tp + fp + 1e-10)
                    recall = tp / (tp + fn + 1e-10)
                    f1 = 2 * precision * recall / (precision + recall + 1e-10)
                    per_class_f1[class_name] = f1
            
            results["per_class_accuracy"] = MetricResult(
                "per_class_accuracy",
                np.mean(list(per_class_acc.values())),
                per_class=per_class_acc,
            )
            
            # Macro F1
            macro_f1 = np.mean(list(per_class_f1.values()))
            results["macro_f1"] = MetricResult(
                "macro_f1",
                macro_f1,
                per_class=per_class_f1,
            )
        
        return results
    
    def _compute_regression_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, MetricResult]:
        """Compute regression metrics."""
        results = {}
        
        # MSE
        mse = np.mean((predictions - targets) ** 2)
        results["mse"] = MetricResult("mse", mse)
        
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        results["mae"] = MetricResult("mae", mae)
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        results["r2"] = MetricResult("r2", r2)
        
        # RMSE
        rmse = np.sqrt(mse)
        results["rmse"] = MetricResult("rmse", rmse)
        
        return results
    
    def _compute_calibration_metrics(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, MetricResult]:
        """Compute calibration metrics (ECE, MCE)."""
        results = {}
        
        # Expected Calibration Error
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        accuracies = (predictions == targets).astype(float)
        
        ece = 0.0
        mce = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(accuracies[in_bin])
                
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
                mce = max(mce, np.abs(avg_accuracy - avg_confidence))
        
        results["ece"] = MetricResult("expected_calibration_error", ece)
        results["mce"] = MetricResult("max_calibration_error", mce)
        
        return results


class EmotionMetrics(MusicMetrics):
    """
    Metrics specific to emotion recognition.
    
    Includes emotional distance-aware metrics.
    """
    
    # Emotion coordinates in valence-arousal space
    EMOTION_COORDS = {
        "happy": (0.8, 0.6),
        "sad": (-0.6, -0.4),
        "angry": (-0.6, 0.7),
        "fear": (-0.7, 0.5),
        "surprise": (0.3, 0.8),
        "disgust": (-0.5, 0.3),
        "neutral": (0.0, 0.0),
    }
    
    def __init__(self, class_names: Optional[List[str]] = None):
        if class_names is None:
            class_names = list(self.EMOTION_COORDS.keys())
        super().__init__(
            task="classification",
            num_classes=len(class_names),
            class_names=class_names,
        )
    
    def compute(self) -> Dict[str, MetricResult]:
        """Compute emotion-specific metrics."""
        results = super().compute()
        
        predictions = np.concatenate(self._predictions, axis=0)
        targets = np.concatenate(self._targets, axis=0)
        
        # Emotional distance error
        ede = self._compute_emotional_distance_error(predictions, targets)
        results["emotional_distance_error"] = MetricResult(
            "emotional_distance_error",
            ede,
        )
        
        # Valence-arousal accuracy (within threshold)
        va_acc = self._compute_va_accuracy(predictions, targets, threshold=0.5)
        results["va_accuracy"] = MetricResult("va_accuracy", va_acc)
        
        return results
    
    def _compute_emotional_distance_error(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Compute mean emotional distance in VA space."""
        total_distance = 0.0
        
        for pred, target in zip(predictions, targets):
            pred_name = self.class_names[pred]
            target_name = self.class_names[target]
            
            pred_coord = self.EMOTION_COORDS.get(pred_name, (0, 0))
            target_coord = self.EMOTION_COORDS.get(target_name, (0, 0))
            
            distance = np.sqrt(
                (pred_coord[0] - target_coord[0]) ** 2 +
                (pred_coord[1] - target_coord[1]) ** 2
            )
            total_distance += distance
        
        return total_distance / len(predictions)
    
    def _compute_va_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """Compute accuracy within VA distance threshold."""
        correct = 0
        
        for pred, target in zip(predictions, targets):
            pred_name = self.class_names[pred]
            target_name = self.class_names[target]
            
            pred_coord = self.EMOTION_COORDS.get(pred_name, (0, 0))
            target_coord = self.EMOTION_COORDS.get(target_name, (0, 0))
            
            distance = np.sqrt(
                (pred_coord[0] - target_coord[0]) ** 2 +
                (pred_coord[1] - target_coord[1]) ** 2
            )
            
            if distance <= threshold:
                correct += 1
        
        return correct / len(predictions)


class GrooveMetrics(MusicMetrics):
    """
    Metrics specific to groove/timing prediction.
    """
    
    def __init__(self):
        super().__init__(task="regression")
    
    def compute(self) -> Dict[str, MetricResult]:
        """Compute groove-specific metrics."""
        results = super().compute()
        
        predictions = np.concatenate(self._predictions, axis=0)
        targets = np.concatenate(self._targets, axis=0)
        
        # Timing accuracy (within threshold)
        timing_acc = self._compute_timing_accuracy(predictions, targets, threshold_ms=10)
        results["timing_accuracy"] = MetricResult("timing_accuracy_10ms", timing_acc)
        
        # Groove consistency
        consistency = self._compute_groove_consistency(predictions)
        results["groove_consistency"] = MetricResult("groove_consistency", consistency)
        
        return results
    
    def _compute_timing_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold_ms: float = 10,
    ) -> float:
        """Compute percentage of predictions within timing threshold."""
        # Assuming predictions are in milliseconds
        errors = np.abs(predictions - targets)
        return np.mean(errors < threshold_ms)
    
    def _compute_groove_consistency(self, predictions: np.ndarray) -> float:
        """Compute consistency of groove patterns."""
        if predictions.ndim == 1:
            return 1.0
        
        # Measure smoothness of predictions
        diffs = np.diff(predictions, axis=-1)
        smoothness = 1.0 / (1.0 + np.std(diffs))
        
        return smoothness


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(
    model: "nn.Module",
    dataloader: "torch.utils.data.DataLoader",
    metrics: BaseMetrics,
    device: Optional[str] = None,
) -> Dict[str, MetricResult]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: Test data loader
        metrics: Metrics object
        device: Device to use
    
    Returns:
        Dict of metric results
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for evaluation")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    metrics.reset()
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get predictions
            if outputs.dim() > 1 and outputs.size(1) > 1:
                # Classification
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
            else:
                # Regression
                probs = None
                preds = outputs.cpu().numpy()
            
            if targets is not None:
                targets_np = targets.cpu().numpy()
                metrics.update(preds, targets_np, probs)
    
    return metrics.compute()


def cross_validate(
    model_class: type,
    dataset: "torch.utils.data.Dataset",
    model_kwargs: Dict[str, Any],
    metrics: BaseMetrics,
    n_folds: int = 5,
    epochs: int = 10,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model_class: Model class to instantiate
        dataset: Full dataset
        model_kwargs: Arguments for model constructor
        metrics: Metrics object
        n_folds: Number of folds
        epochs: Training epochs per fold
        batch_size: Batch size
        device: Device to use
    
    Returns:
        Dict with mean and std for each metric
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for cross-validation")
    
    from torch.utils.data import DataLoader, Subset
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_folds
    
    all_results = {name: [] for name in ["accuracy", "macro_f1", "mse", "mae"]}
    
    for fold in range(n_folds):
        logger.info(f"Fold {fold + 1}/{n_folds}")
        
        # Split indices
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        # Create data loaders
        train_dataset = Subset(dataset, train_indices.tolist())
        val_dataset = Subset(dataset, val_indices.tolist())
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Train model
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        fold_results = evaluate_model(model, val_loader, metrics, device)
        
        for name, result in fold_results.items():
            if name in all_results:
                all_results[name].append(result.value)
    
    # Compute mean and std
    summary = {}
    for name, values in all_results.items():
        if values:
            summary[name] = {
                "mean": np.mean(values),
                "std": np.std(values),
            }
    
    return summary


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices
        targets: True class indices
        num_classes: Number of classes
        normalize: Normalize by row (true labels)
    
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.float32)
    
    for pred, target in zip(predictions, targets):
        cm[target, pred] += 1
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / (row_sums + 1e-10)
    
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
) -> Optional[Any]:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Class names
        output_path: Optional path to save figure
        title: Plot title
    
    Returns:
        Matplotlib figure if available
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn required for plotting")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved confusion matrix: {output_path}")
    
    return fig


# =============================================================================
# Validation Utilities
# =============================================================================

class ModelValidator:
    """
    Comprehensive model validation framework.
    
    Performs:
    - Performance evaluation
    - Robustness testing
    - Latency measurement
    - Memory profiling
    """
    
    def __init__(
        self,
        model: "nn.Module",
        device: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def validate_performance(
        self,
        dataloader: "torch.utils.data.DataLoader",
        metrics: BaseMetrics,
    ) -> Dict[str, MetricResult]:
        """Evaluate model performance."""
        return evaluate_model(self.model, dataloader, metrics, self.device)
    
    def validate_robustness(
        self,
        dataloader: "torch.utils.data.DataLoader",
        perturbations: List[Callable],
    ) -> Dict[str, float]:
        """Test model robustness to perturbations."""
        results = {}
        
        self.model.eval()
        
        for perturb_fn in perturbations:
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Apply perturbation
                    perturbed = perturb_fn(inputs)
                    
                    outputs = self.model(perturbed)
                    preds = outputs.argmax(dim=1)
                    
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            
            results[perturb_fn.__name__] = correct / total
        
        return results
    
    def measure_latency(
        self,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Measure inference latency."""
        import time
        
        self.model.eval()
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
        
        # Synchronize if CUDA
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "p95_ms": np.percentile(latencies, 95),
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
    
    def full_validation(
        self,
        dataloader: "torch.utils.data.DataLoader",
        metrics: BaseMetrics,
        input_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        """Run full validation suite."""
        results = {}
        
        # Performance
        perf_results = self.validate_performance(dataloader, metrics)
        results["performance"] = {k: v.value for k, v in perf_results.items()}
        
        # Latency
        results["latency"] = self.measure_latency(input_shape)
        
        # Parameters
        results["parameters"] = self.count_parameters()
        
        return results

