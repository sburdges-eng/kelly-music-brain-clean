#!/usr/bin/env python3
"""
Kelly NVIDIA CUDA Training Module

Unified training pipeline for all Kelly ML models using CUDA acceleration.

Integrates:
- Static datasets (chords, scales, emotions, theory)
- Downloaded audio samples (Freesound)
- Generated MIDI sequences (chord generators)
- ML OSC patterns
- MIDI analysis features

Supports:
- NVIDIA CUDA (primary)
- Apple MPS (fallback)
- CPU (fallback)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Kelly ML model types - aligned with models/registry.json"""
    # Core 7 models from registry
    EMOTION_RECOGNIZER = "emotion_recognizer"
    MELODY_TRANSFORMER = "melody_transformer"
    HARMONY_PREDICTOR = "harmony_predictor"
    DYNAMICS_ENGINE = "dynamics_engine"
    GROOVE_PREDICTOR = "groove_predictor"
    INSTRUMENT_RECOGNIZER = "instrument_recognizer"
    EMOTION_NODE_CLASSIFIER = "emotion_node_classifier"

    # Additional ML modules
    OSC_PATTERN_LEARNER = "osc_pattern_learner"
    STYLE_TRANSFER = "style_transfer"
    CHORD_PREDICTOR = "chord_predictor"
    NEURAL_VOICE = "neural_voice"

    # Audio analysis models
    AUDIO_CLASSIFIER = "audio_classifier"
    BEAT_TRACKER = "beat_tracker"
    KEY_DETECTOR = "key_detector"
    ONSET_DETECTOR = "onset_detector"
    PITCH_TRACKER = "pitch_tracker"
    SOURCE_SEPARATOR = "source_separator"
    TEMPO_ESTIMATOR = "tempo_estimator"
    TIMBRE_ENCODER = "timbre_encoder"
    VOICE_ACTIVITY_DETECTOR = "voice_activity_detector"


# SSD Storage paths
SSD_PATHS = {
    "darwin": "/Volumes/Extreme SSD/kelly-audio-data",
    "linux": "/mnt/ssd/kelly-audio-data",
    "windows": "D:\\kelly-audio-data",
}


def get_ssd_data_path() -> Optional[Path]:
    """Get SSD data path if available."""
    import platform
    system = platform.system().lower()
    ssd_path = Path(SSD_PATHS.get(system, SSD_PATHS["linux"]))

    # Check environment override
    env_path = os.environ.get("KELLY_SSD_PATH")
    if env_path:
        ssd_path = Path(env_path)

    if ssd_path.exists():
        return ssd_path
    return None


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_type: ModelType
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10

    # CUDA specific
    use_cuda: bool = True
    cuda_device: int = 0
    mixed_precision: bool = True  # FP16 training
    gradient_accumulation_steps: int = 1

    # Data paths
    data_dir: Optional[Path] = None  # Auto-detect SSD if None
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every_n_epochs: int = 5

    # Logging
    log_every_n_steps: int = 100
    use_tensorboard: bool = True
    use_wandb: bool = False

    # Model-specific architecture params (from registry)
    arch_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-detect SSD path if not specified."""
        if self.data_dir is None:
            ssd_path = get_ssd_data_path()
            if ssd_path:
                self.data_dir = ssd_path
                logger.info(f"Using SSD data path: {ssd_path}")


# Model architecture specifications from registry
MODEL_SPECS = {
    ModelType.EMOTION_RECOGNIZER: {
        "input_size": 128, "output_size": 64,
        "arch": "128→512→256→128→64", "params": "~500K"
    },
    ModelType.MELODY_TRANSFORMER: {
        "input_size": 64, "output_size": 128,
        "arch": "64→256→256→256→128", "params": "~400K"
    },
    ModelType.HARMONY_PREDICTOR: {
        "input_size": 128, "output_size": 64,
        "arch": "128→256→128→64", "params": "~100K"
    },
    ModelType.DYNAMICS_ENGINE: {
        "input_size": 32, "output_size": 16,
        "arch": "32→128→64→16", "params": "~20K"
    },
    ModelType.GROOVE_PREDICTOR: {
        "input_size": 64, "output_size": 32,
        "arch": "64→128→64→32", "params": "~25K"
    },
    ModelType.INSTRUMENT_RECOGNIZER: {
        "input_size": 128, "output_size": 160,
        "arch": "CNN:64→128→256→512", "params": "~2M"
    },
    ModelType.EMOTION_NODE_CLASSIFIER: {
        "input_size": 128, "output_size": 258,
        "arch": "CNN+Multi-head", "params": "~3M"
    },
    ModelType.STYLE_TRANSFER: {
        "input_size": 64, "output_size": 64,
        "arch": "VAE/CycleGAN", "params": "~1M"
    },
    ModelType.CHORD_PREDICTOR: {
        "input_size": 128, "output_size": 128,
        "arch": "LSTM:256→256→128", "params": "~800K"
    },
    ModelType.NEURAL_VOICE: {
        "input_size": 256, "output_size": 512,
        "arch": "DiffSinger", "params": "~10M"
    },
    # Voice synthesis sub-models
    ModelType.AUDIO_CLASSIFIER: {
        "input_size": 128, "output_size": 10,
        "arch": "CNN:64→128→256", "params": "~500K"
    },
    ModelType.BEAT_TRACKER: {
        "input_size": 128, "output_size": 2,
        "arch": "TCN:64→128→256", "params": "~300K"
    },
    ModelType.KEY_DETECTOR: {
        "input_size": 12, "output_size": 24,
        "arch": "CNN+LSTM", "params": "~200K"
    },
    ModelType.ONSET_DETECTOR: {
        "input_size": 128, "output_size": 1,
        "arch": "CNN:64→128", "params": "~150K"
    },
    ModelType.PITCH_TRACKER: {
        "input_size": 128, "output_size": 360,
        "arch": "CREPE-style", "params": "~800K"
    },
    ModelType.SOURCE_SEPARATOR: {
        "input_size": 2048, "output_size": 2048,
        "arch": "U-Net/Demucs", "params": "~50M"
    },
    ModelType.TEMPO_ESTIMATOR: {
        "input_size": 128, "output_size": 300,
        "arch": "CNN+Dense", "params": "~400K"
    },
    ModelType.TIMBRE_ENCODER: {
        "input_size": 128, "output_size": 256,
        "arch": "VAE encoder", "params": "~1M"
    },
    ModelType.VOICE_ACTIVITY_DETECTOR: {
        "input_size": 64, "output_size": 1,
        "arch": "RNN:128→64", "params": "~100K"
    },
}


@dataclass
class TrainingResult:
    """Training result metrics."""
    model_type: ModelType
    best_epoch: int
    best_val_loss: float
    final_metrics: Dict[str, float]
    training_time_seconds: float
    device_used: str
    checkpoint_path: Optional[Path] = None


class CUDATrainer:
    """NVIDIA CUDA-accelerated trainer for Kelly ML models."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision

    def _setup_device(self) -> str:
        """Setup compute device (CUDA/MPS/CPU)."""
        try:
            import torch

            if self.config.use_cuda and torch.cuda.is_available():
                device = f"cuda:{self.config.cuda_device}"
                gpu_name = torch.cuda.get_device_name(self.config.cuda_device)
                gpu_memory = torch.cuda.get_device_properties(self.config.cuda_device).total_memory / 1e9
                logger.info(f"Using CUDA: {gpu_name} ({gpu_memory:.1f}GB)")

                # Enable TF32 for Ampere+ GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                return device
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Using Apple MPS (Metal Performance Shaders)")
                return "mps"
            else:
                logger.warning("No GPU available, falling back to CPU")
                return "cpu"
        except ImportError:
            logger.error("PyTorch not installed. Install with: pip install torch")
            return "cpu"

    def _build_model(self, input_shape: Tuple[int, ...], num_classes: int):
        """Build model architecture based on type."""
        try:
            import torch
            import torch.nn as nn

            # Import Kelly architectures
            from python.penta_core.ml.training.architectures import (
                EmotionCNN, MelodyLSTM, HarmonyMLP, MultiTaskModel
            )
        except ImportError:
            logger.error("Could not import model architectures")
            return None

        model_builders = {
            ModelType.EMOTION_RECOGNIZER: lambda: EmotionCNN(
                input_channels=input_shape[0] if len(input_shape) > 1 else 1,
                num_classes=num_classes
            ),
            ModelType.MELODY_TRANSFORMER: lambda: MelodyLSTM(
                input_size=input_shape[-1],
                hidden_size=256,
                num_layers=3,
                num_classes=num_classes
            ),
            ModelType.HARMONY_PREDICTOR: lambda: HarmonyMLP(
                input_size=input_shape[-1],
                hidden_sizes=[512, 256, 128],
                num_classes=num_classes
            ),
            ModelType.OSC_PATTERN_LEARNER: lambda: MelodyLSTM(
                input_size=input_shape[-1],
                hidden_size=128,
                num_layers=2,
                num_classes=num_classes
            ),
        }

        builder = model_builders.get(self.config.model_type)
        if builder:
            return builder()

        # Default: Simple MLP
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(input_shape))), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def prepare_data(self, dataset_path: Path) -> Tuple[Any, Any, Any]:
        """Prepare data loaders from unified dataset."""
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset, random_split
            import numpy as np
        except ImportError:
            logger.error("PyTorch not installed")
            return None, None, None

        # Load unified dataset
        with open(dataset_path) as f:
            data = json.load(f)

        samples = data.get("samples", [])
        if not samples:
            logger.error(f"No samples found in {dataset_path}")
            return None, None, None

        # Convert to tensors based on model type
        X, y = self._samples_to_tensors(samples)

        if X is None:
            return None, None, None

        # Create dataset
        dataset = TensorDataset(X, y)

        # Split
        total = len(dataset)
        train_size = int(total * self.config.train_split)
        val_size = int(total * self.config.val_split)
        test_size = total - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Create loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if "cuda" in self.device else False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True if "cuda" in self.device else False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )

        logger.info(f"Data split: train={train_size}, val={val_size}, test={test_size}")
        return train_loader, val_loader, test_loader

    def _samples_to_tensors(self, samples: List[Dict]) -> Tuple[Any, Any]:
        """Convert dataset samples to PyTorch tensors."""
        try:
            import torch
            import numpy as np
        except ImportError:
            return None, None

        # Filter samples by model type relevance
        model_data_map = {
            ModelType.EMOTION_RECOGNIZER: ["emotions", "audio_samples"],
            ModelType.HARMONY_PREDICTOR: ["chord_progressions", "scales"],
            ModelType.GROOVE_PREDICTOR: ["grooves"],
            ModelType.OSC_PATTERN_LEARNER: ["osc_patterns"],
            ModelType.MELODY_TRANSFORMER: ["midi_sequences", "scales"],
        }

        relevant_types = model_data_map.get(self.config.model_type, [])
        filtered = [s for s in samples if s.get("dataset_type") in relevant_types]

        if not filtered:
            # Use all samples
            filtered = samples

        # Encode features and labels
        X_list = []
        y_list = []
        label_encoder = {}

        for sample in filtered:
            # Extract features from data
            data = sample.get("data", {})
            features = self._extract_features(data)
            X_list.append(features)

            # Extract labels
            labels = sample.get("labels", {})
            label_key = list(labels.keys())[0] if labels else "unknown"
            label_val = labels.get(label_key, "unknown")

            if label_val not in label_encoder:
                label_encoder[label_val] = len(label_encoder)
            y_list.append(label_encoder[label_val])

        # Pad/truncate to uniform size
        max_len = max(len(x) for x in X_list)
        X_padded = []
        for x in X_list:
            if len(x) < max_len:
                x = np.pad(x, (0, max_len - len(x)))
            else:
                x = x[:max_len]
            X_padded.append(x)

        X = torch.tensor(np.array(X_padded), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.long)

        logger.info(f"Prepared {len(X)} samples, {len(label_encoder)} classes")
        return X, y

    def _extract_features(self, data: Dict) -> np.ndarray:
        """Extract numerical features from sample data."""
        import numpy as np

        features = []

        def flatten(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    flatten(v, f"{prefix}{k}_")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    if isinstance(v, (int, float)):
                        features.append(float(v))
                    elif isinstance(v, str):
                        # Hash strings to numeric
                        features.append(float(hash(v) % 1000) / 1000)
            elif isinstance(obj, (int, float)):
                features.append(float(obj))
            elif isinstance(obj, str):
                features.append(float(hash(obj) % 1000) / 1000)

        flatten(data)

        # Ensure minimum length
        if len(features) < 16:
            features.extend([0.0] * (16 - len(features)))

        return np.array(features, dtype=np.float32)

    def train(self, train_loader, val_loader) -> TrainingResult:
        """Run training loop with CUDA acceleration."""
        try:
            import torch
            import torch.nn as nn
            from torch.cuda.amp import GradScaler, autocast
        except ImportError:
            logger.error("PyTorch not installed")
            return None

        # Get sample batch to determine input shape
        sample_batch = next(iter(train_loader))
        input_shape = sample_batch[0].shape[1:]
        num_classes = len(torch.unique(sample_batch[1]))

        # Build model
        self.model = self._build_model(input_shape, num_classes)
        if self.model is None:
            return None

        self.model = self.model.to(self.device)
        logger.info(f"Model: {self.model.__class__.__name__} on {self.device}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs
        )

        # Mixed precision scaler
        if self.config.mixed_precision and "cuda" in self.device:
            self.scaler = GradScaler()

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training tracking
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                if self.scaler:
                    with autocast():
                        outputs = self.model(X)
                        loss = criterion(outputs, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += y.size(0)
                train_correct += predicted.eq(y).sum().item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += y.size(0)
                    val_correct += predicted.eq(y).sum().item()

            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total

            # Step scheduler
            self.scheduler.step()

            # Logging
            if epoch % self.config.log_every_n_steps == 0 or epoch == self.config.epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
                )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # Save best model
                if self.config.checkpoint_dir:
                    self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        training_time = time.time() - start_time

        return TrainingResult(
            model_type=self.config.model_type,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            final_metrics={
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            training_time_seconds=training_time,
            device_used=self.device,
            checkpoint_path=self.config.checkpoint_dir / f"{self.config.model_type.value}_best.pt"
        )

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        import torch

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.checkpoint_dir / f"{self.config.model_type.value}_best.pt"

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config.__dict__,
        }, path)

    def evaluate(self, test_loader) -> Dict[str, float]:
        """Evaluate model on test set."""
        import torch

        if self.model is None:
            logger.error("No model to evaluate")
            return {}

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        accuracy = 100. * correct / total
        logger.info(f"Test Accuracy: {accuracy:.2f}%")

        return {"test_accuracy": accuracy, "test_samples": total}


def train_all_models(dataset_path: Path, output_dir: Path = Path("checkpoints")):
    """Train all Kelly models on unified dataset."""
    results = {}

    for model_type in ModelType:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type.value}")
        logger.info(f"{'='*60}")

        config = TrainingConfig(
            model_type=model_type,
            checkpoint_dir=output_dir / model_type.value,
            epochs=50,
            batch_size=32,
        )

        trainer = CUDATrainer(config)
        train_loader, val_loader, test_loader = trainer.prepare_data(dataset_path)

        if train_loader is None:
            logger.warning(f"Skipping {model_type.value}: no data")
            continue

        result = trainer.train(train_loader, val_loader)
        if result:
            test_metrics = trainer.evaluate(test_loader)
            result.final_metrics.update(test_metrics)
            results[model_type.value] = result

    return results


def main():
    """Run CUDA training."""
    import argparse

    parser = argparse.ArgumentParser(description="Kelly CUDA Training")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/unified/kelly_unified_v1.0.0.json"))
    parser.add_argument("--model", type=str, choices=[m.value for m in ModelType], default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--all", action="store_true", help="Train all models")
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        logger.info("Run: python -m python.penta_core.ml.datasets.unified_generator")
        return

    if args.all:
        results = train_all_models(args.dataset, args.output)
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        for name, result in results.items():
            print(f"{name}: val_loss={result.best_val_loss:.4f}, "
                  f"test_acc={result.final_metrics.get('test_accuracy', 0):.2f}%")
    else:
        model_type = ModelType(args.model) if args.model else ModelType.EMOTION_RECOGNIZER
        config = TrainingConfig(
            model_type=model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint_dir=args.output / model_type.value,
        )

        trainer = CUDATrainer(config)
        train_loader, val_loader, test_loader = trainer.prepare_data(args.dataset)

        if train_loader:
            result = trainer.train(train_loader, val_loader)
            if result:
                test_metrics = trainer.evaluate(test_loader)
                print(f"\nTraining Complete!")
                print(f"Best epoch: {result.best_epoch}")
                print(f"Best val loss: {result.best_val_loss:.4f}")
                print(f"Test accuracy: {test_metrics.get('test_accuracy', 0):.2f}%")
                print(f"Checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
