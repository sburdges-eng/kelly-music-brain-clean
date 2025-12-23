#!/usr/bin/env python3
"""
Comprehensive ML Training Script for Kelly.

Trains models using the full training pipeline:
- Data loading with augmentation
- Custom loss functions
- Advanced architectures
- Comprehensive evaluation

Usage:
    python scripts/train_model.py --config configs/emotion_recognizer.yaml
    python scripts/train_model.py --model emotion --epochs 50 --batch-size 16
    python scripts/train_model.py --list-models

Note:
  This script is a full pipeline runner; for lightweight MPS laptop training
  see scripts/train_mps_stub.py and configs/laptop_m4_small.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install: pip install torch torchaudio")
    sys.exit(1)

# Try to import torchaudio
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

# Import training utilities
from python.penta_core.ml.training import (
    AudioAugmentor,
    AugmentationConfig,
    EmotionCNN,
    MelodyLSTM,
    HarmonyMLP,
    MultiTaskModel,
    MusicAwareLoss,
    FocalLoss,
    LabelSmoothingLoss,
    MusicMetrics,
    EmotionMetrics,
    GrooveMetrics,
    ModelValidator,
    evaluate_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
AUDIO_DATA_ROOT = Path("/Volumes/Extreme SSD/kelly-audio-data")
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs" / "training"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_type: str = "emotion"
    model_id: str = "emotionrecognizer"
    
    # Data
    data_path: str = ""
    sample_rate: int = 16000
    n_mels: int = 64
    max_duration: float = 5.0
    
    # Training
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Loss
    loss_type: str = "cross_entropy"
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    
    # Augmentation
    use_augmentation: bool = True
    aug_time_stretch: float = 0.3
    aug_pitch_shift: float = 0.3
    aug_noise: float = 0.2
    
    # Validation
    val_split: float = 0.1
    test_split: float = 0.1
    early_stopping_patience: int = 15
    
    # Output
    output_dir: str = ""
    save_best: bool = True
    log_interval: int = 10
    
    # Device
    device: str = "auto"
    num_workers: int = 0
    
    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: getattr(self, k) for k in dir(self)
            if not k.startswith("_") and not callable(getattr(self, k))
        }


# =============================================================================
# Dataset
# =============================================================================

class AudioDataset(Dataset):
    """
    Audio dataset with lazy loading and augmentation.
    """
    
    def __init__(
        self,
        data_dir: Path,
        sample_rate: int = 16000,
        n_mels: int = 64,
        max_duration: float = 5.0,
        augmentor: Optional[AudioAugmentor] = None,
        transform: Optional[Any] = None,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_duration = max_duration
        self.augmentor = augmentor
        self.transform = transform
        
        # Load metadata
        self.samples = self._load_samples()
        
        # Mel spectrogram transform
        if TORCHAUDIO_AVAILABLE:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=1024,
                hop_length=256,
            )
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load sample metadata."""
        samples = []
        
        # Check for metadata file
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
            return data.get("samples", [])
        
        # Fallback: scan directories
        processed_dir = self.data_dir / "processed"
        if processed_dir.exists():
            for emotion_dir in processed_dir.iterdir():
                if emotion_dir.is_dir():
                    emotion = emotion_dir.name
                    for audio_file in emotion_dir.glob("*.wav"):
                        samples.append({
                            "file": str(audio_file),
                            "emotion": emotion,
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Get file path
        if os.path.isabs(sample["file"]):
            audio_path = Path(sample["file"])
        else:
            audio_path = self.data_dir / sample["file"]
        
        # Load audio
        if TORCHAUDIO_AVAILABLE and audio_path.exists():
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Truncate/pad
            max_samples = int(self.max_duration * self.sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            elif waveform.shape[1] < max_samples:
                padding = max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Augmentation
            if self.augmentor is not None:
                waveform_np = waveform.squeeze().numpy()
                waveform_np = self.augmentor.augment(waveform_np, self.sample_rate)
                waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            
            # Convert to mel spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec = self.amplitude_to_db(mel_spec)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        else:
            # Dummy data for testing
            mel_spec = torch.randn(1, self.n_mels, 128)
        
        # Get label
        emotion = sample.get("emotion", "neutral")
        label = self._emotion_to_label(emotion)
        
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        return mel_spec, label
    
    def _emotion_to_label(self, emotion: str) -> int:
        """Convert emotion string to label index."""
        emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
        return emotions.index(emotion) if emotion in emotions else 6


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""
    
    def __init__(self, num_samples: int = 1000, num_classes: int = 7):
        self.num_samples = num_samples
        self.num_classes = num_classes
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Random mel spectrogram
        mel_spec = torch.randn(1, 64, 128)
        
        # Random label
        label = idx % self.num_classes
        
        return mel_spec, label


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """Training manager."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
        )
        
        # Loss function
        if config.loss_type == "focal":
            self.criterion = FocalLoss(gamma=config.focal_gamma)
        elif config.loss_type == "label_smoothing":
            self.criterion = LabelSmoothingLoss(
                smoothing=config.label_smoothing,
                num_classes=7,
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.metrics = EmotionMetrics()
        
        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        
        # Output directory
        self.output_dir = Path(config.output_dir) if config.output_dir else CHECKPOINTS_DIR / config.model_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % self.config.log_interval == 0:
                logger.debug(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.4f}, "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                
                if self.config.save_best:
                    self._save_checkpoint("best.pt")
            else:
                self.epochs_no_improve += 1
            
            # Early stopping
            if self.epochs_no_improve >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time:.1f}s")
        
        # Final evaluation
        if self.test_loader:
            test_results = self._evaluate_test()
        else:
            test_results = {}
        
        # Save final checkpoint
        self._save_checkpoint("final.pt")
        
        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "total_time": total_time,
            "test_results": test_results,
        }
    
    def _evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        logger.info("Evaluating on test set...")
        
        # Load best model
        best_path = self.output_dir / "best.pt"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        
        results = evaluate_model(self.model, self.test_loader, self.metrics, str(self.device))
        
        logger.info("Test Results:")
        for name, result in results.items():
            logger.info(f"  {name}: {result.value:.4f}")
        
        return {name: result.value for name, result in results.items()}
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save(self.model.state_dict(), path)
        logger.debug(f"Saved checkpoint: {path}")


# =============================================================================
# Model Factory
# =============================================================================

def create_model(model_type: str, **kwargs) -> nn.Module:
    """Create model based on type."""
    models = {
        "emotion": EmotionCNN,
        "melody": MelodyLSTM,
        "harmony": HarmonyMLP,
        "multitask": MultiTaskModel,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)


def create_dataloaders(
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders."""
    
    # Check for real data
    data_path = Path(config.data_path) if config.data_path else None
    
    if data_path and data_path.exists():
        logger.info(f"Loading data from: {data_path}")
        
        # Create augmentor
        augmentor = None
        if config.use_augmentation:
            aug_config = AugmentationConfig(
                p_time_stretch=config.aug_time_stretch,
                p_pitch_shift=config.aug_pitch_shift,
                p_noise=config.aug_noise,
            )
            augmentor = AudioAugmentor(aug_config)
        
        dataset = AudioDataset(
            data_dir=data_path,
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            max_duration=config.max_duration,
            augmentor=augmentor,
        )
    else:
        logger.warning("No data path specified, using synthetic data")
        dataset = SyntheticDataset(num_samples=1000, num_classes=7)
    
    # Split dataset
    n_samples = len(dataset)
    n_val = int(n_samples * config.val_split)
    n_test = int(n_samples * config.test_split)
    n_train = n_samples - n_val - n_test
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test]
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Main
# =============================================================================

def list_models():
    """List available models."""
    print("\nAvailable Models:")
    print("-" * 50)
    print("  emotion  - EmotionCNN for emotion recognition")
    print("  melody   - MelodyLSTM for melody generation")
    print("  harmony  - HarmonyMLP for chord prediction")
    print("  multitask - MultiTaskModel for joint learning")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Train Kelly ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, default="emotion", help="Model type")
    parser.add_argument("--data", type=str, help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_models()
        return
    
    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(Path(args.config))
    else:
        config = TrainingConfig()
    
    # Override with command line args
    if args.model:
        config.model_type = args.model
    if args.data:
        config.data_path = args.data
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.device:
        config.device = args.device
    if args.output:
        config.output_dir = args.output
    
    if args.synthetic:
        config.data_path = ""  # Force synthetic data
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("Kelly ML Training")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_type}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Data path: {config.data_path or 'synthetic'}")
    logger.info("=" * 60)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    model_kwargs = {}
    if config.model_type == "emotion":
        model_kwargs = {"num_classes": 7, "use_attention": True}
    elif config.model_type == "melody":
        model_kwargs = {"vocab_size": 128, "num_classes": 128}
    elif config.model_type == "harmony":
        model_kwargs = {"input_dim": 128, "num_chords": 48}
    elif config.model_type == "multitask":
        model_kwargs = {"tasks": ["emotion", "genre"]}
    
    model = create_model(config.model_type, **model_kwargs)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    
    # Train
    results = trainer.train()
    
    # Save results
    results_path = trainer.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": config.to_dict(),
            "results": {
                "best_val_loss": results["best_val_loss"],
                "total_time": results["total_time"],
                "test_results": results["test_results"],
            },
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

