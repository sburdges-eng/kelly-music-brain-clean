#!/usr/bin/env python3
"""
MidiKompanion ML Training Pipeline

Trains all 5 models for the emotion-to-music generation pipeline:
1. EmotionRecognizer: Audio features → 64-dim emotion embedding
2. MelodyTransformer: Emotion embedding → MIDI note probabilities
3. HarmonyPredictor: Context → Chord probabilities
4. DynamicsEngine: Node intensity → Expression parameters
5. GroovePredictor: Node arousal → Groove parameters

Usage:
    python train_all_models.py --epochs 100 --batch-size 32 --output-dir ./models
    
Or with Docker:
    docker-compose up training
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Definitions
# =============================================================================

class EmotionRecognizer(nn.Module):
    """Audio features → 64-dim emotion embedding"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Output in [-1, 1] for VAD-like values
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MelodyTransformer(nn.Module):
    """Emotion embedding → MIDI note probabilities"""
    
    def __init__(self, input_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # Probabilities [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HarmonyPredictor(nn.Module):
    """Context → Chord probabilities"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)  # Chord probability distribution
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DynamicsEngine(nn.Module):
    """Node intensity → Expression parameters"""
    
    def __init__(self, input_dim: int = 32, output_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()  # Expression values [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GroovePredictor(nn.Module):
    """Node arousal → Groove parameters"""
    
    def __init__(self, input_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()  # Groove parameters [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# Dataset Classes
# =============================================================================

class SyntheticEmotionDataset(Dataset):
    """
    Synthetic dataset for initial training.
    Creates emotion-labeled data based on VAD coordinates.
    
    For production: Replace with real audio/MIDI dataset.
    """
    
    def __init__(self, num_samples: int = 10000, model_type: str = "emotion"):
        self.num_samples = num_samples
        self.model_type = model_type
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        data = []
        
        for _ in range(self.num_samples):
            if self.model_type == "emotion":
                # Audio features → Emotion embedding
                x = np.random.randn(128).astype(np.float32)
                # Create coherent VAD-based output
                vad = np.random.uniform(-1, 1, 4).astype(np.float32)  # V, A, D, I
                y = np.concatenate([vad, np.random.randn(60).astype(np.float32) * 0.1])
                
            elif self.model_type == "melody":
                # Emotion embedding → Note probabilities
                x = np.random.randn(64).astype(np.float32)
                # Higher notes more likely for high valence
                valence = np.clip(x[0], -1, 1)
                y = np.zeros(128, dtype=np.float32)
                center = int(64 + valence * 20)  # Center pitch based on valence
                for i in range(128):
                    y[i] = np.exp(-((i - center) ** 2) / 200) * np.random.uniform(0.5, 1.0)
                y = y / y.max()
                
            elif self.model_type == "harmony":
                # Context → Chord probabilities
                x = np.random.randn(128).astype(np.float32)
                y = np.random.dirichlet(np.ones(64)).astype(np.float32)
                
            elif self.model_type == "dynamics":
                # Intensity → Expression
                x = np.random.uniform(0, 1, 32).astype(np.float32)
                intensity = x.mean()
                y = np.full(16, intensity, dtype=np.float32)
                y += np.random.randn(16).astype(np.float32) * 0.1
                y = np.clip(y, 0, 1)
                
            elif self.model_type == "groove":
                # Arousal → Groove
                x = np.random.uniform(0, 1, 64).astype(np.float32)
                arousal = x.mean()
                y = np.zeros(32, dtype=np.float32)
                y[0] = arousal * 0.5  # Swing
                y[1] = arousal * 0.3  # Humanize
                y[2:] = np.random.uniform(0, arousal, 30)
                
            else:
                x = np.random.randn(128).astype(np.float32)
                y = np.random.randn(64).astype(np.float32)
            
            data.append((x, y))
        
        return data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# =============================================================================
# Training Functions
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cuda"
) -> Dict:
    """Train a single model."""
    
    logger.info(f"Training {model_name}...")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{epochs}: "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
    
    logger.info(f"  Best validation loss: {best_val_loss:.6f}")
    
    return history


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: Path,
    model_name: str
):
    """Export PyTorch model to ONNX format."""
    
    logger.info(f"Exporting {model_name} to ONNX...")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    # Verify export
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    logger.info(f"  Saved to: {output_path}")


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train MidiKompanion ML models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model configurations
    models_config = {
        "emotion_recognizer": {
            "class": EmotionRecognizer,
            "input_dim": 128,
            "output_dim": 64,
            "dataset_type": "emotion"
        },
        "melody_transformer": {
            "class": MelodyTransformer,
            "input_dim": 64,
            "output_dim": 128,
            "dataset_type": "melody"
        },
        "harmony_predictor": {
            "class": HarmonyPredictor,
            "input_dim": 128,
            "output_dim": 64,
            "dataset_type": "harmony"
        },
        "dynamics_engine": {
            "class": DynamicsEngine,
            "input_dim": 32,
            "output_dim": 16,
            "dataset_type": "dynamics"
        },
        "groove_predictor": {
            "class": GroovePredictor,
            "input_dim": 64,
            "output_dim": 32,
            "dataset_type": "groove"
        }
    }
    
    all_history = {}
    
    # Train each model
    for model_name, config in models_config.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*60}")
        
        # Create model
        model = config["class"](config["input_dim"], config["output_dim"])
        
        # Create datasets
        train_dataset = SyntheticEmotionDataset(
            num_samples=args.num_samples,
            model_type=config["dataset_type"]
        )
        val_dataset = SyntheticEmotionDataset(
            num_samples=args.num_samples // 5,
            model_type=config["dataset_type"]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device
        )
        
        all_history[model_name] = history
        
        # Export to ONNX
        onnx_path = output_dir / f"{model_name}.onnx"
        export_to_onnx(
            model=model.cpu(),
            input_shape=(config["input_dim"],),
            output_path=onnx_path,
            model_name=model_name
        )
        
        # Save PyTorch model too
        torch_path = output_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), torch_path)
        logger.info(f"  Saved PyTorch model: {torch_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        # Convert numpy types to Python types for JSON
        json_history = {}
        for name, hist in all_history.items():
            json_history[name] = {
                "train_loss": [float(x) for x in hist["train_loss"]],
                "val_loss": [float(x) for x in hist["val_loss"]]
            }
        json.dump(json_history, f, indent=2)
    
    logger.info(f"\nTraining complete! Models saved to: {output_dir}")
    logger.info(f"Training history: {history_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("MODEL SUMMARY")
    logger.info("="*60)
    for name, config in models_config.items():
        onnx_path = output_dir / f"{name}.onnx"
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        logger.info(f"  {name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
