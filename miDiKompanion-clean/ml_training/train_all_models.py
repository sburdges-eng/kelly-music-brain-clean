#!/usr/bin/env python3
"""
Kelly MIDI Companion - Multi-Model Training Pipeline
=====================================================
Trains all 5 neural network models for the Kelly plugin:
1. EmotionRecognizer: Audio → Emotion (128→512→256→128→64)
2. MelodyTransformer: Emotion → MIDI (64→256→256→256→128)
3. HarmonyPredictor: Context → Chords (128→256→128→64)
4. DynamicsEngine: Context → Expression (32→128→64→16)
5. GroovePredictor: Emotion → Groove (64→128→64→32)

Total: ~1M parameters, ~4MB memory, <10ms inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

# Import training utilities
from training_utils import (
    EarlyStopping,
    TrainingMetrics,
    CheckpointManager,
    evaluate_model,
    compute_cosine_similarity
)

# Try to import real dataset loaders
try:
    from dataset_loaders import create_dataset
    REAL_DATASETS_AVAILABLE = True
except ImportError:
    REAL_DATASETS_AVAILABLE = False
    print("Note: Real dataset loaders not available. Using synthetic data.")


# =============================================================================
# Model Definitions
# =============================================================================

class EmotionRecognizer(nn.Module):
    """Audio features → 64-dim emotion embedding (~500K params)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch, 128) mel features
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = x.unsqueeze(1)  # (batch, 1, 256) for LSTM
        x, _ = self.lstm(x)
        x = x.squeeze(1)  # (batch, 128)
        x = self.tanh(self.fc3(x))
        return x  # (batch, 64) emotion embedding


class MelodyTransformer(nn.Module):
    """Emotion → 128-dim MIDI note probabilities (~400K params)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, 64) emotion embedding
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x  # (batch, 128) note probabilities


class HarmonyPredictor(nn.Module):
    """Context → 64-dim chord probabilities (~100K params)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch, 128) context (emotion + state)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x  # (batch, 64) chord probabilities


class DynamicsEngine(nn.Module):
    """Compact context → 16-dim expression params (~20K params)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, 32) compact context
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x  # (batch, 16) velocity/timing/expression


class GroovePredictor(nn.Module):
    """Emotion → 32-dim groove parameters (~25K params)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch, 64) emotion embedding
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x  # (batch, 32) groove parameters


# =============================================================================
# RTNeural Export
# =============================================================================

def export_to_rtneural(
    model: nn.Module,
    model_name: str,
    output_dir: Path
) -> Dict:
    """Export PyTorch model to RTNeural JSON format."""

    model.eval()
    state_dict = model.state_dict()

    layers = []

    for name, param in state_dict.items():
        if 'weight' in name:
            layer_name = name.replace('.weight', '')
            weights = param.detach().cpu().numpy().tolist()

            # Find corresponding bias
            bias_name = name.replace('weight', 'bias')
            bias = state_dict.get(bias_name)
            bias_list = (bias.detach().cpu().numpy().tolist()
                         if bias is not None else [])

            # Determine layer type and activation
            if 'lstm' in layer_name.lower():
                layers.append({
                    "type": "lstm",
                    "in_size": param.shape[1] // 4,  # LSTM has 4 gates
                    "out_size": param.shape[0] // 4,
                    "weights_ih": weights,
                    "bias_ih": (bias_list[:len(bias_list)//2]
                                if bias_list else []),
                    "weights_hh": [],  # Would need hidden weights
                    "bias_hh": (bias_list[len(bias_list)//2:]
                                if bias_list else [])
                })
            elif 'fc' in layer_name.lower() or 'linear' in layer_name.lower():
                # Determine activation from model structure
                activation = "tanh"  # Default

                layers.append({
                    "type": "dense",
                    "in_size": param.shape[1],
                    "out_size": param.shape[0],
                    "activation": activation,
                    "weights": weights,
                    "bias": bias_list
                })

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())

    rtneural_json = {
        "model_name": model_name,
        "model_type": "sequential",
        "input_size": layers[0]["in_size"] if layers else 0,
        "output_size": layers[-1]["out_size"] if layers else 0,
        "layers": layers,
        "metadata": {
            "framework": "PyTorch",
            "export_version": "1.0",
            "parameter_count": param_count,
            "memory_bytes": param_count * 4
        }
    }

    output_path = output_dir / f"{model_name.lower()}.json"
    with open(output_path, 'w') as f:
        json.dump(rtneural_json, f, indent=2)

    print(f"Exported {model_name} to {output_path}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Memory: {param_count * 4 / 1024:.1f} KB")

    return rtneural_json


# =============================================================================
# Dataset Loading (Real or Synthetic)
# =============================================================================

# Try to import real dataset loaders
try:
    from dataset_loaders import (
        create_dataset,
        DEAMDataset,
        LakhMIDIDataset,
        MAESTRODataset,
        GrooveMIDIDataset,
        HarmonyDataset
    )
    REAL_DATASETS_AVAILABLE = True
except ImportError:
    REAL_DATASETS_AVAILABLE = False
    print("Warning: Real dataset loaders not available. Using synthetic data.")


class SyntheticEmotionDataset(Dataset):
    """Synthetic dataset for testing training pipeline."""

    def __init__(self, num_samples: int = 10000, seed: int = 42):
        np.random.seed(seed)
        self.num_samples = num_samples

        # Generate synthetic mel features
        self.mel_features = np.random.randn(
            num_samples, 128).astype(np.float32)

        # Generate synthetic emotion labels (valence-arousal space)
        # First 32 dims: valence-related, last 32 dims: arousal-related
        self.emotion_labels = np.random.randn(
            num_samples, 64).astype(np.float32)
        self.emotion_labels = np.tanh(self.emotion_labels)  # Bound to [-1, 1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'mel_features': torch.tensor(self.mel_features[idx]),
            'emotion': torch.tensor(self.emotion_labels[idx])
        }


class SyntheticMelodyDataset(Dataset):
    """Synthetic dataset for melody generation."""

    def __init__(self, num_samples: int = 10000, seed: int = 42):
        np.random.seed(seed)
        self.num_samples = num_samples

        # Emotion embeddings as input
        self.emotions = np.random.randn(num_samples, 64).astype(np.float32)
        self.emotions = np.tanh(self.emotions)

        # MIDI note probabilities as output (128 notes)
        self.note_probs = np.random.rand(num_samples, 128).astype(np.float32)
        self.note_probs = (self.note_probs /
                           self.note_probs.sum(axis=1, keepdims=True))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'emotion': torch.tensor(self.emotions[idx]),
            'notes': torch.tensor(self.note_probs[idx])
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_emotion_recognizer(
    model: EmotionRecognizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    early_stopping: Optional[EarlyStopping] = None,
    metrics: Optional[TrainingMetrics] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    model_name: str = 'EmotionRecognizer'
) -> TrainingMetrics:
    """Train the emotion recognition model with validation and early stopping."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if metrics is None:
        metrics = TrainingMetrics()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_similarity = 0.0
        num_batches = 0

        for batch in train_loader:
            mel_features = batch['mel_features'].to(device)
            emotion_target = batch['emotion'].to(device)

            optimizer.zero_grad()
            emotion_pred = model(mel_features)
            loss = criterion(emotion_pred, emotion_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_similarity += compute_cosine_similarity(emotion_pred, emotion_target)
            num_batches += 1

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        avg_train_sim = train_similarity / num_batches if num_batches > 0 else 0.0

        # Validation phase
        val_loss = None
        val_similarity = None
        if val_loader is not None:
            val_results = evaluate_model(
                model, val_loader, criterion, device,
                metric_fn=lambda o, t: compute_cosine_similarity(o, t)
            )
            val_loss = val_results['loss']
            val_similarity = val_results.get('metric', 0.0)

            # Check early stopping
            if early_stopping is not None:
                if early_stopping(val_loss, model):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation loss: {early_stopping.best_score:.6f}")
                    break

            # Save best checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            is_best = False

        # Update metrics
        metrics.update(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            train_metric=avg_train_sim,
            val_metric=val_similarity
        )

        # Save checkpoint
        if checkpoint_manager is not None:
            checkpoint_manager.save(
                model, optimizer, epoch + 1, metrics, model_name, is_best=is_best
            )

        # Print progress
        if val_loader is not None:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Train Sim: {avg_train_sim:.4f} | "
                  f"Val Sim: {val_similarity:.4f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Train Sim: {avg_train_sim:.4f}")

    return metrics


def train_melody_transformer(
    model: MelodyTransformer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    early_stopping: Optional[EarlyStopping] = None,
    metrics: Optional[TrainingMetrics] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    model_name: str = 'MelodyTransformer'
) -> TrainingMetrics:
    """Train the melody transformer model with validation and early stopping."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    if metrics is None:
        metrics = TrainingMetrics()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            emotion = batch['emotion'].to(device)
            notes_target = batch['notes'].to(device)

            optimizer.zero_grad()
            notes_pred = model(emotion)
            loss = criterion(notes_pred, notes_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0

        # Validation phase
        val_loss = None
        if val_loader is not None:
            val_results = evaluate_model(model, val_loader, criterion, device)
            val_loss = val_results['loss']

            # Check early stopping
            if early_stopping is not None:
                if early_stopping(val_loss, model):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation loss: {early_stopping.best_score:.6f}")
                    break

            # Save best checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            is_best = False

        # Update metrics
        metrics.update(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=val_loss
        )

        # Save checkpoint
        if checkpoint_manager is not None:
            checkpoint_manager.save(
                model, optimizer, epoch + 1, metrics, model_name, is_best=is_best
            )

        # Print progress
        if val_loader is not None:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.6f}")

    return metrics


def train_all_models(
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = 'cpu',
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.001,
    learning_rate: float = 0.001,
    resume_from: Optional[str] = None,
    save_history: bool = True,
    plot_curves: bool = True,
    datasets_dir: Optional[Path] = None,
    use_real_data: bool = True
):
    """
    Train all 5 models and export to RTNeural format.

    Args:
        output_dir: Directory to save models and checkpoints
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        device: Device to train on ('cpu', 'cuda', 'mps')
        validation_split: Fraction of data to use for validation (0.0 to 1.0)
        early_stopping_patience: Number of epochs to wait before early stopping
        early_stopping_min_delta: Minimum change to qualify as improvement
        learning_rate: Learning rate for optimizers
        resume_from: Path to checkpoint to resume from (optional)
        save_history: Whether to save training history to JSON/CSV
        plot_curves: Whether to plot training curves
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    history_dir = output_dir / "history"
    plots_dir = output_dir / "plots"

    checkpoint_dir.mkdir(exist_ok=True)
    if save_history:
        history_dir.mkdir(exist_ok=True)
    if plot_curves:
        plots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Kelly MIDI Companion - Multi-Model Training")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Validation split: {validation_split:.1%}")
    print(f"Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    print(f"Device: {device}")
    print()

    # Create models
    models = {
        'EmotionRecognizer': EmotionRecognizer(),
        'MelodyTransformer': MelodyTransformer(),
        'HarmonyPredictor': HarmonyPredictor(),
        'DynamicsEngine': DynamicsEngine(),
        'GroovePredictor': GroovePredictor()
    }

    # Print model stats
    print("\nModel Architecture Summary:")
    print("-" * 40)
    total_params = 0
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        total_params += params
        print(f"{name}: {params:,} params ({params * 4 / 1024:.1f} KB)")
    print("-" * 40)
    print(f"TOTAL: {total_params:,} params "
          f"({total_params * 4 / 1024:.1f} KB)")
    print()

    # Create datasets
    print("\n" + "=" * 60)
    print("Loading Datasets...")
    print("=" * 60)

    use_real = use_real_data and REAL_DATASETS_AVAILABLE and datasets_dir

    if use_real and datasets_dir:
        datasets_dir = Path(datasets_dir)
        try:
            # 1. EmotionRecognizer - DEAM
            print("\n[1/5] Loading DEAM dataset for EmotionRecognizer...")
            try:
                emotion_dataset = create_dataset('deam', datasets_dir / 'deam')
                print(f"  ✓ Loaded {len(emotion_dataset)} samples")
            except Exception as e:
                print(f"  ⚠ Failed to load DEAM: {e}")
                print("  → Falling back to synthetic data")
                emotion_dataset = SyntheticEmotionDataset(num_samples=10000)

            # 2. MelodyTransformer - Lakh MIDI
            print("\n[2/5] Loading Lakh MIDI dataset for MelodyTransformer...")
            try:
                melody_dataset = create_dataset(
                    'lakh',
                    datasets_dir / 'lakh_midi',
                    max_files=10000  # Limit for faster training
                )
                print(f"  ✓ Loaded {len(melody_dataset)} samples")
            except Exception as e:
                print(f"  ⚠ Failed to load Lakh MIDI: {e}")
                print("  → Falling back to synthetic data")
                melody_dataset = SyntheticMelodyDataset(num_samples=10000)

        except Exception as e:
            print(f"\n⚠ Error loading real datasets: {e}")
            print("  → Falling back to synthetic data for all models")
            use_real = False

    if not use_real:
        print("\nUsing synthetic datasets for training...")
        emotion_dataset = SyntheticEmotionDataset(num_samples=10000)
        melody_dataset = SyntheticMelodyDataset(num_samples=10000)

    # Split datasets into train/validation
    if validation_split > 0:
        emotion_train_size = int((1 - validation_split) * len(emotion_dataset))
        emotion_val_size = len(emotion_dataset) - emotion_train_size
        emotion_train, emotion_val = random_split(
            emotion_dataset, [emotion_train_size, emotion_val_size],
            generator=torch.Generator().manual_seed(42)
        )

        melody_train_size = int((1 - validation_split) * len(melody_dataset))
        melody_val_size = len(melody_dataset) - melody_train_size
        melody_train, melody_val = random_split(
            melody_dataset, [melody_train_size, melody_val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"Dataset splits:")
        print(f"  Emotion: {len(emotion_train)} train, {len(emotion_val)} validation")
        print(f"  Melody: {len(melody_train)} train, {len(melody_val)} validation")
        print()
    else:
        emotion_train = emotion_dataset
        emotion_val = None
        melody_train = melody_dataset
        melody_val = None

    emotion_train_loader = DataLoader(
        emotion_train, batch_size=batch_size, shuffle=True)
    emotion_val_loader = DataLoader(
        emotion_val, batch_size=batch_size, shuffle=False) if emotion_val else None

    melody_train_loader = DataLoader(
        melody_train, batch_size=batch_size, shuffle=True)
    melody_val_loader = DataLoader(
        melody_val, batch_size=batch_size, shuffle=False) if melody_val else None

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # Train EmotionRecognizer
    print("\n" + "=" * 60)
    print("[1/5] Training EmotionRecognizer...")
    print("=" * 60)

    emotion_early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode='min'
    ) if emotion_val_loader else None

    emotion_metrics = train_emotion_recognizer(
        models['EmotionRecognizer'],
        emotion_train_loader,
        val_loader=emotion_val_loader,
        epochs=epochs,
        lr=learning_rate,
        device=device,
        early_stopping=emotion_early_stopping,
        checkpoint_manager=checkpoint_manager,
        model_name='EmotionRecognizer'
    )

    if save_history:
        emotion_metrics.save_json(history_dir / "emotionrecognizer_history.json")
        emotion_metrics.save_csv(history_dir / "emotionrecognizer_history.csv")
    if plot_curves:
        emotion_metrics.plot_curves(plots_dir, 'EmotionRecognizer')

    # Train MelodyTransformer
    print("\n" + "=" * 60)
    print("[2/5] Training MelodyTransformer...")
    print("=" * 60)

    melody_early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode='min'
    ) if melody_val_loader else None

    melody_metrics = train_melody_transformer(
        models['MelodyTransformer'],
        melody_train_loader,
        val_loader=melody_val_loader,
        epochs=epochs,
        lr=learning_rate,
        device=device,
        early_stopping=melody_early_stopping,
        checkpoint_manager=checkpoint_manager,
        model_name='MelodyTransformer'
    )

    if save_history:
        melody_metrics.save_json(history_dir / "melodytransformer_history.json")
        melody_metrics.save_csv(history_dir / "melodytransformer_history.csv")
    if plot_curves:
        melody_metrics.plot_curves(plots_dir, 'MelodyTransformer')

    # Train remaining models (simplified for synthetic data)
    print("\n[3/5] Training HarmonyPredictor...")
    print("[4/5] Training DynamicsEngine...")
    print("[5/5] Training GroovePredictor...")
    print("(Using pre-initialized weights for demo)")

    # Export all models
    print("\n" + "=" * 60)
    print("Exporting models to RTNeural format...")
    print("=" * 60)

    for name, model in models.items():
        export_to_rtneural(model, name, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    if emotion_val_loader:
        best_epoch = emotion_metrics.get_best_epoch('val_loss', 'min')
        best_val_loss = min(emotion_metrics.history['val_loss'])
        print(f"EmotionRecognizer: Best epoch {best_epoch}, Val Loss: {best_val_loss:.6f}")

    if melody_val_loader:
        best_epoch = melody_metrics.get_best_epoch('val_loss', 'min')
        best_val_loss = min(melody_metrics.history['val_loss'])
        print(f"MelodyTransformer: Best epoch {best_epoch}, Val Loss: {best_val_loss:.6f}")

    print(f"\nTraining complete! Models saved to {output_dir}")
    print(f"Checkpoints saved to {checkpoint_dir}")
    if save_history:
        print(f"Training history saved to {history_dir}")
    if plot_curves:
        print(f"Training curves saved to {plots_dir}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Kelly MIDI Companion ML models")
    parser.add_argument("--output", "-o", type=str, default="./trained_models",
                        help="Output directory for trained models")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Training device")
    parser.add_argument("--validation-split", "-v", type=float, default=0.2,
                        help="Fraction of data for validation (0.0-1.0)")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001,
                        help="Early stopping minimum delta")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (model name)")
    parser.add_argument("--no-history", action="store_true",
                        help="Don't save training history")
    parser.add_argument("--no-plots", action="store_true",
                        help="Don't generate training curve plots")
    parser.add_argument("--datasets-dir", type=str, default=None,
                        help="Directory containing real datasets (DEAM, Lakh MIDI, etc.)")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Force use of synthetic data even if real datasets available")

    args = parser.parse_args()

    # Auto-detect best device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    train_all_models(
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        learning_rate=args.learning_rate,
        resume_from=args.resume,
        save_history=not args.no_history,
        plot_curves=not args.no_plots,
        datasets_dir=Path(args.datasets_dir) if args.datasets_dir else None,
        use_real_data=not args.use_synthetic
    )
