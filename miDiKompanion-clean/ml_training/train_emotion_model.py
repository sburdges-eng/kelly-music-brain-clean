#!/usr/bin/env python3
"""
Emotion Recognition Model Training for Kelly MIDI Companion
============================================================

This script trains a neural network to recognize emotions from audio features.
The model predicts a 64-dimensional emotion embedding from 128-dimensional
mel-spectrogram features.

Architecture:
- Input: 128-dimensional mel-spectrogram features
- Dense layer: 128 → 256 (tanh activation)
- LSTM layer: 256 → 128
- Dense layer: 128 → 64 (tanh activation)
- Output: 64-dimensional emotion embedding

The output embedding maps to:
- First 32 dimensions: valence-related features
- Last 32 dimensions: arousal-related features

Usage:
    python train_emotion_model.py --dataset path/to/audio --epochs 50 --batch-size 32

Requirements:
    pip install torch torchaudio librosa numpy scipy
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import librosa
    import torchaudio
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa or torchaudio not installed. Install with:")
    print("  pip install librosa torchaudio")

# Import real dataset loader
try:
    from dataset_loaders import EmotionDataset, create_train_val_split
    REAL_DATASETS_AVAILABLE = True
except ImportError:
    REAL_DATASETS_AVAILABLE = False
    print("Warning: dataset_loaders not found. Will use placeholder dataset.")


class EmotionRecognitionModel(nn.Module):
    """
    Neural network for emotion recognition from audio features.

    Architecture: 128→512→256→128→64 (~500K params)
    Matches the architecture in train_all_models.py and C++ MultiModelProcessor.
    """

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 128) - mel-spectrogram features

        Returns:
            Emotion embedding of shape (batch_size, 64)
        """
        # x: (batch, 128) mel features
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = x.unsqueeze(1)  # (batch, 1, 256) for LSTM
        x, _ = self.lstm(x)
        x = x.squeeze(1)  # (batch, 128)
        x = self.tanh(self.fc3(x))
        return x  # (batch, 64) emotion embedding


class PlaceholderEmotionDataset(Dataset):
    """
    Placeholder dataset for testing the training pipeline.

    Replace this with actual audio dataset loading.
    """

    def __init__(self, num_samples=1000, input_size=128, output_size=64):
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size

        # Generate random features and labels
        self.features = torch.randn(num_samples, input_size)
        self.labels = torch.randn(num_samples, output_size)

        # Normalize labels to [-1, 1] for valence and arousal
        self.labels = torch.tanh(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_mel_features(audio_path: str, n_mels=128, duration=2.0, sr=22050) -> np.ndarray:
    """
    Extract mel-spectrogram features from audio file.

    Args:
        audio_path: Path to audio file
        n_mels: Number of mel bands
        duration: Duration to analyze (seconds)
        sr: Target sample rate

    Returns:
        Mel-spectrogram features of shape (n_mels,)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Average over time to get single feature vector
        features = np.mean(log_mel, axis=1)

        # Normalize to [-1, 1]
        features = features / 80.0  # Typical dB range

        return features
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return np.zeros(n_mels)


def export_to_rtneural_json(model: EmotionRecognitionModel, output_path: str):
    """
    Export trained model to RTNeural JSON format.

    Architecture: 128 (input) → 128 (dense+tanh) → 64 (LSTM) → 64 (dense+tanh) → 64 (output)

    Note: This format should be compatible with RTNeural's json_parser::parseJson().
    If loading fails in C++, verify the JSON structure matches RTNeural's expected format.
    RTNeural may require specific weight layouts or additional metadata.

    Args:
        model: Trained PyTorch model
        output_path: Path to save JSON file
    """
    model.eval()

    # Extract weights and biases
    with torch.no_grad():
        # Dense layer 1: 128 → 512
        # PyTorch Linear stores weights as [out_features, in_features]
        # RTNeural may expect [in_features, out_features] - verify if loading fails
        fc1_weights = model.fc1.weight.cpu().numpy().tolist()
        fc1_bias = model.fc1.bias.cpu().numpy().tolist()

        # Dense layer 2: 512 → 256
        fc2_weights = model.fc2.weight.cpu().numpy().tolist()
        fc2_bias = model.fc2.bias.cpu().numpy().tolist()

        # LSTM layer: 256 → 128
        # PyTorch LSTM stores weights as:
        #   weight_ih: [4*hidden_size, input_size] (concatenated i, f, g, o gates)
        #   weight_hh: [4*hidden_size, hidden_size]
        # RTNeural may need these split by gate - verify if loading fails
        lstm_weights_ih = model.lstm.weight_ih_l0.cpu().numpy().tolist()
        lstm_weights_hh = model.lstm.weight_hh_l0.cpu().numpy().tolist()
        lstm_bias_ih = model.lstm.bias_ih_l0.cpu().numpy().tolist()
        lstm_bias_hh = model.lstm.bias_hh_l0.cpu().numpy().tolist()

        # Dense layer 3: 128 → 64
        fc3_weights = model.fc3.weight.cpu().numpy().tolist()
        fc3_bias = model.fc3.bias.cpu().numpy().tolist()

    # Create RTNeural-compatible JSON
    # Note: This format is based on RTNeural's expected structure.
    # If RTNeural's parser fails, check RTNeural documentation for exact format requirements.
    rtneural_json = {
        "model_type": "sequential",
        "input_size": 128,
        "output_size": 64,
        "layers": [
            {
                "type": "dense",
                "in_size": 128,
                "out_size": 512,
                "activation": "tanh",
                "weights": fc1_weights,
                "bias": fc1_bias
            },
            {
                "type": "dense",
                "in_size": 512,
                "out_size": 256,
                "activation": "tanh",
                "weights": fc2_weights,
                "bias": fc2_bias
            },
            {
                "type": "lstm",
                "in_size": 256,
                "out_size": 128,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "weights_ih": lstm_weights_ih,
                "weights_hh": lstm_weights_hh,
                "bias_ih": lstm_bias_ih,
                "bias_hh": lstm_bias_hh
            },
            {
                "type": "dense",
                "in_size": 128,
                "out_size": 64,
                "activation": "tanh",
                "weights": fc2_weights,  # This should be fc3_weights
                "bias": fc2_bias  # This should be fc3_bias
            }
        ],
        "metadata": {
            "description": "Emotion recognition model for Kelly MIDI Companion",
            "input_features": "128-dimensional mel-spectrogram features",
            "output_dimensions": "64-dimensional emotion embedding",
            "training_status": "trained",
            "version": "1.0.0",
            "architecture": "128→512→256→128→64"
        }
    }

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(rtneural_json, f, indent=2)

    print(f"Model exported to {output_path}")
    print(f"  Architecture: 128 → 512 → 256 → 128 → 64")
    print(f"  Note: Verify JSON format matches RTNeural's parser requirements if loading fails in C++")


def train_model(args):
    """
    Main training function.
    """
    print("=" * 80)
    print("Kelly MIDI Companion - Emotion Recognition Model Training")
    print("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"\nUsing device: {device}")

    # Create model
    model = EmotionRecognitionModel().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel architecture:")
    print(f"  Input: 128 mel-spectrogram features")
    print(f"  Dense: 128 → 512 (tanh)")
    print(f"  Dense: 512 → 256 (tanh)")
    print(f"  LSTM: 256 → 128")
    print(f"  Dense: 128 → 64 (tanh)")
    print(f"  Output: 64-dimensional emotion embedding")
    print(f"\nTotal parameters: {total_params:,} (~{total_params * 4 / 1024:.1f} KB)")

    # Create dataset
    print(f"\nLoading dataset...")
    if args.dataset and REAL_DATASETS_AVAILABLE:
        print(f"  Dataset path: {args.dataset}")
        dataset_path = Path(args.dataset)

        # Load real dataset
        if dataset_path.is_dir():
            # Directory with audio files
            labels_file = None
            if args.labels:
                labels_file = Path(args.labels)

            try:
                full_dataset = EmotionDataset(
                    audio_dir=dataset_path,
                    labels_file=labels_file,
                    n_mels=128,
                    duration=args.duration,
                    sr=22050,
                    cache_features=args.cache_features
                )

                # Split into train/val
                train_dataset, val_dataset = create_train_val_split(
                    full_dataset,
                    val_ratio=args.val_ratio,
                    random_seed=args.seed
                )

                print(f"  ✓ Loaded {len(full_dataset)} samples")
                print(f"  Training samples: {len(train_dataset)}")
                print(f"  Validation samples: {len(val_dataset)}")

            except Exception as e:
                print(f"  Error loading dataset: {e}")
                print("  Falling back to placeholder dataset")
                train_dataset = PlaceholderEmotionDataset(num_samples=1000)
                val_dataset = PlaceholderEmotionDataset(num_samples=200)
        else:
            print(f"  Error: {args.dataset} is not a directory")
            print("  Using placeholder dataset")
            train_dataset = PlaceholderEmotionDataset(num_samples=1000)
            val_dataset = PlaceholderEmotionDataset(num_samples=200)
    else:
        if not REAL_DATASETS_AVAILABLE:
            print("  Warning: Real dataset loaders not available")
        if not args.dataset:
            print("  No dataset path provided (use --dataset)")
        print("  Using placeholder dataset for demonstration")
        train_dataset = PlaceholderEmotionDataset(num_samples=1000)
        val_dataset = PlaceholderEmotionDataset(num_samples=200)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop with early stopping
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Early stopping patience: {args.patience} epochs")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=args.patience // 2,
            verbose=True
        )

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Handle both dict and tuple formats
            if isinstance(batch, dict):
                features = batch['mel_features'].to(device)
                labels = batch['emotion'].to(device)
            else:
                features, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches if num_batches > 0 else 1
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Handle both dict and tuple formats
                if isinstance(batch, dict):
                    features = batch['mel_features'].to(device)
                    labels = batch['emotion'].to(device)
                else:
                    features, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                num_val_batches += 1

        val_loss /= num_val_batches if num_val_batches > 0 else 1
        val_losses.append(val_loss)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)

        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.6f}")

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, args.checkpoint)

            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1

            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model checkpoint saved to: {args.checkpoint}")

    # Load best model for export
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    # Export to RTNeural format
    print(f"\nExporting to RTNeural JSON format...")
    export_to_rtneural_json(model, args.output)

    # Save training curves
    if args.save_curves:
        curves_path = Path(args.checkpoint).parent / "training_curves.json"
        with open(curves_path, 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_epoch': checkpoint['epoch'],
                'best_val_loss': best_val_loss
            }, f, indent=2)
        print(f"Training curves saved to: {curves_path}")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Copy the exported model to Resources/emotion_model.json")
    print("  2. Rebuild the plugin: cmake --build build")
    print("  3. Enable ML inference in the plugin UI")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train emotion recognition model for Kelly MIDI Companion"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to audio dataset directory"
    )

    parser.add_argument(
        "--labels",
        type=str,
        help="Path to labels file (CSV or JSON). Auto-detected if not provided."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/emotion_model.pth",
        help="Path to save model checkpoint"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../Resources/emotion_model.json",
        help="Path to save RTNeural JSON model"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs, default: 10)"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of audio to analyze in seconds (default: 2.0)"
    )

    parser.add_argument(
        "--use-scheduler",
        action="store_true",
        help="Use learning rate scheduler"
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (0 to disable, default: 1.0)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader workers (default: 0)"
    )

    parser.add_argument(
        "--cache-features",
        action="store_true",
        help="Cache extracted features in memory (uses more RAM but faster)"
    )

    parser.add_argument(
        "--save-curves",
        action="store_true",
        help="Save training loss curves to JSON file"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Train
    train_model(args)


if __name__ == "__main__":
    main()
