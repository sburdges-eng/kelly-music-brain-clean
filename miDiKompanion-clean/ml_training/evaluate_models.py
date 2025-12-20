#!/usr/bin/env python3
"""
Model Evaluation Script for Kelly MIDI Companion
=================================================
Evaluates trained models on test datasets and generates comprehensive metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

# Import model definitions and utilities
from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor,
    SyntheticEmotionDataset,
    SyntheticMelodyDataset
)
from training_utils import evaluate_model, compute_cosine_similarity, compute_accuracy


def evaluate_emotion_recognizer(
    model: EmotionRecognizer,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Evaluate emotion recognition model."""
    criterion = nn.MSELoss()

    results = evaluate_model(
        model, test_loader, criterion, device,
        metric_fn=lambda o, t: compute_cosine_similarity(o, t)
    )

    # Additional metrics
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            mel_features = batch['mel_features'].to(device)
            emotion_target = batch['emotion'].to(device)
            emotion_pred = model(mel_features)

            all_preds.append(emotion_pred.cpu().numpy())
            all_targets.append(emotion_target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute mean absolute error per dimension
    mae = np.mean(np.abs(all_preds - all_targets))
    results['mae'] = float(mae)

    # Compute correlation
    correlations = []
    for i in range(all_preds.shape[1]):
        corr = np.corrcoef(all_preds[:, i], all_targets[:, i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    results['mean_correlation'] = float(np.mean(correlations)) if correlations else 0.0

    return results


def evaluate_melody_transformer(
    model: MelodyTransformer,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Evaluate melody transformer model."""
    criterion = nn.BCELoss()

    results = evaluate_model(model, test_loader, criterion, device)

    # Additional metrics
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            emotion = batch['emotion'].to(device)
            notes_target = batch['notes'].to(device)
            notes_pred = model(emotion)

            all_preds.append(notes_pred.cpu().numpy())
            all_targets.append(notes_target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Top-k accuracy (top 10 notes)
    k = 10
    top_k_preds = np.argsort(all_preds, axis=1)[:, -k:]
    top_k_targets = np.argsort(all_targets, axis=1)[:, -k:]

    top_k_accuracy = np.mean([
        len(np.intersect1d(top_k_preds[i], top_k_targets[i])) > 0
        for i in range(len(top_k_preds))
    ])
    results['top_k_accuracy'] = float(top_k_accuracy)

    # Mean precision (for active notes)
    threshold = 0.5
    pred_active = all_preds > threshold
    target_active = all_targets > threshold

    precision = np.sum(pred_active & target_active) / (np.sum(pred_active) + 1e-8)
    recall = np.sum(pred_active & target_active) / (np.sum(target_active) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    results['precision'] = float(precision)
    results['recall'] = float(recall)
    results['f1_score'] = float(f1)

    return results


def evaluate_all_models(
    checkpoint_dir: Path,
    output_dir: Path,
    batch_size: int = 64,
    device: str = 'cpu',
    test_samples: int = 1000
):
    """
    Evaluate all trained models.

    Args:
        checkpoint_dir: Directory containing model checkpoints
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        test_samples: Number of test samples to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Kelly MIDI Companion - Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print()

    # Create test datasets
    print("Creating test datasets...")
    emotion_test = SyntheticEmotionDataset(num_samples=test_samples, seed=123)
    melody_test = SyntheticMelodyDataset(num_samples=test_samples, seed=123)

    emotion_loader = DataLoader(emotion_test, batch_size=batch_size, shuffle=False)
    melody_loader = DataLoader(melody_test, batch_size=batch_size, shuffle=False)

    all_results = {}

    # Evaluate EmotionRecognizer
    print("\n[1/2] Evaluating EmotionRecognizer...")
    try:
        model = EmotionRecognizer()
        checkpoint_path = checkpoint_dir / "emotionrecognizer_best.pt"

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            print("  Warning: No checkpoint found, using random weights")

        model = model.to(device)
        results = evaluate_emotion_recognizer(model, emotion_loader, device)
        all_results['EmotionRecognizer'] = results

        print(f"  Loss: {results['loss']:.6f}")
        print(f"  Cosine Similarity: {results['metric']:.4f}")
        print(f"  MAE: {results['mae']:.6f}")
        print(f"  Mean Correlation: {results['mean_correlation']:.4f}")
    except Exception as e:
        print(f"  Error evaluating EmotionRecognizer: {e}")
        all_results['EmotionRecognizer'] = {'error': str(e)}

    # Evaluate MelodyTransformer
    print("\n[2/2] Evaluating MelodyTransformer...")
    try:
        model = MelodyTransformer()
        checkpoint_path = checkpoint_dir / "melodytransformer_best.pt"

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            print("  Warning: No checkpoint found, using random weights")

        model = model.to(device)
        results = evaluate_melody_transformer(model, melody_loader, device)
        all_results['MelodyTransformer'] = results

        print(f"  Loss: {results['loss']:.6f}")
        if 'top_k_accuracy' in results:
            print(f"  Top-10 Accuracy: {results['top_k_accuracy']:.4f}")
        if 'f1_score' in results:
            print(f"  F1 Score: {results['f1_score']:.4f}")
    except Exception as e:
        print(f"  Error evaluating MelodyTransformer: {e}")
        all_results['MelodyTransformer'] = {'error': str(e)}

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Evaluation complete! Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    for model_name, results in all_results.items():
        if 'error' not in results:
            print(f"\n{model_name}:")
            for metric, value in results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Kelly MIDI Companion ML models")
    parser.add_argument(
        "--checkpoint-dir", "-c",
        type=str,
        default="./trained_models/checkpoints",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=1000,
        help="Number of test samples to use"
    )

    args = parser.parse_args()

    # Auto-detect best device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    evaluate_all_models(
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output),
        batch_size=args.batch_size,
        device=args.device,
        test_samples=args.test_samples
    )
