#!/usr/bin/env python3
"""
ML Training Experiment Tracking Example

Demonstrates structured experiment tracking for ML training runs:
- Configuration management and versioning
- Run metadata tracking (git commit, hardware, timing)
- Results and metrics logging
- Checkpoint organization
- Output file management

This example shows how to use the TrainRun and TrainConfig classes
from scripts/train.py to track experiments systematically.

Usage:
    python examples/ml/experiment_tracking_example.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Define minimal classes needed for the example
# (These are also defined in scripts/train.py, but we include them here
# to make this example self-contained and avoid requiring torch/numpy)


@dataclass
class TrainConfig:
    """Training configuration container (minimal version for example)."""
    
    model_id: str = "emotion_recognizer"
    model_type: str = "RTNeural"
    task: str = "emotion_embedding"
    input_size: int = 128
    output_size: int = 64
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "relu"
    dropout: float = 0.2
    architecture_type: str = "mlp"
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss: str = "cross_entropy"
    author: str = ""
    notes: str = ""
    labels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class TrainRun:
    """Metadata for a training run (minimal version for example)."""
    
    run_id: str = ""
    config: Optional[TrainConfig] = None
    git_commit: str = ""
    hardware: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    final_test_loss: float = 0.0
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    output_files: List[str] = field(default_factory=list)
    success: bool = False
    error: str = ""
    
    def save(self, path: Path):
        """Save run metadata to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        if self.config:
            data["config"] = self.config.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "TrainRun":
        """Load run metadata from JSON."""
        with open(path) as f:
            data = json.load(f)
        if data.get("config"):
            data["config"] = TrainConfig(**data["config"])
        return cls(**data)


def get_git_commit() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_hardware_info() -> str:
    """Get hardware description."""
    info = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info = result.stdout.strip()
        except Exception:
            pass
    return info


def ensure_dirs():
    """Create necessary directories."""
    (ROOT / "logs" / "training").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)


def demo_basic_tracking():
    """
    Demo 1: Basic experiment tracking with TrainRun
    
    Shows how to create and save a training run with metadata.
    """
    print("\n" + "=" * 70)
    print("Demo 1: Basic Experiment Tracking")
    print("=" * 70)
    
    # Create training configuration
    config = TrainConfig(
        model_id="emotion_recognizer_demo",
        model_type="RTNeural",
        task="emotion_embedding",
        architecture_type="mlp",
        input_size=128,
        output_size=64,
        hidden_layers=[512, 256, 128],
        epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        author="Demo User",
        notes="Example experiment for documentation",
    )
    
    print("\n1. Configuration created:")
    print(f"   Model: {config.model_id}")
    print(f"   Task: {config.task}")
    print(f"   Architecture: {config.architecture_type}")
    print(f"   Epochs: {config.epochs}")
    
    # Initialize run metadata
    run = TrainRun(
        run_id=f"{config.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        git_commit=get_git_commit(),
        hardware=get_hardware_info(),
        start_time=datetime.now().isoformat(),
    )
    
    print("\n2. Run metadata initialized:")
    print(f"   Run ID: {run.run_id}")
    print(f"   Git commit: {run.git_commit}")
    print(f"   Hardware: {run.hardware}")
    print(f"   Start time: {run.start_time}")
    
    # Simulate training results
    run.final_train_loss = 0.234
    run.final_val_loss = 0.267
    run.final_test_loss = 0.289
    run.best_epoch = 7
    run.best_val_loss = 0.245
    run.success = True
    
    # Complete the run
    run.end_time = datetime.now().isoformat()
    start = datetime.fromisoformat(run.start_time)
    end = datetime.fromisoformat(run.end_time)
    run.duration_seconds = (end - start).total_seconds()
    
    print("\n3. Training completed (simulated):")
    print(f"   Best epoch: {run.best_epoch}")
    print(f"   Best val loss: {run.best_val_loss:.4f}")
    print(f"   Test loss: {run.final_test_loss:.4f}")
    print(f"   Duration: {run.duration_seconds:.2f}s")
    
    # Save run metadata
    ensure_dirs()
    logs_dir = ROOT / "logs" / "training"
    log_path = logs_dir / f"{run.run_id}.json"
    run.save(log_path)
    
    print(f"\n4. Run metadata saved to: {log_path}")
    
    return run, log_path


def demo_results_output():
    """
    Demo 2: Structured results output
    
    Shows how to save training results with config and metadata.
    """
    print("\n" + "=" * 70)
    print("Demo 2: Structured Results Output")
    print("=" * 70)
    
    # Create a simple config
    config = TrainConfig(
        model_id="harmony_predictor_demo",
        task="chord_prediction",
        epochs=50,
        batch_size=32,
        learning_rate=2e-4,
    )
    
    # Simulate training results
    results = {
        "train_losses": [0.523, 0.412, 0.356, 0.298, 0.267],
        "val_losses": [0.545, 0.434, 0.389, 0.321, 0.298],
        "test_loss": 0.312,
        "test_accuracy": 0.876,
        "best_epoch": 4,
        "n_params": 2_456_789,
    }
    
    # Create results directory
    output_dir = ROOT / "checkpoints" / config.model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results with config and timestamp
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": config.to_dict(),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    
    print("\n1. Results structure:")
    print(f"   Config keys: {list(config.to_dict().keys())[:5]}...")
    print(f"   Result keys: {list(results.keys())}")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    
    print(f"\n2. Results saved to: {results_path}")
    
    # Display sample content
    print("\n3. Sample results.json content:")
    with open(results_path) as f:
        content = json.load(f)
    print(f"   Model: {content['config']['model_id']}")
    print(f"   Best epoch: {content['results']['best_epoch']}")
    print(f"   Test accuracy: {content['results']['test_accuracy']:.2%}")
    
    return results_path


def demo_checkpoint_organization():
    """
    Demo 3: Checkpoint and output file organization
    
    Shows recommended directory structure for experiment artifacts.
    """
    print("\n" + "=" * 70)
    print("Demo 3: Checkpoint and Output Organization")
    print("=" * 70)
    
    model_id = "melody_transformer_demo"
    
    # Create organized directory structure
    checkpoints_dir = ROOT / "checkpoints" / model_id
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate checkpoint files
    checkpoint_files = [
        "best.pt",
        "epoch_5.pt",
        "epoch_10.pt",
        "final.pt",
    ]
    
    # Simulate output files
    output_files = [
        "results.json",
        "training_curves.png",
        "confusion_matrix.png",
    ]
    
    print("\n1. Recommended checkpoint structure:")
    print(f"\n   checkpoints/{model_id}/")
    for f in checkpoint_files:
        filepath = checkpoints_dir / f
        # Create empty placeholder files for demo
        filepath.touch()
        print(f"   ├── {f}")
    
    for f in output_files:
        filepath = checkpoints_dir / f
        filepath.touch()
        print(f"   ├── {f}")
    
    # Create run log
    logs_dir = ROOT / "logs" / "training"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = logs_dir / f"{run_id}.json"
    
    print(f"\n   logs/training/")
    print(f"   └── {log_file.name}")
    
    # Save sample run metadata
    run = TrainRun(
        run_id=run_id,
        config=TrainConfig(model_id=model_id),
        git_commit=get_git_commit(),
        hardware=get_hardware_info(),
        start_time=datetime.now().isoformat(),
        success=True,
        output_files=[str(checkpoints_dir / f) for f in checkpoint_files + output_files],
    )
    run.save(log_file)
    
    print("\n2. Output files tracked in run metadata:")
    print(f"   Total files: {len(run.output_files)}")
    
    return checkpoints_dir, log_file


def demo_config_versioning():
    """
    Demo 4: Configuration versioning and comparison
    
    Shows how to track and compare different experiment configs.
    """
    print("\n" + "=" * 70)
    print("Demo 4: Configuration Versioning")
    print("=" * 70)
    
    # Create baseline config
    baseline_config = TrainConfig(
        model_id="baseline_model",
        learning_rate=1e-3,
        batch_size=16,
        dropout=0.2,
        notes="Baseline configuration",
    )
    
    # Create experiment variant
    variant_config = TrainConfig(
        model_id="variant_model",
        learning_rate=2e-4,
        batch_size=32,
        dropout=0.3,
        notes="Higher batch size, lower LR, more dropout",
    )
    
    print("\n1. Configuration comparison:")
    print("\n   Baseline:")
    print(f"     LR: {baseline_config.learning_rate}")
    print(f"     Batch size: {baseline_config.batch_size}")
    print(f"     Dropout: {baseline_config.dropout}")
    
    print("\n   Variant:")
    print(f"     LR: {variant_config.learning_rate}")
    print(f"     Batch size: {variant_config.batch_size}")
    print(f"     Dropout: {variant_config.dropout}")
    
    # Save configs for reference
    configs_dir = ROOT / "configs" / "experiments"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_path = configs_dir / "baseline_config.json"
    variant_path = configs_dir / "variant_config.json"
    
    with open(baseline_path, "w") as f:
        json.dump(baseline_config.to_dict(), f, indent=2)
    
    with open(variant_path, "w") as f:
        json.dump(variant_config.to_dict(), f, indent=2)
    
    print(f"\n2. Configs saved for comparison:")
    print(f"   Baseline: {baseline_path}")
    print(f"   Variant: {variant_path}")
    
    return baseline_path, variant_path


def demo_run_querying():
    """
    Demo 5: Querying experiment history
    
    Shows how to load and analyze past experiment runs.
    """
    print("\n" + "=" * 70)
    print("Demo 5: Querying Experiment History")
    print("=" * 70)
    
    logs_dir = ROOT / "logs" / "training"
    
    if not logs_dir.exists() or not list(logs_dir.glob("*.json")):
        print("\n   No experiment logs found. Run other demos first.")
        return
    
    # Load all run logs
    runs = []
    for log_file in logs_dir.glob("*.json"):
        try:
            run = TrainRun.load(log_file)
            runs.append(run)
        except Exception as e:
            print(f"   Warning: Could not load {log_file.name}: {e}")
    
    if not runs:
        print("\n   No valid runs found.")
        return
    
    print(f"\n1. Found {len(runs)} experiment run(s):")
    
    # Display summary of runs
    for i, run in enumerate(runs, 1):
        print(f"\n   Run {i}:")
        print(f"     ID: {run.run_id}")
        print(f"     Model: {run.config.model_id if run.config else 'N/A'}")
        print(f"     Success: {run.success}")
        if run.best_val_loss < float('inf'):
            print(f"     Best val loss: {run.best_val_loss:.4f}")
        print(f"     Duration: {run.duration_seconds:.2f}s")
    
    # Find best run
    successful_runs = [r for r in runs if r.success and r.best_val_loss < float('inf')]
    if successful_runs:
        best_run = min(successful_runs, key=lambda r: r.best_val_loss)
        print(f"\n2. Best performing run:")
        print(f"   ID: {best_run.run_id}")
        print(f"   Best val loss: {best_run.best_val_loss:.4f}")
        print(f"   Best epoch: {best_run.best_epoch}")


def main():
    """Run all experiment tracking demos."""
    print("\n" + "=" * 70)
    print("ML Training Experiment Tracking Examples")
    print("=" * 70)
    print("\nThis example demonstrates experiment tracking features:")
    print("  - TrainRun metadata tracking")
    print("  - Structured results output")
    print("  - Checkpoint organization")
    print("  - Configuration versioning")
    print("  - Experiment querying")
    
    # Run demos
    demo_basic_tracking()
    demo_results_output()
    demo_checkpoint_organization()
    demo_config_versioning()
    demo_run_querying()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - logs/training/*.json         (run metadata)")
    print("  - checkpoints/*/results.json   (training results)")
    print("  - checkpoints/*/*.pt           (model checkpoints)")
    print("  - configs/experiments/*.json   (config versions)")
    print("\nSee docs/ml/EXPERIMENT_TRACKING.md for more details.")
    print()


if __name__ == "__main__":
    main()
