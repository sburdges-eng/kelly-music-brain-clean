# ML Training Experiment Tracking

> Structured tracking for ML training experiments in Kelly

## Overview

This guide documents the experiment tracking system built into Kelly's ML training pipeline. The system provides:

- **Configuration management** - Reproducible training configs with versioning
- **Run metadata tracking** - Git commit, hardware info, timing, and provenance
- **Results logging** - Structured output of metrics, losses, and artifacts
- **Checkpoint organization** - Systematic storage of model checkpoints and outputs
- **Experiment querying** - Load and compare past training runs

## Quick Start

### Running a Training Experiment

```bash
# Train with automatic experiment tracking
python scripts/train.py --model emotion_recognizer --epochs 50

# Train with custom config
python scripts/train.py --config configs/emotion_recognizer.yaml
```

This automatically:
1. Creates a unique run ID with timestamp
2. Captures git commit and hardware info
3. Saves run metadata to `logs/training/{run_id}.json`
4. Saves results to `checkpoints/{model_id}/results.json`
5. Organizes checkpoints in `checkpoints/{model_id}/`

### Example Code

```python
from scripts.train import TrainConfig, TrainRun, get_git_commit, get_hardware_info
from datetime import datetime

# Create configuration
config = TrainConfig(
    model_id="emotion_recognizer",
    task="emotion_embedding",
    epochs=100,
    batch_size=16,
    learning_rate=1e-3,
    author="Your Name",
    notes="Experiment description",
)

# Initialize run metadata
run = TrainRun(
    run_id=f"{config.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config=config,
    git_commit=get_git_commit(),
    hardware=get_hardware_info(),
    start_time=datetime.now().isoformat(),
)

# ... train model ...

# Save results
run.final_train_loss = train_loss
run.final_val_loss = val_loss
run.final_test_loss = test_loss
run.best_epoch = best_epoch
run.success = True
run.end_time = datetime.now().isoformat()

# Save run metadata
run.save(logs_dir / f"{run.run_id}.json")
```

See `examples/ml/experiment_tracking_example.py` for complete examples.

---

## Core Components

### 1. TrainConfig

Configuration container for all training hyperparameters and settings.

```python
@dataclass
class TrainConfig:
    # Model identity
    model_id: str = "emotion_recognizer"
    model_type: str = "RTNeural"
    task: str = "emotion_embedding"
    
    # Architecture
    input_size: int = 128
    output_size: int = 64
    hidden_layers: List[int] = [512, 256, 128]
    activation: str = "relu"
    dropout: float = 0.2
    architecture_type: str = "mlp"
    
    # Training
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss: str = "cross_entropy"
    
    # Metadata
    author: str = ""
    notes: str = ""
    labels: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """Load config from YAML file."""
        ...
```

**Key methods:**
- `to_dict()` - Serialize to JSON-compatible dict
- `from_yaml()` - Load from YAML config file

### 2. TrainRun

Metadata container for a complete training run.

```python
@dataclass
class TrainRun:
    # Identity
    run_id: str = ""
    config: Optional[TrainConfig] = None
    
    # Provenance
    git_commit: str = ""
    hardware: str = ""
    
    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    
    # Results
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    final_test_loss: float = 0.0
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    
    # Outputs
    output_files: List[str] = []
    success: bool = False
    error: str = ""
    
    def save(self, path: Path):
        """Save run metadata to JSON."""
        ...
    
    @classmethod
    def load(cls, path: Path) -> "TrainRun":
        """Load run metadata from JSON."""
        ...
```

**Key fields:**
- `run_id` - Unique identifier (format: `{model_id}_{timestamp}`)
- `config` - Full training configuration
- `git_commit` - Git commit SHA for reproducibility
- `hardware` - Hardware info (CPU/GPU/MPS)
- `output_files` - List of generated artifacts

---

## Directory Structure

Experiment artifacts are organized in a standard structure:

```
kelly-music-brain-clean/
├── logs/
│   └── training/
│       ├── emotion_recognizer_20231215_143022.json
│       ├── melody_transformer_20231215_150145.json
│       └── harmony_predictor_20231216_091234.json
│
├── checkpoints/
│   ├── emotion_recognizer/
│   │   ├── best.pt                  # Best model checkpoint
│   │   ├── epoch_5.pt               # Periodic checkpoints
│   │   ├── epoch_10.pt
│   │   ├── final.pt                 # Final checkpoint
│   │   └── results.json             # Training results
│   │
│   └── melody_transformer/
│       ├── best.pt
│       ├── final.pt
│       └── results.json
│
├── models/
│   ├── emotion_recognizer.json      # RTNeural JSON export
│   ├── emotion_recognizer.onnx      # ONNX export
│   ├── emotion_recognizer.mlmodel   # Core ML export
│   └── registry.json                # Model registry
│
└── configs/
    ├── emotion_recognizer.yaml
    ├── melody_transformer.yaml
    └── experiments/
        ├── baseline_config.json
        └── variant_config.json
```

### Directory Purposes

| Directory | Purpose | Contents |
|-----------|---------|----------|
| `logs/training/` | Run metadata | `{run_id}.json` files with complete run info |
| `checkpoints/{model_id}/` | Training artifacts | `.pt` checkpoints, `results.json` |
| `models/` | Exported models | `.json`, `.onnx`, `.mlmodel` files |
| `configs/` | Configuration files | YAML configs for reproducibility |

---

## Run Metadata Format

### Run Log JSON Structure

File: `logs/training/{run_id}.json`

```json
{
  "run_id": "emotion_recognizer_20231215_143022",
  "config": {
    "model_id": "emotion_recognizer",
    "model_type": "RTNeural",
    "task": "emotion_embedding",
    "architecture_type": "mlp",
    "input_size": 128,
    "output_size": 64,
    "hidden_layers": [512, 256, 128],
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "cross_entropy",
    "author": "Kelly Dev",
    "notes": "Baseline emotion recognition model"
  },
  "git_commit": "a3f5d892c1b4",
  "hardware": "Apple M2 Max",
  "start_time": "2023-12-15T14:30:22.145678",
  "end_time": "2023-12-15T15:45:33.987654",
  "duration_seconds": 4511.84,
  "final_train_loss": 0.234,
  "final_val_loss": 0.267,
  "final_test_loss": 0.289,
  "best_epoch": 73,
  "best_val_loss": 0.245,
  "output_files": [
    "/path/to/checkpoints/emotion_recognizer/best.pt",
    "/path/to/checkpoints/emotion_recognizer/final.pt",
    "/path/to/models/emotion_recognizer.json",
    "/path/to/models/emotion_recognizer.onnx"
  ],
  "success": true,
  "error": ""
}
```

### Results JSON Structure

File: `checkpoints/{model_id}/results.json`

```json
{
  "config": {
    "model_id": "emotion_recognizer",
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001
  },
  "results": {
    "train_losses": [0.523, 0.412, 0.356, ..., 0.234],
    "val_losses": [0.545, 0.434, 0.389, ..., 0.267],
    "test_loss": 0.289,
    "test_accuracy": 0.876,
    "best_epoch": 73,
    "n_params": 2456789
  },
  "timestamp": "2023-12-15T15:45:33.987654"
}
```

---

## Usage Patterns

### 1. Track a Training Run

```python
from scripts.train import run_training, TrainConfig

config = TrainConfig(
    model_id="my_model",
    epochs=50,
    batch_size=32,
)

# Runs complete training pipeline with automatic tracking
run = run_training(config)

print(f"Run ID: {run.run_id}")
print(f"Success: {run.success}")
print(f"Best val loss: {run.best_val_loss}")
```

### 2. Load and Compare Past Runs

```python
from pathlib import Path
from scripts.train import TrainRun

logs_dir = Path("logs/training")

# Load all runs
runs = []
for log_file in logs_dir.glob("*.json"):
    run = TrainRun.load(log_file)
    runs.append(run)

# Find best run
best_run = min(
    (r for r in runs if r.success),
    key=lambda r: r.best_val_loss
)

print(f"Best run: {best_run.run_id}")
print(f"Val loss: {best_run.best_val_loss:.4f}")
print(f"Config: {best_run.config.to_dict()}")
```

### 3. Reproduce an Experiment

```python
from pathlib import Path
from scripts.train import TrainRun, run_training

# Load previous run
old_run = TrainRun.load(Path("logs/training/emotion_recognizer_20231215_143022.json"))

# Use same config
config = old_run.config

# Re-run with exact same settings
new_run = run_training(config)

print(f"Original val loss: {old_run.best_val_loss:.4f}")
print(f"New val loss: {new_run.best_val_loss:.4f}")
```

### 4. Save Custom Metrics

```python
# During training, add custom metrics to results
results = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "test_loss": test_loss,
    "test_accuracy": test_acc,
    "best_epoch": best_epoch,
    "n_params": n_params,
    # Custom metrics
    "confusion_matrix": confusion_matrix.tolist(),
    "per_class_accuracy": per_class_acc.tolist(),
    "training_curves": "training_curves.png",
}

# Save with config and timestamp
results_path = output_dir / "results.json"
with open(results_path, "w") as f:
    json.dump({
        "config": config.to_dict(),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }, f, indent=2, default=str)
```

---

## Best Practices

### Configuration Management

1. **Version your configs** - Save configs as YAML files in `configs/`
2. **Use descriptive IDs** - `model_id` should be unique and descriptive
3. **Document experiments** - Use `notes` and `author` fields
4. **Tag experiments** - Use `labels` for experiment grouping

```python
config = TrainConfig(
    model_id="emotion_recognizer_v2_highcap",
    author="Jane Doe",
    notes="Increased capacity experiment - 3x hidden layers",
    labels=["emotion", "high-capacity", "ablation"],
)
```

### Run Tracking

1. **Always track git commit** - Ensures reproducibility
2. **Log hardware info** - Important for performance comparisons
3. **Save all outputs** - Track checkpoints, exports, plots
4. **Mark success/failure** - Use `success` and `error` fields

```python
try:
    # Training code
    run.success = True
except Exception as e:
    run.success = False
    run.error = str(e)
finally:
    run.end_time = datetime.now().isoformat()
    run.save(log_path)
```

### Results Organization

1. **Use standard file names** - `best.pt`, `final.pt`, `results.json`
2. **Keep checkpoints separate** - Don't mix with source code
3. **Save training curves** - Visual inspection is valuable
4. **Include test metrics** - Not just validation metrics

### Reproducibility

1. **Set random seeds** - Document in config
2. **Pin dependencies** - Track PyTorch version
3. **Save data version** - Use `data_version` field
4. **Log full environment** - OS, Python version, hardware

---

## Integration with Training Scripts

### scripts/train.py

Main training script with full experiment tracking:

```python
def run_training(config: TrainConfig) -> TrainRun:
    """Execute full training pipeline."""
    ensure_dirs()
    
    # Initialize run metadata
    run = TrainRun(
        run_id=f"{config.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        git_commit=get_git_commit(),
        hardware=get_hardware_info(),
        start_time=datetime.now().isoformat(),
    )
    
    logger.info(f"Starting training run: {run.run_id}")
    
    try:
        # Train model
        model, results = train_model(config, run)
        
        # Export to formats
        export_to_rtneural_json(model, config, run)
        
        # Update registry
        update_registry(config, run)
        
        run.success = True
        
    except Exception as e:
        run.success = False
        run.error = str(e)
        logger.exception("Training failed")
    
    finally:
        run.end_time = datetime.now().isoformat()
        run.duration_seconds = calculate_duration(run)
        
        # Save run metadata
        log_path = LOGS_DIR / f"{run.run_id}.json"
        run.save(log_path)
    
    return run
```

### scripts/train_model.py

Alternative training script using Trainer class:

```python
class Trainer:
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        # ... training code ...
        
        # Save results
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump({
                "config": self.config.to_dict(),
                "results": {
                    "best_val_loss": self.best_val_loss,
                    "total_time": total_time,
                    "test_results": test_results,
                },
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, default=str)
        
        return results
```

---

## Advanced Features

### Custom Run Attributes

Extend `TrainRun` with custom fields for your experiments:

```python
# In your training code
run = TrainRun(...)
run.custom_metric = my_metric
run.dataset_stats = {"mean": 0.5, "std": 0.2}
run.augmentation_applied = True

# These will be saved in the run JSON
run.save(log_path)
```

### Experiment Comparison

```python
def compare_runs(run_ids: List[str], metric: str = "best_val_loss"):
    """Compare multiple runs by a metric."""
    runs = [TrainRun.load(f"logs/training/{rid}.json") for rid in run_ids]
    
    for run in sorted(runs, key=lambda r: getattr(r, metric)):
        print(f"{run.run_id}: {getattr(run, metric):.4f}")
        print(f"  Config: {run.config.learning_rate} LR, {run.config.batch_size} BS")
```

### Hyperparameter Logging

```python
# Log all hyperparameters systematically
config = TrainConfig(
    learning_rate=1e-3,
    batch_size=16,
    dropout=0.2,
    weight_decay=1e-4,
)

# All fields automatically tracked in run metadata
run = TrainRun(config=config, ...)
run.save(log_path)

# Later: analyze hyperparameter impact
for log_file in logs_dir.glob("*.json"):
    run = TrainRun.load(log_file)
    print(f"LR={run.config.learning_rate}, Val Loss={run.best_val_loss}")
```

---

## Troubleshooting

### Common Issues

**Run logs not saving:**
- Ensure `logs/training/` directory exists (use `ensure_dirs()`)
- Check file permissions
- Verify JSON serialization (use `default=str` in `json.dump()`)

**Checkpoints too large:**
- Save only `state_dict()`, not full model
- Use periodic checkpoints (`epoch_5.pt`, `epoch_10.pt`)
- Keep only best and final checkpoints

**Config not reproducible:**
- Save data version and random seed
- Document data preprocessing steps
- Track augmentation settings

**Cannot load old runs:**
- Check for breaking changes in `TrainRun`/`TrainConfig`
- Add version field to run metadata
- Handle missing fields with defaults

---

## See Also

- `examples/ml/experiment_tracking_example.py` - Complete working examples
- `scripts/train.py` - Main training script
- `scripts/train_model.py` - Alternative trainer implementation
- `docs/MK_TRAINING_GUIDELINES.md` - Full training workflow
- `configs/` - Example configuration files

---

## Summary

Kelly's experiment tracking system provides:

✅ **Automatic tracking** - Run metadata captured by default  
✅ **Reproducibility** - Git commit, config, and hardware info saved  
✅ **Structured output** - Standard directory layout and file formats  
✅ **Easy querying** - Load and compare runs programmatically  
✅ **Extensible** - Add custom metrics and metadata  

Start experimenting:
```bash
python examples/ml/experiment_tracking_example.py
```
