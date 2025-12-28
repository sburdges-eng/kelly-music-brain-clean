# ML Training Examples

This directory contains examples demonstrating Kelly's ML training features.

## Available Examples

### Experiment Tracking Example

**File:** `experiment_tracking_example.py`

Demonstrates structured experiment tracking for ML training:
- Configuration management and versioning
- Run metadata tracking (git commit, hardware, timing)
- Results and metrics logging
- Checkpoint organization
- Experiment querying

**Usage:**
```bash
python examples/ml/experiment_tracking_example.py
```

**What it shows:**
1. **Basic tracking** - Creating and saving TrainRun metadata
2. **Results output** - Structured results.json format
3. **Checkpoint organization** - Recommended directory structure
4. **Config versioning** - Comparing different experiments
5. **Run querying** - Loading and analyzing past experiments

**Output:**
- `logs/training/*.json` - Run metadata files
- `checkpoints/*/results.json` - Training results
- `checkpoints/*/*.pt` - Model checkpoints (placeholder)
- `configs/experiments/*.json` - Config versions

## Documentation

See **[docs/ml/EXPERIMENT_TRACKING.md](../../docs/ml/EXPERIMENT_TRACKING.md)** for complete documentation on experiment tracking.

## Related Scripts

- `scripts/train.py` - Main training script with full experiment tracking
- `scripts/train_model.py` - Alternative trainer with Trainer class
- `configs/` - Example configuration files

## Quick Start

```python
from scripts.train import TrainConfig, TrainRun, get_git_commit, get_hardware_info
from datetime import datetime

# Create config
config = TrainConfig(
    model_id="my_model",
    epochs=50,
    batch_size=16,
)

# Initialize run
run = TrainRun(
    run_id=f"{config.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config=config,
    git_commit=get_git_commit(),
    hardware=get_hardware_info(),
    start_time=datetime.now().isoformat(),
)

# ... train model ...

# Save results
run.success = True
run.save(Path("logs/training") / f"{run.run_id}.json")
```

## Testing

To verify the examples work:

```bash
# Run the experiment tracking example
python examples/ml/experiment_tracking_example.py

# Check generated files
ls -la logs/training/
ls -la checkpoints/*/
```
