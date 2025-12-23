import pytest
import subprocess
import os
import yaml
from pathlib import Path

def test_budget_limit_safety():
    """Test that training fails if budget limit is over $100."""
    config_content = {
        "device": "cpu",
        "budget_limit": 150.0,
        "model": {"backbone": "htsat-small", "embedding_dim": 256, "dropout": 0.1},
        "optim": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001, "betas": [0.9, 0.999]},
        "training": {"epochs": 1, "log_every": 1, "eval_every": 1, "grad_accum_steps": 1}
    }
    
    config_path = "tests/test_over_budget.yaml"
    os.makedirs("tests", exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
        
    try:
        # Run the training script with the over-budget config
        result = subprocess.run(
            ["python3", "scripts/train_model.py", "--config", config_path, "--synthetic", "--epochs", "1"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0
        assert "Budget limit $150.0 exceeds safety threshold $100.0" in result.stderr or "Budget limit $150.0 exceeds safety threshold $100.0" in result.stdout
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

def test_model_copy_creation():
    """Test that a copy of the model state is created when ensure_model_copy is True."""
    config_content = {
        "device": "cpu",
        "budget_limit": 50.0,
        "ensure_model_copy": True,
        "output_dir": "tests/test_output_copy",
        "model": {"backbone": "htsat-small", "embedding_dim": 256, "dropout": 0.1},
        "optim": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001, "betas": [0.9, 0.999]},
        "training": {"epochs": 1, "log_every": 1, "eval_every": 1, "grad_accum_steps": 1}
    }
    
    config_path = "tests/test_copy_config.yaml"
    os.makedirs("tests", exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
        
    try:
        # Run training for 1 epoch on synthetic data
        subprocess.run(
            ["python3", "scripts/train_model.py", "--config", config_path, "--synthetic", "--epochs", "1"],
            capture_output=True,
            text=True
        )
        
        copy_path = Path("tests/test_output_copy/initial_state_copy.pt")
        assert copy_path.exists()
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)
        if os.path.exists("tests/test_output_copy"):
            import shutil
            shutil.rmtree("tests/test_output_copy")

if __name__ == "__main__":
    test_budget_limit_safety()
    test_model_copy_creation()
    print("All safety tests passed!")
