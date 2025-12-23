import pytest
from pathlib import Path
import torch
import os
import sys
import importlib.util

# Handle directory with spaces by using absolute path and importlib
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parents[1]
HARDWARE_SCRIPT_PATH = REPO_ROOT / "ML Kelly Training" / "tests" / "hardware_benchmarks.py"
CONFIG_DIR = REPO_ROOT / "ML Kelly Training" / "backup" / "configs"

def import_hardware_benchmarks():
    spec = importlib.util.spec_from_file_location("hardware_benchmarks", str(HARDWARE_SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

hardware_benchmarks = import_hardware_benchmarks()

@pytest.mark.parametrize("config_name", ["mac_mps_16gb.yaml", "nvidia_cuda_gpu.yaml"])
def test_hardware_training_integrity(config_name):
    config_path = CONFIG_DIR / config_name
    
    if not config_path.exists():
        pytest.skip(f"Config {config_path} not found")
        
    # Run the hardware test
    # This verifies both budget enforcement and model copy integrity
    success = hardware_benchmarks.run_hardware_test(str(config_path))
    
    assert success, f"Hardware training integrity test failed for {config_name}"

def test_spending_limit_configs():
    """Verify that all hardware configs have a $100 spending limit."""
    for config_name in ["mac_mps_16gb.yaml", "nvidia_cuda_gpu.yaml"]:
        config_path = CONFIG_DIR / config_name
        if config_path.exists():
            cfg = hardware_benchmarks.load_config(str(config_path))
            # Some configs might use spending_limit_usd, others budget_usd. We check both.
            limit = cfg.get('spending_limit_usd') or cfg.get('budget_usd')
            assert limit == 100.0, f"Spending limit for {config_name} is not $100"

