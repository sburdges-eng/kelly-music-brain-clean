"""
Test for ML experiment tracking example.
"""
import subprocess
import sys
from pathlib import Path


def test_experiment_tracking_example():
    """Test that the experiment tracking example runs successfully."""
    example_path = Path(__file__).parent.parent.parent / "examples" / "ml" / "experiment_tracking_example.py"
    
    assert example_path.exists(), f"Example file not found: {example_path}"
    
    # Run the example
    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    # Check it completed successfully
    assert result.returncode == 0, f"Example failed with error:\n{result.stderr}"
    
    # Check output contains expected sections
    assert "Demo 1: Basic Experiment Tracking" in result.stdout
    assert "Demo 2: Structured Results Output" in result.stdout
    assert "Demo 3: Checkpoint and Output Organization" in result.stdout
    assert "Demo 4: Configuration Versioning" in result.stdout
    assert "Demo 5: Querying Experiment History" in result.stdout
    assert "Examples complete!" in result.stdout
    
    # Check that key files were created
    root = Path(__file__).parent.parent.parent
    logs_dir = root / "logs" / "training"
    
    assert logs_dir.exists(), "Logs directory not created"
    assert any(logs_dir.glob("*.json")), "No log files created"
    
    print("✓ Experiment tracking example test passed!")


if __name__ == "__main__":
    test_experiment_tracking_example()
