#!/usr/bin/env python3
"""
Kelly ML Quick Start Training Demo

Demonstrates the training pipeline with synthetic data.
Use this to verify your environment is set up correctly.

Usage:
    python scripts/quickstart_training.py
    python scripts/quickstart_training.py --model emotion_recognizer
    python scripts/quickstart_training.py --epochs 10 --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_environment():
    """Check that all required packages are installed."""
    print("=" * 60)
    print("Kelly ML Environment Check")
    print("=" * 60)
    
    issues = []
    
    # Python version
    import platform
    py_version = platform.python_version()
    print(f"✓ Python: {py_version}")
    
    # NumPy
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        issues.append("NumPy not installed: pip install numpy")
    
    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        # Check device
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            device_name = "Apple Silicon (Metal)"
        else:
            device = "cpu"
            device_name = "CPU"
        
        print(f"✓ Device: {device} ({device_name})")
        
    except ImportError:
        issues.append("PyTorch not installed: pip install torch")
        return False
    
    # Optional packages
    optional = [
        ("tqdm", "tqdm"),
        ("yaml", "pyyaml"),
        ("onnx", "onnx"),
        ("coremltools", "coremltools"),
    ]
    
    for module, package in optional:
        try:
            __import__(module)
            print(f"✓ {package}: installed")
        except ImportError:
            print(f"○ {package}: not installed (optional)")
    
    print("=" * 60)
    
    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✓ Environment ready for training!")
    return True


def run_quick_demo(model_name: str = "emotion_recognizer", epochs: int = 5, quick: bool = False):
    """Run a quick training demo with synthetic data."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    print("\n" + "=" * 60)
    print(f"Quick Training Demo: {model_name}")
    print("=" * 60)
    
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Model configs
    configs = {
        "emotion_recognizer": {
            "input_size": (1, 64, 128),  # CNN input
            "output_size": 7,
            "model_type": "cnn",
        },
        "melody_transformer": {
            "input_size": (32, 64),  # LSTM input (seq, features)
            "output_size": 128,
            "model_type": "lstm",
        },
        "harmony_predictor": {
            "input_size": (128,),  # MLP input
            "output_size": 48,
            "model_type": "mlp",
        },
        "dynamics_engine": {
            "input_size": (32,),
            "output_size": 16,
            "model_type": "mlp",
        },
        "groove_predictor": {
            "input_size": (32, 64),
            "output_size": 32,
            "model_type": "lstm",
        },
    }
    
    if model_name not in configs:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(configs.keys())}")
        return
    
    config = configs[model_name]
    
    # Create synthetic data
    n_samples = 200 if quick else 1000
    batch_size = 16
    
    print(f"Creating synthetic data: {n_samples} samples")
    
    if config["model_type"] == "cnn":
        x = torch.randn(n_samples, *config["input_size"])
    else:
        x = torch.randn(n_samples, *config["input_size"])
    
    y = torch.randint(0, config["output_size"], (n_samples,))
    
    dataset = TensorDataset(x, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Build model
    print(f"Building {config['model_type'].upper()} model...")
    
    if config["model_type"] == "cnn":
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, config["output_size"]),
        )
    elif config["model_type"] == "lstm":
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                return self.fc(h_n[-1])
        
        model = SimpleLSTM(config["input_size"][1], 64, config["output_size"])
    else:
        model = nn.Sequential(
            nn.Linear(config["input_size"][0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config["output_size"]),
        )
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 40)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = loss_fn(output, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(output, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.2%}")
    
    elapsed = time.time() - start_time
    print("-" * 40)
    print(f"Training completed in {elapsed:.1f}s")
    
    # Export demo
    print("\nExport demo:")
    
    # ONNX export
    try:
        dummy_input = torch.randn(1, *config["input_size"]).to("cpu")
        model = model.to("cpu")
        
        onnx_path = ROOT / "models" / f"{model_name}_demo.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
        )
        print(f"✓ ONNX exported: {onnx_path}")
    except Exception as e:
        print(f"○ ONNX export skipped: {e}")
    
    # Core ML export (macOS only)
    try:
        import coremltools as ct
        import platform
        
        if platform.system() == "Darwin":
            traced = torch.jit.trace(model, dummy_input)
            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
            )
            coreml_path = ROOT / "models" / f"{model_name}_demo.mlmodel"
            mlmodel.save(coreml_path)
            print(f"✓ Core ML exported: {coreml_path}")
    except Exception as e:
        print(f"○ Core ML export skipped: {e}")
    
    print("\n" + "=" * 60)
    print("Demo complete! Your environment is ready for training.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Prepare your training data in data/raw/")
    print("  2. Create metadata.csv with labels")
    print("  3. Run: python scripts/train.py --model emotion_recognizer")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Kelly ML Quick Start Training Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="emotion_recognizer",
        choices=[
            "emotion_recognizer",
            "melody_transformer",
            "harmony_predictor",
            "dynamics_engine",
            "groove_predictor",
        ],
        help="Model to train",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--quick", action="store_true", help="Use smaller dataset")
    parser.add_argument("--check-only", action="store_true", help="Only check environment")
    
    args = parser.parse_args()
    
    if not check_environment():
        sys.exit(1)
    
    if args.check_only:
        return
    
    run_quick_demo(args.model, args.epochs, args.quick)


if __name__ == "__main__":
    main()

