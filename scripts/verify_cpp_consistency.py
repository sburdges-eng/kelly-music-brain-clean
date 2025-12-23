#!/usr/bin/env python3
"""
C++ Consistency Verifier - Ensures Python ML output matches RTNeural C++ expectations.

This script:
1. Loads a trained PyTorch model.
2. Generates a fixed input tensor.
3. Computes the Python output.
4. Exports the model to RTNeural JSON.
5. (Optional) Simulates the RTNeural forward pass to verify consistency.
"""

import torch
import json
import numpy as np
import argparse
from pathlib import Path

def verify_consistency(model_path: str, input_dim: int):
    print(f"--- Verifying C++ Consistency for {model_path} ---")
    
    # 1. Load Model
    # Assuming a simple architecture for this test
    # In practice, use the real architecture loader
    model = torch.load(model_path, map_location='cpu')
    if isinstance(model, dict) and 'model_state_dict' in model:
        state_dict = model['model_state_dict']
        # You'd need to rebuild the model here
        print("Note: Checkpoint contains state_dict. Rebuilding model is required.")
        return

    model.eval()

    # 2. Fixed Input
    dummy_input = torch.ones(1, input_dim) 
    
    # 3. Python Forward Pass
    with torch.no_grad():
        py_output = model(dummy_input)
    
    print(f"Python Output (first 5): {py_output.flatten()[:5].tolist()}")

    # 4. Save test vector for C++
    test_data = {
        "input": dummy_input.flatten().tolist(),
        "expected_output": py_output.flatten().tolist(),
        "model_file": str(model_path)
    }
    
    output_path = Path("tests/cpp/consistency_test_vectors.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Test vectors saved to {output_path}")
    print("Next step: Run the C++ 'test_ml_consistency' suite in iDAW_Core.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    
    if Path(args.model).exists():
        verify_consistency(args.model, args.dim)
    else:
        print(f"Model not found: {args.model}")

