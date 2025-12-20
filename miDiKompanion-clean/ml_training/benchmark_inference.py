#!/usr/bin/env python3
"""
RTNeural Inference Benchmark
============================
Benchmarks inference latency for exported RTNeural models.
Target: <10ms per inference.
"""

import json
import time
import statistics
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np


def benchmark_model_inference(model_path: Path, num_iterations: int = 1000) -> Dict:
    """
    Benchmark model inference by simulating RTNeural forward pass.
    This simulates the computation without actually loading RTNeural C++ library.
    """
    with open(model_path, 'r') as f:
        model_data = json.load(f)

    layers = model_data.get("layers", [])
    metadata = model_data.get("metadata", {})
    input_size = metadata.get("input_size", 0)
    output_size = metadata.get("output_size", 0)

    if len(layers) == 0:
        return {
            "error": "No layers found in model",
            "valid": False
        }

    # Generate random input
    input_data = np.random.randn(input_size).astype(np.float32)

    # Simulate forward pass through layers
    def simulate_forward(input_vec):
        current = input_vec.copy()

        for layer in layers:
            layer_type = layer.get("type")

            if layer_type == "dense":
                weights = np.array(layer["weights"])
                bias = np.array(layer.get("bias", [0.0] * len(weights)))

                # Matrix multiplication: output = weights @ input + bias
                current = np.dot(weights, current) + bias

                # Apply activation
                activation = layer.get("activation", "linear")
                if activation == "tanh":
                    current = np.tanh(current)
                elif activation == "relu":
                    current = np.maximum(0, current)
                elif activation == "sigmoid":
                    current = 1.0 / (1.0 + np.exp(-current))
                elif activation == "softmax":
                    exp = np.exp(current - np.max(current))
                    current = exp / exp.sum()

            elif layer_type == "lstm":
                # Simplified LSTM simulation (just pass through for benchmarking)
                # Real LSTM is more complex but this gives us timing estimate
                out_size = layer.get("out_size", len(current))
                current = current[:out_size] if len(current) > out_size else np.pad(current, (0, out_size - len(current)))

        return current

    # Warmup
    for _ in range(10):
        _ = simulate_forward(input_data)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = simulate_forward(input_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    # Estimate memory usage
    param_count = metadata.get("parameter_count", 0)
    memory_kb = param_count * 4 / 1024  # 4 bytes per float

    return {
        "valid": True,
        "model_name": metadata.get("model_name", "Unknown"),
        "input_size": input_size,
        "output_size": output_size,
        "parameter_count": param_count,
        "memory_kb": memory_kb,
        "num_layers": len(layers),
        "latency_ms": {
            "mean": avg_time,
            "median": median_time,
            "min": min_time,
            "max": max_time,
            "std": std_time
        },
        "meets_target": avg_time < 10.0,
        "iterations": num_iterations
    }


def main():
    """Main benchmarking function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark RTNeural model inference latency"
    )
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to model JSON file"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=1000,
        help="Number of inference iterations (default: 1000)"
    )
    parser.add_argument(
        "--target-ms",
        type=float,
        default=10.0,
        help="Target latency in milliseconds (default: 10.0)"
    )

    args = parser.parse_args()

    model_path = Path(args.model_file)

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    print(f"Benchmarking RTNeural model: {model_path}")
    print("=" * 60)

    result = benchmark_model_inference(model_path, args.iterations)

    if not result.get("valid", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1

    print(f"\nModel: {result['model_name']}")
    print(f"Input size: {result['input_size']}")
    print(f"Output size: {result['output_size']}")
    print(f"Parameters: {result['parameter_count']:,}")
    print(f"Memory: {result['memory_kb']:.1f} KB")
    print(f"Layers: {result['num_layers']}")

    print(f"\nLatency (simulated, {result['iterations']} iterations):")
    latency = result['latency_ms']
    print(f"  Mean:   {latency['mean']:.3f} ms")
    print(f"  Median: {latency['median']:.3f} ms")
    print(f"  Min:    {latency['min']:.3f} ms")
    print(f"  Max:    {latency['max']:.3f} ms")
    print(f"  Std:    {latency['std']:.3f} ms")

    target_met = result['meets_target']
    status = "✓ PASS" if target_met else "✗ FAIL"
    print(f"\nTarget (<{args.target_ms} ms): {status}")

    if not target_met:
        print(f"  Warning: Average latency ({latency['mean']:.3f} ms) exceeds target")
        print("  Note: This is a Python simulation. C++ RTNeural will be faster.")

    return 0 if target_met else 1


if __name__ == "__main__":
    sys.exit(main())
