#!/usr/bin/env python3
"""
Model Parameter Analysis
========================
Detailed breakdown of model parameters to understand discrepancies.
"""

import torch
import torch.nn as nn

# Model definitions
class EmotionRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.tanh(self.fc3(x))
        return x


class MelodyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def count_layer_params(layer):
    """Count parameters in a single layer."""
    total = 0
    for name, param in layer.named_parameters():
        count = param.numel()
        total += count
        print(f"    {name}: {count:,} ({param.shape})")
    return total


def analyze_model(model, model_name):
    """Analyze model parameters in detail."""
    print(f"\n{'='*70}")
    print(f"{model_name} Parameter Breakdown")
    print(f"{'='*70}\n")

    total_params = 0

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name}:")
                if isinstance(module, nn.Linear):
                    print(f"  Type: Linear")
                    print(f"  Input: {module.in_features}, Output: {module.out_features}")
                    print(f"  Weights: {module.in_features * module.out_features:,}")
                    print(f"  Bias: {module.out_features:,}")
                    print(f"  Total: {params:,}")
                elif isinstance(module, nn.LSTM):
                    print(f"  Type: LSTM")
                    print(f"  Input: {module.input_size}, Hidden: {module.hidden_size}")
                    # LSTM has 4 gates: i, f, g, o
                    # weight_ih: [4*hidden_size, input_size]
                    # weight_hh: [4*hidden_size, hidden_size]
                    # bias_ih: [4*hidden_size] (if bias=True)
                    # bias_hh: [4*hidden_size] (if bias=True)
                    ih_params = 4 * module.hidden_size * module.input_size
                    hh_params = 4 * module.hidden_size * module.hidden_size
                    bias_params = 0
                    if module.bias:
                        bias_params = 2 * 4 * module.hidden_size  # bias_ih + bias_hh
                    print(f"  weight_ih: {ih_params:,}")
                    print(f"  weight_hh: {hh_params:,}")
                    print(f"  bias: {bias_params:,}")
                    print(f"  Total: {params:,}")
                else:
                    print(f"  Type: {type(module).__name__}")
                    print(f"  Total: {params:,}")
                total_params += params
                print()

    print(f"{'='*70}")
    print(f"Total Parameters: {total_params:,}")
    print(f"{'='*70}\n")

    return total_params


# C++ expected values
cpp_specs = {
    "EmotionRecognizer": 497664,
    "MelodyTransformer": 412672,
}

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Model Parameter Analysis")
    print("="*70)

    # Analyze EmotionRecognizer
    emotion_model = EmotionRecognizer()
    emotion_params = analyze_model(emotion_model, "EmotionRecognizer")
    print(f"C++ Expected: {cpp_specs['EmotionRecognizer']:,}")
    print(f"Difference: {abs(emotion_params - cpp_specs['EmotionRecognizer']):,}")

    # Analyze MelodyTransformer
    melody_model = MelodyTransformer()
    melody_params = analyze_model(melody_model, "MelodyTransformer")
    print(f"C++ Expected: {cpp_specs['MelodyTransformer']:,}")
    print(f"Difference: {abs(melody_params - cpp_specs['MelodyTransformer']):,}")
