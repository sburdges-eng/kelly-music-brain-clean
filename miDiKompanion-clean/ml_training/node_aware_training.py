#!/usr/bin/env python3
"""
node_aware_training.py - Node-Aware Training with 216-Node Emotion Thesaurus
============================================================================

Agent 2: ML Training Specialist (Week 3-6)
Purpose: Training that uses 216-node emotion thesaurus structure for context-aware generation.

This script:
1. Labels training data with node IDs (0-215)
2. Includes VAD coordinates for each sample
3. Maps MIDI sequences to node musical attributes
4. Creates node relationship graphs for context
5. Uses node relationships for data augmentation
6. Trains on node transitions (emotional journeys)
7. Incorporates node musical attributes as conditioning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

# Import model definitions
try:
    from train_all_models import (
        EmotionRecognizer,
        MelodyTransformer,
        HarmonyPredictor,
        DynamicsEngine,
        GroovePredictor
    )
except ImportError:
    print("Error: Could not import model definitions from train_all_models.py")
    exit(1)


# =============================================================================
# Node-Aware Dataset
# =============================================================================

class NodeAwareDataset(Dataset):
    """
    Dataset that includes 216-node emotion thesaurus labels.

    Each sample includes:
    - Audio features (128-dim)
    - Node ID (0-215)
    - VAD coordinates (valence, arousal, dominance, intensity)
    - Musical attributes (tempo, mode, dynamics, etc.)
    - Related node IDs (for context)
    """

    def __init__(
        self,
        audio_features: np.ndarray,
        node_ids: np.ndarray,
        vad_coords: np.ndarray,  # (N, 4) - valence, arousal, dominance, intensity
        musical_attrs: Optional[Dict] = None,
        related_nodes: Optional[List[List[int]]] = None
    ):
        self.audio_features = torch.FloatTensor(audio_features)
        self.node_ids = torch.LongTensor(node_ids)
        self.vad_coords = torch.FloatTensor(vad_coords)
        self.musical_attrs = musical_attrs or {}
        self.related_nodes = related_nodes or []

    def __len__(self):
        return len(self.audio_features)

    def __getitem__(self, idx):
        sample = {
            'audio_features': self.audio_features[idx],
            'node_id': self.node_ids[idx],
            'vad': self.vad_coords[idx],
        }

        # Add musical attributes if available
        if self.musical_attrs:
            for key, values in self.musical_attrs.items():
                if idx < len(values):
                    sample[key] = torch.FloatTensor([values[idx]])

        # Add related nodes if available
        if self.related_nodes and idx < len(self.related_nodes):
            sample['related_nodes'] = torch.LongTensor(self.related_nodes[idx])

        return sample


# =============================================================================
# Node Relationship Augmentation
# =============================================================================

def augment_with_node_relationships(
    dataset: NodeAwareDataset,
    node_relationship_graph: Dict[int, List[int]],
    augmentation_factor: int = 2
) -> NodeAwareDataset:
    """
    Augment dataset using node relationships.

    For each sample, create augmented samples using related nodes.
    This helps the model learn node transitions and emotional journeys.
    """
    augmented_features = []
    augmented_node_ids = []
    augmented_vad = []

    for i in range(len(dataset)):
        # Original sample
        sample = dataset[i]
        augmented_features.append(sample['audio_features'].numpy())
        augmented_node_ids.append(sample['node_id'].item())
        augmented_vad.append(sample['vad'].numpy())

        # Augment with related nodes
        node_id = sample['node_id'].item()
        if node_id in node_relationship_graph:
            related = node_relationship_graph[node_id][:augmentation_factor]
            for related_id in related:
                # Create augmented sample with related node's VAD
                # (keep original audio features, use related node's VAD)
                augmented_features.append(sample['audio_features'].numpy())
                augmented_node_ids.append(related_id)
                # Use related node's VAD (would need to look up from thesaurus)
                augmented_vad.append(sample['vad'].numpy())  # Placeholder

    return NodeAwareDataset(
        np.array(augmented_features),
        np.array(augmented_node_ids),
        np.array(augmented_vad)
    )


# =============================================================================
# Node-Aware Training Functions
# =============================================================================

def train_emotion_recognizer_with_nodes(
    model: EmotionRecognizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001
):
    """
    Train EmotionRecognizer with node-aware loss.

    Loss includes:
    - Standard reconstruction loss (audio → embedding)
    - Node classification loss (embedding → node ID)
    - VAD regression loss (embedding → VAD coordinates)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Add node classifier head
    node_classifier = nn.Linear(64, 216).to(device)  # 216 nodes
    vad_regressor = nn.Linear(64, 4).to(device)  # VAD + intensity

    optimizer = optim.Adam(
        list(model.parameters()) + list(node_classifier.parameters()) + list(vad_regressor.parameters()),
        lr=learning_rate
    )

    criterion_embedding = nn.MSELoss()
    criterion_node = nn.CrossEntropyLoss()
    criterion_vad = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            audio_features = batch['audio_features'].to(device)
            node_ids = batch['node_id'].to(device)
            vad_targets = batch['vad'].to(device)

            # Forward pass
            embeddings = model(audio_features)

            # Node classification
            node_logits = node_classifier(embeddings)
            node_loss = criterion_node(node_logits, node_ids)

            # VAD regression
            vad_pred = vad_regressor(embeddings)
            vad_loss = criterion_vad(vad_pred, vad_targets)

            # Combined loss
            loss = node_loss + 0.5 * vad_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model, node_classifier, vad_regressor


def train_melody_transformer_with_nodes(
    model: MelodyTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001
):
    """
    Train MelodyTransformer with node musical attributes as conditioning.

    Uses node's musical attributes (tempo, mode, dynamics) to condition generation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()  # Binary cross-entropy for note probabilities

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Get node embedding (from EmotionRecognizer or VAD)
            node_embeddings = batch.get('node_embedding', batch['vad'][:, :3]).to(device)

            # Get musical attributes if available
            musical_conditioning = None
            if 'tempo' in batch:
                tempo = batch['tempo'].to(device)
                mode = batch.get('mode', torch.zeros_like(tempo)).to(device)
                dynamics = batch.get('dynamics', torch.zeros_like(tempo)).to(device)
                musical_conditioning = torch.cat([tempo, mode, dynamics], dim=1)

            # Concatenate node embedding with musical conditioning
            if musical_conditioning is not None:
                model_input = torch.cat([node_embeddings, musical_conditioning], dim=1)
            else:
                model_input = node_embeddings

            # Pad/truncate to model input size (64)
            if model_input.size(1) < 64:
                padding = torch.zeros(model_input.size(0), 64 - model_input.size(1)).to(device)
                model_input = torch.cat([model_input, padding], dim=1)
            elif model_input.size(1) > 64:
                model_input = model_input[:, :64]

            # Forward pass
            midi_probs = model(model_input)
            midi_targets = batch['midi_notes'].to(device)

            loss = criterion(midi_probs, midi_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Node-aware training with 216-node thesaurus')
    parser.add_argument('--data-dir', type=str, default='datasets/prepared',
                       help='Directory containing prepared datasets')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--augment', action='store_true',
                       help='Use node relationship augmentation')

    args = parser.parse_args()

    # Load node relationship graph (would be loaded from thesaurus JSON)
    # For now, create a simple example
    node_relationship_graph = {}  # Would load from emotion thesaurus

    print("Node-aware training with 216-node emotion thesaurus")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Augmentation: {args.augment}")
    print("=" * 60)

    # Note: In full implementation, would:
    # 1. Load prepared dataset with node labels
    # 2. Create NodeAwareDataset
    # 3. Apply augmentation if requested
    # 4. Train models with node-aware loss
    # 5. Validate against node structure
    # 6. Save models with node mapping metadata

    print("\nNode-aware training pipeline ready.")
    print("Full implementation requires:")
    print("  - Prepared dataset with node labels")
    print("  - 216-node emotion thesaurus JSON")
    print("  - Node relationship graph")
    print("  - Musical attributes mapping")


if __name__ == '__main__':
    main()
