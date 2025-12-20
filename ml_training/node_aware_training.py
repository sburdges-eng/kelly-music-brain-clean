#!/usr/bin/env python3
"""
Node-Aware Training for 216-Node Emotion Thesaurus Integration

This module provides training utilities that integrate with the 216-node
emotion thesaurus structure:
- 6 Base emotions × 6 Sub-emotions × 6 Intensity levels = 216 nodes
- VAD (Valence-Arousal-Dominance-Intensity) coordinates per node
- Node relationships for context-aware generation

Usage:
    python node_aware_training.py --thesaurus ../emotion_thesaurus/nodes.json
"""

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VADCoordinates:
    """Valence-Arousal-Dominance-Intensity coordinates."""
    valence: float  # -1.0 to 1.0
    arousal: float  # 0.0 to 1.0
    dominance: float  # 0.0 to 1.0
    intensity: float  # 0.0 to 1.0

    def to_numpy(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.dominance, self.intensity], 
                        dtype=np.float32)

    def distance_to(self, other: "VADCoordinates") -> float:
        """Weighted Euclidean distance."""
        dv = self.valence - other.valence
        da = self.arousal - other.arousal
        dd = self.dominance - other.dominance
        di = self.intensity - other.intensity
        return math.sqrt(2 * dv**2 + 2 * da**2 + dd**2 + di**2)


@dataclass
class EmotionNode:
    """Node in the 216-node emotion thesaurus."""
    id: int
    name: str
    category: str
    subcategory: str
    vad: VADCoordinates
    related_emotions: List[int]
    mode: str = "major"
    tempo_multiplier: float = 1.0
    dynamics_scale: float = 1.0


class EmotionThesaurus:
    """216-node emotion thesaurus manager."""
    
    def __init__(self):
        self.nodes: List[EmotionNode] = []
        self.category_index: Dict[str, List[int]] = {}
    
    def load_from_json(self, path: Path) -> bool:
        """Load thesaurus from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            
            for node_data in data.get("nodes", []):
                vad = VADCoordinates(
                    valence=node_data.get("vad", {}).get("valence", 0.0),
                    arousal=node_data.get("vad", {}).get("arousal", 0.5),
                    dominance=node_data.get("vad", {}).get("dominance", 0.5),
                    intensity=node_data.get("vad", {}).get("intensity", 0.5)
                )
                
                node = EmotionNode(
                    id=node_data.get("id", len(self.nodes)),
                    name=node_data.get("name", ""),
                    category=node_data.get("category", ""),
                    subcategory=node_data.get("subcategory", ""),
                    vad=vad,
                    related_emotions=node_data.get("relatedEmotions", []),
                    mode=node_data.get("mode", "major"),
                    tempo_multiplier=node_data.get("tempoMultiplier", 1.0),
                    dynamics_scale=node_data.get("dynamicsScale", 1.0)
                )
                
                self.nodes.append(node)
            
            self._build_category_index()
            return True
        except Exception as e:
            logger.error(f"Failed to load thesaurus: {e}")
            return False
    
    def initialize_default(self):
        """Create default 216-node thesaurus."""
        categories = [
            ("happy", 0.8, 0.6, 0.6, "major", 1.1),
            ("sad", -0.6, 0.3, 0.3, "minor", 0.8),
            ("angry", -0.5, 0.8, 0.8, "minor", 1.2),
            ("fear", -0.7, 0.7, 0.2, "minor", 0.9),
            ("surprise", 0.3, 0.8, 0.5, "major", 1.15),
            ("disgust", -0.6, 0.5, 0.6, "minor", 0.85)
        ]
        
        node_id = 0
        for cat, base_v, base_a, base_d, mode, tempo in categories:
            for sub in range(6):
                for intensity in range(6):
                    # Vary VAD based on sub-emotion and intensity
                    sub_var = (sub - 2.5) / 5.0 * 0.3
                    int_scale = (intensity + 1) / 6.0
                    
                    vad = VADCoordinates(
                        valence=max(-1, min(1, base_v + sub_var)),
                        arousal=max(0, min(1, base_a * (0.5 + int_scale * 0.5))),
                        dominance=max(0, min(1, base_d + sub_var * 0.5)),
                        intensity=int_scale
                    )
                    
                    # Create related emotions list
                    related = []
                    for r in range(-3, 4):
                        rel_id = node_id + r
                        if 0 <= rel_id < 216 and rel_id != node_id:
                            related.append(rel_id)
                    
                    node = EmotionNode(
                        id=node_id,
                        name=f"{cat}_{sub}_{intensity}",
                        category=cat,
                        subcategory=f"{cat}_{sub}",
                        vad=vad,
                        related_emotions=related,
                        mode=mode,
                        tempo_multiplier=tempo * (0.9 + int_scale * 0.2),
                        dynamics_scale=0.5 + int_scale * 0.5
                    )
                    
                    self.nodes.append(node)
                    node_id += 1
        
        self._build_category_index()
    
    def _build_category_index(self):
        """Build category lookup index."""
        self.category_index.clear()
        for node in self.nodes:
            if node.category not in self.category_index:
                self.category_index[node.category] = []
            self.category_index[node.category].append(node.id)
    
    def get_node(self, node_id: int) -> Optional[EmotionNode]:
        """Get node by ID."""
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None
    
    def find_nearest_node(self, vad: VADCoordinates) -> int:
        """Find nearest node by VAD distance."""
        min_dist = float("inf")
        nearest_id = 0
        
        for node in self.nodes:
            dist = vad.distance_to(node.vad)
            if dist < min_dist:
                min_dist = dist
                nearest_id = node.id
        
        return nearest_id
    
    def get_context(self, node_id: int) -> List[EmotionNode]:
        """Get context nodes (related emotions)."""
        node = self.get_node(node_id)
        if not node:
            return []
        
        return [self.nodes[rid] for rid in node.related_emotions 
                if 0 <= rid < len(self.nodes)]


class NodeAwareDataset(Dataset):
    """
    Dataset that uses 216-node thesaurus for training.
    Creates node-aware samples for more coherent emotion generation.
    """
    
    def __init__(
        self, 
        thesaurus: EmotionThesaurus,
        num_samples: int = 10000,
        model_type: str = "emotion"
    ):
        self.thesaurus = thesaurus
        self.num_samples = num_samples
        self.model_type = model_type
        self.data = self._generate_data()
    
    def _vad_to_embedding(self, vad: VADCoordinates, dim: int = 64) -> np.ndarray:
        """Convert VAD to embedding space."""
        embedding = np.zeros(dim, dtype=np.float32)
        
        # Use Fourier-like encoding
        for i in range(dim // 4):
            phase = i / (dim // 4) * 2 * np.pi
            embedding[i] = vad.valence * np.cos(phase) + vad.arousal * np.sin(phase)
            embedding[i + dim//4] = vad.arousal * np.cos(phase) + vad.dominance * np.sin(phase)
            embedding[i + dim//2] = vad.dominance * np.cos(phase) + vad.intensity * np.sin(phase)
            embedding[i + 3*dim//4] = vad.intensity * np.cos(phase) + (vad.valence + vad.arousal) * 0.5 * np.sin(phase)
        
        return embedding
    
    def _generate_data(self) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Generate node-aware training data."""
        data = []
        
        for _ in range(self.num_samples):
            # Pick a random node
            node_id = np.random.randint(0, len(self.thesaurus.nodes))
            node = self.thesaurus.nodes[node_id]
            
            if self.model_type == "emotion":
                # Audio features → Emotion embedding (VAD-based)
                x = np.random.randn(128).astype(np.float32)
                # Add node-specific bias
                x[:4] = node.vad.to_numpy() + np.random.randn(4).astype(np.float32) * 0.1
                y = self._vad_to_embedding(node.vad)
                
            elif self.model_type == "melody":
                # Emotion embedding → Note probabilities
                x = self._vad_to_embedding(node.vad)
                y = np.zeros(128, dtype=np.float32)
                
                # Center pitch based on valence + arousal
                center = int(60 + node.vad.valence * 12 + node.vad.arousal * 6)
                
                # Spread based on intensity
                spread = 10 + int(node.vad.intensity * 10)
                
                for i in range(128):
                    y[i] = np.exp(-((i - center) ** 2) / (2 * spread ** 2))
                
                # Major vs minor influence
                if node.mode == "minor":
                    y[center + 3] *= 0.5  # Flatten third
                
                y = y / (y.max() + 1e-8)
                
            elif self.model_type == "harmony":
                # Context → Chord probabilities
                # Get context from related nodes
                context_nodes = self.thesaurus.get_context(node_id)
                
                x = np.zeros(128, dtype=np.float32)
                x[:64] = self._vad_to_embedding(node.vad)
                
                # Add context from related nodes
                if context_nodes:
                    avg_vad = VADCoordinates(
                        valence=np.mean([n.vad.valence for n in context_nodes]),
                        arousal=np.mean([n.vad.arousal for n in context_nodes]),
                        dominance=np.mean([n.vad.dominance for n in context_nodes]),
                        intensity=np.mean([n.vad.intensity for n in context_nodes])
                    )
                    x[64:] = self._vad_to_embedding(avg_vad)
                
                # Generate chord distribution
                y = np.random.dirichlet(np.ones(64)).astype(np.float32)
                
                # Bias toward mode-appropriate chords
                if node.mode == "major":
                    y[[0, 4, 7]] *= 2  # I, IV, V
                else:
                    y[[0, 3, 7]] *= 2  # i, iv, V
                
                y = y / y.sum()
                
            elif self.model_type == "dynamics":
                # Intensity → Expression
                x = np.zeros(32, dtype=np.float32)
                x[:4] = node.vad.to_numpy()
                x[4:8] = [node.intensity, node.vad.arousal, node.dynamics_scale, 0.0]
                x[8:] = np.random.randn(24).astype(np.float32) * 0.1
                
                intensity = node.vad.intensity
                y = np.zeros(16, dtype=np.float32)
                y[0] = intensity * 0.8  # Velocity
                y[1] = node.vad.arousal * 0.7  # Attack
                y[2] = (1 - node.vad.arousal) * 0.8  # Release
                y[3] = intensity * 0.5  # Expression CC
                y[4:] = np.random.uniform(0.3, 0.7, 12) * intensity
                
            elif self.model_type == "groove":
                # Arousal → Groove parameters
                x = self._vad_to_embedding(node.vad)
                
                arousal = node.vad.arousal
                y = np.zeros(32, dtype=np.float32)
                y[0] = arousal * 0.4  # Swing amount
                y[1] = (1 - arousal) * 0.3  # Humanize
                y[2] = arousal * node.tempo_multiplier  # Tempo factor
                y[3] = node.vad.intensity * 0.5  # Accent strength
                y[4:] = np.random.uniform(0.2, 0.8, 28) * arousal
                
            else:
                x = np.random.randn(128).astype(np.float32)
                y = np.random.randn(64).astype(np.float32)
            
            data.append((x, y, node_id))
        
        return data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        x, y, node_id = self.data[idx]
        return torch.from_numpy(x), torch.from_numpy(y), node_id


class NodeContextLoss(nn.Module):
    """
    Loss function that considers node relationships.
    Penalizes outputs that are inconsistent with related nodes.
    """
    
    def __init__(self, thesaurus: EmotionThesaurus, context_weight: float = 0.1):
        super().__init__()
        self.thesaurus = thesaurus
        self.context_weight = context_weight
        self.mse = nn.MSELoss()
    
    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        node_ids: torch.Tensor
    ) -> torch.Tensor:
        # Base MSE loss
        base_loss = self.mse(outputs, targets)
        
        # Context consistency loss
        context_loss = 0.0
        batch_size = outputs.size(0)
        
        for i in range(batch_size):
            node_id = node_ids[i].item()
            node = self.thesaurus.get_node(node_id)
            
            if node and node.related_emotions:
                # Check consistency with related nodes
                # (simplified - in practice would use cached embeddings)
                context_loss += outputs[i].var()  # Encourage smooth outputs
        
        context_loss = context_loss / batch_size
        
        return base_loss + self.context_weight * context_loss


def train_with_node_awareness(
    model: nn.Module,
    thesaurus: EmotionThesaurus,
    model_type: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cuda"
) -> Dict:
    """Train model with node-aware dataset."""
    
    logger.info(f"Training {model_type} with node awareness...")
    
    # Create datasets
    train_dataset = NodeAwareDataset(thesaurus, num_samples=10000, model_type=model_type)
    val_dataset = NodeAwareDataset(thesaurus, num_samples=2000, model_type=model_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = model.to(device)
    criterion = NodeContextLoss(thesaurus)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, batch_y, batch_nodes in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y, batch_nodes)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_nodes in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y, batch_nodes)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch + 1}/{epochs}: Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Node-aware ML training")
    parser.add_argument("--thesaurus", type=str, help="Path to thesaurus JSON")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    # Load or create thesaurus
    thesaurus = EmotionThesaurus()
    
    if args.thesaurus and Path(args.thesaurus).exists():
        thesaurus.load_from_json(Path(args.thesaurus))
    else:
        logger.info("Initializing default 216-node thesaurus...")
        thesaurus.initialize_default()
    
    logger.info(f"Thesaurus loaded: {len(thesaurus.nodes)} nodes")
    
    # Example: train emotion recognizer with node awareness
    from train_all_models import EmotionRecognizer
    
    model = EmotionRecognizer(input_dim=128, output_dim=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    history = train_with_node_awareness(
        model=model,
        thesaurus=thesaurus,
        model_type="emotion",
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device
    )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
