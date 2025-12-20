"""
NodeMLMapper - Python implementation of the ML-to-Node bridge.

Maps ML emotion detection results to the 216-node emotion thesaurus,
providing rich musical parameter extraction based on detected emotional states.

This mirrors the C++ NodeMLMapper (src/ml/NodeMLMapper.cpp) but provides
a pure Python implementation for use with the music_brain API.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class VADCoordinates:
    """Valence-Arousal-Dominance coordinates in emotional space."""
    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (excited)
    dominance: float  # 0.0 (submissive) to 1.0 (dominant)
    intensity: float = 0.5  # 0.0 to 1.0

    def distance_to(self, other: "VADCoordinates") -> float:
        """Calculate Euclidean distance to another VAD point."""
        return math.sqrt(
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "intensity": self.intensity,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VADCoordinates":
        """Create from dictionary."""
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.5),
            dominance=data.get("dominance", 0.5),
            intensity=data.get("intensity", 0.5),
        )


@dataclass
class EmotionNode:
    """A single node in the 216-node emotion thesaurus."""
    id: int
    name: str
    category: str  # e.g., "joy", "sad", "anger", "fear", "surprise", "disgust"
    subcategory: str  # e.g., "euphoria", "contentment", "grief", "despair"
    vad: VADCoordinates
    related_emotions: List[int] = field(default_factory=list)
    
    # Musical parameters
    mode: str = "major"  # "major" or "minor"
    tempo_multiplier: float = 1.0  # Multiplier for base tempo
    dynamics_scale: float = 0.75  # 0.0 to 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "subcategory": self.subcategory,
            "vad": self.vad.to_dict(),
            "relatedEmotions": self.related_emotions,
            "mode": self.mode,
            "tempoMultiplier": self.tempo_multiplier,
            "dynamicsScale": self.dynamics_scale,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EmotionNode":
        """Create from dictionary (matches JSON structure)."""
        vad_data = data.get("vad", {})
        return cls(
            id=data.get("id", 0),
            name=data.get("name", "unknown"),
            category=data.get("category", "unknown"),
            subcategory=data.get("subcategory", "unknown"),
            vad=VADCoordinates.from_dict(vad_data),
            related_emotions=data.get("relatedEmotions", []),
            mode=data.get("mode", "major"),
            tempo_multiplier=data.get("tempoMultiplier", 1.0),
            dynamics_scale=data.get("dynamicsScale", 0.75),
        )


@dataclass
class NodeContext:
    """Context for an emotion node including related nodes and transitions."""
    node: EmotionNode
    confidence: float
    related_nodes: List[EmotionNode] = field(default_factory=list)
    category_siblings: List[EmotionNode] = field(default_factory=list)
    intensity_neighbors: List[EmotionNode] = field(default_factory=list)


@dataclass
class MusicalMapping:
    """Musical parameters derived from an emotion node."""
    mode: str  # "major" or "minor"
    tempo_multiplier: float  # Apply to base tempo (e.g., 120 * multiplier)
    dynamics_scale: float  # 0.0 to 1.0 for velocity scaling
    
    # Additional derived parameters
    suggested_tempo_bpm: int = 120  # Based on tempo_multiplier * 120
    velocity_range: Tuple[int, int] = (60, 100)  # MIDI velocity range
    dissonance_level: float = 0.3  # 0.0 to 1.0
    
    @classmethod
    def from_node(cls, node: EmotionNode, base_tempo: int = 120) -> "MusicalMapping":
        """Create musical mapping from an emotion node."""
        suggested_tempo = int(base_tempo * node.tempo_multiplier)
        
        # Calculate velocity range based on dynamics scale
        min_vel = int(40 + node.dynamics_scale * 30)  # 40-70
        max_vel = int(80 + node.dynamics_scale * 47)  # 80-127
        
        # Estimate dissonance from valence and category
        if node.category in ("anger", "fear", "disgust"):
            dissonance = 0.4 + (1 - node.vad.valence) * 0.3
        elif node.category == "sad":
            dissonance = 0.3 + abs(node.vad.valence) * 0.2
        else:
            dissonance = 0.2 + (1 - node.vad.valence) * 0.15
        
        return cls(
            mode=node.mode,
            tempo_multiplier=node.tempo_multiplier,
            dynamics_scale=node.dynamics_scale,
            suggested_tempo_bpm=suggested_tempo,
            velocity_range=(min_vel, max_vel),
            dissonance_level=min(1.0, max(0.0, dissonance)),
        )


class NodeMLMapper:
    """
    Maps ML emotion detection outputs to the 216-node emotion thesaurus.
    
    This provides:
    - VAD coordinate to node mapping
    - Category and subcategory queries
    - Related emotion discovery
    - Musical parameter extraction
    - Smooth emotional transitions
    
    Example:
        >>> mapper = NodeMLMapper()
        >>> # Find node from VAD coordinates
        >>> node = mapper.find_nearest_node(valence=0.7, arousal=0.5, dominance=0.6)
        >>> print(f"Matched: {node.name} ({node.category}/{node.subcategory})")
        >>>
        >>> # Get musical parameters
        >>> params = mapper.get_musical_mapping(node)
        >>> print(f"Mode: {params.mode}, Tempo: {params.suggested_tempo_bpm}")
        >>>
        >>> # Find related emotions for transitions
        >>> related = mapper.get_related_nodes(node.id)
        >>> for rel in related:
        ...     print(f"  Related: {rel.name}")
    """

    def __init__(self, thesaurus_path: Optional[Union[str, Path]] = None):
        """
        Initialize the NodeMLMapper.
        
        Args:
            thesaurus_path: Path to emotion_nodes.json. If None, uses default location.
        """
        self.nodes: Dict[int, EmotionNode] = {}
        self.category_index: Dict[str, List[int]] = {}
        self.subcategory_index: Dict[str, List[int]] = {}
        
        # Load thesaurus
        if thesaurus_path is None:
            # Default: look relative to this file's location
            default_path = Path(__file__).parent.parent.parent / "emotion_thesaurus" / "emotion_nodes.json"
            thesaurus_path = default_path
        
        self.load_thesaurus(thesaurus_path)

    def load_thesaurus(self, path: Union[str, Path]) -> bool:
        """
        Load the emotion thesaurus from JSON file.
        
        Args:
            path: Path to emotion_nodes.json
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        path = Path(path)
        if not path.exists():
            print(f"Warning: Thesaurus not found at {path}")
            return False
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.nodes.clear()
            self.category_index.clear()
            self.subcategory_index.clear()
            
            for node_data in data.get("nodes", []):
                node = EmotionNode.from_dict(node_data)
                self.nodes[node.id] = node
                
                # Build category index
                if node.category not in self.category_index:
                    self.category_index[node.category] = []
                self.category_index[node.category].append(node.id)
                
                # Build subcategory index
                subcat_key = f"{node.category}:{node.subcategory}"
                if subcat_key not in self.subcategory_index:
                    self.subcategory_index[subcat_key] = []
                self.subcategory_index[subcat_key].append(node.id)
            
            return True
            
        except Exception as e:
            print(f"Error loading thesaurus: {e}")
            return False

    def find_nearest_node(
        self,
        valence: float,
        arousal: float,
        dominance: float = 0.5,
        intensity: float = 0.5,
    ) -> Optional[EmotionNode]:
        """
        Find the nearest emotion node to given VAD coordinates.
        
        Args:
            valence: -1.0 to 1.0 (negative to positive)
            arousal: 0.0 to 1.0 (calm to excited)
            dominance: 0.0 to 1.0 (submissive to dominant)
            intensity: 0.0 to 1.0 (optional intensity filter)
            
        Returns:
            The nearest EmotionNode, or None if no nodes loaded.
        """
        if not self.nodes:
            return None
        
        target = VADCoordinates(valence, arousal, dominance, intensity)
        
        nearest_node = None
        min_distance = float("inf")
        
        for node in self.nodes.values():
            distance = target.distance_to(node.vad)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node

    def find_nodes_in_range(
        self,
        valence: float,
        arousal: float,
        dominance: float = 0.5,
        radius: float = 0.3,
        max_results: int = 5,
    ) -> List[Tuple[EmotionNode, float]]:
        """
        Find all nodes within a radius of given VAD coordinates.
        
        Args:
            valence: Center valence
            arousal: Center arousal
            dominance: Center dominance
            radius: Maximum distance from center
            max_results: Maximum number of results to return
            
        Returns:
            List of (node, distance) tuples, sorted by distance.
        """
        target = VADCoordinates(valence, arousal, dominance)
        results = []
        
        for node in self.nodes.values():
            distance = target.distance_to(node.vad)
            if distance <= radius:
                results.append((node, distance))
        
        # Sort by distance and limit results
        results.sort(key=lambda x: x[1])
        return results[:max_results]

    def get_node(self, node_id: int) -> Optional[EmotionNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)

    def get_nodes_by_category(self, category: str) -> List[EmotionNode]:
        """Get all nodes in a category (e.g., 'joy', 'sad', 'anger')."""
        node_ids = self.category_index.get(category.lower(), [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_nodes_by_subcategory(
        self, category: str, subcategory: str
    ) -> List[EmotionNode]:
        """Get all nodes in a specific subcategory."""
        key = f"{category.lower()}:{subcategory.lower()}"
        node_ids = self.subcategory_index.get(key, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_related_nodes(self, node_id: int) -> List[EmotionNode]:
        """Get related emotion nodes for smooth transitions."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        return [
            self.nodes[rel_id]
            for rel_id in node.related_emotions
            if rel_id in self.nodes
        ]

    def get_node_context(self, node_id: int) -> Optional[NodeContext]:
        """
        Get full context for a node including related nodes and siblings.
        
        Args:
            node_id: The node ID to get context for.
            
        Returns:
            NodeContext with related nodes, or None if node not found.
        """
        node = self.nodes.get(node_id)
        if not node:
            return None
        
        # Get related nodes
        related = self.get_related_nodes(node_id)
        
        # Get category siblings (same category, different subcategory)
        category_nodes = self.get_nodes_by_category(node.category)
        siblings = [n for n in category_nodes if n.id != node_id][:5]
        
        # Get intensity neighbors (same subcategory, different intensity)
        subcat_nodes = self.get_nodes_by_subcategory(node.category, node.subcategory)
        intensity_neighbors = [n for n in subcat_nodes if n.id != node_id]
        
        return NodeContext(
            node=node,
            confidence=1.0,
            related_nodes=related,
            category_siblings=siblings,
            intensity_neighbors=intensity_neighbors,
        )

    def get_musical_mapping(
        self, node: EmotionNode, base_tempo: int = 120
    ) -> MusicalMapping:
        """
        Get musical parameters for an emotion node.
        
        Args:
            node: The emotion node
            base_tempo: Base tempo to apply multiplier to
            
        Returns:
            MusicalMapping with tempo, mode, dynamics, etc.
        """
        return MusicalMapping.from_node(node, base_tempo)

    def get_transition_path(
        self,
        from_node_id: int,
        to_node_id: int,
        steps: int = 5,
    ) -> List[EmotionNode]:
        """
        Get a smooth transition path between two emotion nodes.
        
        Args:
            from_node_id: Starting node ID
            to_node_id: Ending node ID
            steps: Number of intermediate steps
            
        Returns:
            List of nodes forming the transition path.
        """
        from_node = self.nodes.get(from_node_id)
        to_node = self.nodes.get(to_node_id)
        
        if not from_node or not to_node:
            return []
        
        path = [from_node]
        
        # Interpolate through VAD space and find nearest nodes
        for i in range(1, steps):
            t = i / steps
            interp_vad = VADCoordinates(
                valence=from_node.vad.valence + t * (to_node.vad.valence - from_node.vad.valence),
                arousal=from_node.vad.arousal + t * (to_node.vad.arousal - from_node.vad.arousal),
                dominance=from_node.vad.dominance + t * (to_node.vad.dominance - from_node.vad.dominance),
            )
            
            # Find nearest node to interpolated position
            nearest = self.find_nearest_node(
                valence=interp_vad.valence,
                arousal=interp_vad.arousal,
                dominance=interp_vad.dominance,
            )
            
            if nearest and nearest.id not in [n.id for n in path]:
                path.append(nearest)
        
        if to_node.id not in [n.id for n in path]:
            path.append(to_node)
        
        return path

    def map_basic_emotion(self, emotion: str, confidence: float = 0.8) -> Optional[EmotionNode]:
        """
        Map a basic emotion label to the best matching node.
        
        This is useful for integrating with EmotionDetector which outputs
        basic labels like "happy", "sad", "angry", "neutral".
        
        Args:
            emotion: Basic emotion label (happy, sad, angry, neutral, etc.)
            confidence: Detection confidence (0-1)
            
        Returns:
            Best matching EmotionNode or None.
        """
        # Map basic emotions to VAD coordinates
        emotion_vad_map = {
            "happy": VADCoordinates(0.7, 0.6, 0.6, confidence),
            "sad": VADCoordinates(-0.6, 0.25, 0.3, confidence),
            "angry": VADCoordinates(-0.5, 0.8, 0.8, confidence),
            "neutral": VADCoordinates(0.0, 0.4, 0.5, confidence),
            "fear": VADCoordinates(-0.7, 0.6, 0.2, confidence),
            "surprise": VADCoordinates(0.2, 0.7, 0.5, confidence),
            "disgust": VADCoordinates(-0.6, 0.4, 0.6, confidence),
            "calm": VADCoordinates(0.5, 0.3, 0.5, confidence),
            "excited": VADCoordinates(0.8, 0.9, 0.7, confidence),
            "anxious": VADCoordinates(-0.5, 0.7, 0.3, confidence),
        }
        
        vad = emotion_vad_map.get(emotion.lower())
        if not vad:
            # Try to find a category match
            category_nodes = self.get_nodes_by_category(emotion.lower())
            if category_nodes:
                # Return middle intensity node
                mid_idx = len(category_nodes) // 2
                return category_nodes[mid_idx]
            return None
        
        return self.find_nearest_node(
            valence=vad.valence,
            arousal=vad.arousal,
            dominance=vad.dominance,
        )

    def get_categories(self) -> List[str]:
        """Get all available emotion categories."""
        return list(self.category_index.keys())

    def get_subcategories(self, category: str) -> List[str]:
        """Get all subcategories for a category."""
        prefix = f"{category.lower()}:"
        return [
            key.split(":")[1]
            for key in self.subcategory_index.keys()
            if key.startswith(prefix)
        ]

    @property
    def node_count(self) -> int:
        """Get the total number of nodes loaded."""
        return len(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


# Convenience function
def create_mapper(thesaurus_path: Optional[str] = None) -> NodeMLMapper:
    """Create a NodeMLMapper with default settings."""
    return NodeMLMapper(thesaurus_path)
