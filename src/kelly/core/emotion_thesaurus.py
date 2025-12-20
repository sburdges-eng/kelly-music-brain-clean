"""Core emotion processing and thesaurus."""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class EmotionCategory(Enum):
    """Primary emotion categories."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


@dataclass
class EmotionNode:
    """Represents a node in the 216-node emotion thesaurus."""
    id: int
    name: str
    category: EmotionCategory
    intensity: float  # 0.0 to 1.0
    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (excited)
    related_emotions: List[int]
    musical_attributes: Dict[str, any]


class EmotionThesaurus:
    """
    216-node emotion thesaurus for mapping emotions to musical properties.
    
    The thesaurus organizes emotions in a hierarchical structure with
    dimensions of valence, arousal, and intensity.
    """
    
    def __init__(self) -> None:
        """Initialize the emotion thesaurus."""
        self.nodes: Dict[int, EmotionNode] = {}
        self._initialize_thesaurus()
    
    def _initialize_thesaurus(self) -> None:
        """Initialize the 216-node emotion network."""
        # Initialize core emotion nodes (simplified for initial implementation)
        # Full implementation would have 216 nodes
        base_emotions = [
            (0, "euphoria", EmotionCategory.JOY, 1.0, 1.0, 1.0),
            (1, "contentment", EmotionCategory.JOY, 0.5, 0.7, 0.3),
            (2, "grief", EmotionCategory.SADNESS, 1.0, -0.9, 0.7),
            (3, "melancholy", EmotionCategory.SADNESS, 0.6, -0.6, 0.3),
            (4, "rage", EmotionCategory.ANGER, 1.0, -0.8, 1.0),
            (5, "annoyance", EmotionCategory.ANGER, 0.4, -0.4, 0.5),
            (6, "terror", EmotionCategory.FEAR, 1.0, -0.9, 1.0),
            (7, "anxiety", EmotionCategory.FEAR, 0.6, -0.5, 0.8),
        ]
        
        for node_id, name, category, intensity, valence, arousal in base_emotions:
            self.nodes[node_id] = EmotionNode(
                id=node_id,
                name=name,
                category=category,
                intensity=intensity,
                valence=valence,
                arousal=arousal,
                related_emotions=[],
                musical_attributes={
                    "tempo_modifier": 1.0 + (arousal - 0.5) * 0.5,
                    "mode": "major" if valence > 0 else "minor",
                    "dynamics": intensity,
                }
            )
    
    def get_emotion(self, emotion_id: int) -> Optional[EmotionNode]:
        """Get emotion node by ID."""
        return self.nodes.get(emotion_id)
    
    def find_emotion_by_name(self, name: str) -> Optional[EmotionNode]:
        """Find emotion by name."""
        for node in self.nodes.values():
            if node.name.lower() == name.lower():
                return node
        return None
    
    def get_nearby_emotions(
        self, emotion_id: int, threshold: float = 0.3
    ) -> List[EmotionNode]:
        """
        Find emotions near the given emotion in emotional space.
        
        Args:
            emotion_id: ID of the source emotion
            threshold: Maximum distance in emotional space
            
        Returns:
            List of nearby emotion nodes
        """
        source = self.get_emotion(emotion_id)
        if not source:
            return []
        
        nearby = []
        for node in self.nodes.values():
            if node.id == emotion_id:
                continue
            
            # Calculate emotional distance
            distance = (
                (source.valence - node.valence) ** 2 +
                (source.arousal - node.arousal) ** 2 +
                (source.intensity - node.intensity) ** 2
            ) ** 0.5
            
            if distance < threshold:
                nearby.append(node)
        
        return nearby
