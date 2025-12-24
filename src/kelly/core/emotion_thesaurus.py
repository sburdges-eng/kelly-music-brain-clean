"""Core emotion processing and thesaurus.

This module provides a 216-node emotion thesaurus for mapping emotions to
musical properties. Emotions are organized using VAD (Valence-Arousal-Dominance)
dimensions and intensity levels.
"""
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class EmotionCategory(Enum):
    """Primary emotion categories based on Plutchik's emotion wheel."""
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
    """Represents a node in the 216-node emotion thesaurus.
    
    Attributes:
        id: Unique identifier for the emotion node
        name: Human-readable name of the emotion
        category: Primary emotion category
        intensity: Intensity level (0.0 to 1.0)
        valence: Valence dimension (-1.0 negative to 1.0 positive)
        arousal: Arousal dimension (0.0 calm to 1.0 excited)
        related_emotions: List of related emotion IDs
        musical_attributes: Dictionary of musical properties derived from emotion
    """
    id: int
    name: str
    category: EmotionCategory
    intensity: float  # 0.0 to 1.0
    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (excited)
    related_emotions: List[int]
    musical_attributes: Dict[str, Any]


class EmotionThesaurus:
    """
    216-node emotion thesaurus for mapping emotions to musical properties.
    
    The thesaurus organizes emotions in a hierarchical structure with
    dimensions of valence, arousal, and intensity. Each emotion maps to
    musical attributes like tempo, mode, dynamics, and harmonic complexity.
    
    Example:
        >>> thesaurus = EmotionThesaurus()
        >>> grief = thesaurus.find_emotion_by_name("grief")
        >>> print(grief.musical_attributes["mode"])
        'minor'
    """
    
    def __init__(self) -> None:
        """Initialize the emotion thesaurus with base emotion nodes."""
        self.nodes: Dict[int, EmotionNode] = {}
        self._name_to_id: Dict[str, int] = {}
        self._initialize_thesaurus()
        self._setup_relationships()
    
    def _initialize_thesaurus(self) -> None:
        """Initialize the emotion network with base emotions.
        
        Currently implements a subset of the full 216-node system.
        Each emotion is defined with (id, name, category, intensity, valence, arousal).
        """
        base_emotions = [
            # Joy category
            (0, "euphoria", EmotionCategory.JOY, 1.0, 1.0, 1.0),
            (1, "contentment", EmotionCategory.JOY, 0.5, 0.7, 0.3),
            (2, "grief", EmotionCategory.SADNESS, 1.0, -0.9, 0.7),
            (3, "cheerful", EmotionCategory.JOY, 0.6, 0.8, 0.6),
            (4, "blissful", EmotionCategory.JOY, 0.9, 0.95, 0.8),
            
            # Sadness category
            (10, "melancholy", EmotionCategory.SADNESS, 0.6, -0.6, 0.3),
            (11, "sorrow", EmotionCategory.SADNESS, 0.8, -0.8, 0.5),
            (12, "despair", EmotionCategory.SADNESS, 0.95, -0.95, 0.6),
            
            # Anger category
            (20, "rage", EmotionCategory.ANGER, 1.0, -0.8, 1.0),
            (21, "annoyance", EmotionCategory.ANGER, 0.4, -0.4, 0.5),
            (22, "fury", EmotionCategory.ANGER, 0.95, -0.85, 0.95),
            (23, "resentment", EmotionCategory.ANGER, 0.7, -0.7, 0.6),
            
            # Fear category
            (30, "terror", EmotionCategory.FEAR, 1.0, -0.9, 1.0),
            (31, "anxiety", EmotionCategory.FEAR, 0.6, -0.5, 0.8),
            (32, "dread", EmotionCategory.FEAR, 0.85, -0.85, 0.9),
            (33, "worry", EmotionCategory.FEAR, 0.5, -0.4, 0.6),
            
            # Surprise category
            (40, "astonishment", EmotionCategory.SURPRISE, 0.8, 0.3, 0.9),
            (41, "amazement", EmotionCategory.SURPRISE, 0.7, 0.5, 0.8),
            
            # Disgust category
            (50, "revulsion", EmotionCategory.DISGUST, 0.8, -0.7, 0.5),
            (51, "contempt", EmotionCategory.DISGUST, 0.6, -0.6, 0.4),
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
                musical_attributes=self._calculate_musical_attributes(
                    valence, arousal, intensity, category
                )
            )
            self._name_to_id[name.lower()] = node_id
    
    def _calculate_musical_attributes(
        self,
        valence: float,
        arousal: float,
        intensity: float,
        category: EmotionCategory
    ) -> Dict[str, Any]:
        """Calculate musical attributes from emotion dimensions.
        
        Args:
            valence: Valence dimension (-1.0 to 1.0)
            arousal: Arousal dimension (0.0 to 1.0)
            intensity: Intensity level (0.0 to 1.0)
            category: Emotion category
            
        Returns:
            Dictionary of musical attributes
        """
        # Tempo multiplier: higher arousal = faster tempo
        # Range: 0.6x to 1.4x base tempo
        tempo_modifier = 0.6 + (arousal * 0.8)
        
        # Mode: major for positive valence, minor for negative
        mode = "major" if valence > 0 else "minor"
        
        # Dynamics: based on intensity and arousal
        # Range: 0.2 (ppp) to 1.0 (fff)
        dynamics = 0.2 + (intensity * 0.6) + (arousal * 0.2)
        dynamics = min(dynamics, 1.0)
        
        # Harmonic complexity: higher for intense, mixed emotions
        # Range: 0.3 (simple) to 1.0 (very complex)
        harmonic_complexity = 0.3 + (intensity * 0.4) + (abs(valence) * 0.3)
        harmonic_complexity = min(harmonic_complexity, 1.0)
        
        # Rhythmic density: higher arousal = more rhythmic activity
        # Range: 0.2 (sparse) to 1.0 (dense)
        rhythmic_density = 0.2 + (arousal * 0.6) + (intensity * 0.2)
        rhythmic_density = min(rhythmic_density, 1.0)
        
        # Melodic contour: positive valence tends upward
        # Range: -1.0 (descending) to 1.0 (ascending)
        melodic_contour = (valence * 0.6) + (arousal * 0.4)
        melodic_contour = max(-1.0, min(1.0, melodic_contour))
        
        # Scale selection based on mode and emotion characteristics
        if mode == "major":
            if arousal > 0.7:
                preferred_scale = "lydian"  # Bright, uplifting
            elif arousal < 0.3:
                preferred_scale = "mixolydian"  # Relaxed major
            else:
                preferred_scale = "major"
        else:  # minor
            if arousal > 0.7:
                preferred_scale = "harmonic_minor"  # Tense, dramatic
            elif arousal < 0.3:
                preferred_scale = "dorian"  # Melancholic but not too dark
            else:
                preferred_scale = "natural_minor"
        
        # Suggested octave: higher for positive emotions
        suggested_octave = 4 + int((valence + 1.0) * 0.5)
        suggested_octave = max(3, min(6, suggested_octave))
        
        return {
            "tempo_modifier": round(tempo_modifier, 2),
            "mode": mode,
            "dynamics": round(dynamics, 2),
            "harmonic_complexity": round(harmonic_complexity, 2),
            "rhythmic_density": round(rhythmic_density, 2),
            "melodic_contour": round(melodic_contour, 2),
            "preferred_scale": preferred_scale,
            "suggested_octave": suggested_octave,
        }
    
    def _setup_relationships(self) -> None:
        """Set up relationships between related emotions."""
        for node in self.nodes.values():
            # Find emotions in the same category
            for other_node in self.nodes.values():
                if (other_node.id != node.id and 
                    other_node.category == node.category):
                    node.related_emotions.append(other_node.id)
            
            # Limit related emotions to prevent excessive connections
            if len(node.related_emotions) > 6:
                node.related_emotions = node.related_emotions[:6]
    
    def get_emotion(self, emotion_id: int) -> Optional[EmotionNode]:
        """Get emotion node by ID.
        
        Args:
            emotion_id: The ID of the emotion to retrieve
            
        Returns:
            EmotionNode if found, None otherwise
        """
        return self.nodes.get(emotion_id)
    
    def find_emotion_by_name(self, name: str) -> Optional[EmotionNode]:
        """Find emotion by name (case-insensitive).
        
        Args:
            name: The name of the emotion to find
            
        Returns:
            EmotionNode if found, None otherwise
        """
        emotion_id = self._name_to_id.get(name.lower())
        if emotion_id is not None:
            return self.nodes.get(emotion_id)
        return None
    
    def get_emotions_by_category(
        self, category: EmotionCategory
    ) -> List[EmotionNode]:
        """Get all emotions in a specific category.
        
        Args:
            category: The emotion category to filter by
            
        Returns:
            List of EmotionNodes in the specified category
        """
        return [
            node for node in self.nodes.values()
            if node.category == category
        ]
    
    def calculate_distance(
        self, emotion1_id: int, emotion2_id: int
    ) -> Optional[float]:
        """Calculate emotional distance between two emotions.
        
        Uses normalized Euclidean distance in VAD space.
        
        Args:
            emotion1_id: ID of the first emotion
            emotion2_id: ID of the second emotion
            
        Returns:
            Distance value (0.0 to ~2.45), or None if either emotion not found
        """
        emotion1 = self.get_emotion(emotion1_id)
        emotion2 = self.get_emotion(emotion2_id)
        
        if not emotion1 or not emotion2:
            return None
        
        # Normalize dimensions to [0, 1] for fair comparison
        # Valence: -1 to 1 -> 0 to 1
        v1 = (emotion1.valence + 1.0) / 2.0
        v2 = (emotion2.valence + 1.0) / 2.0
        
        # Arousal and intensity are already 0-1
        distance = math.sqrt(
            (v1 - v2) ** 2 +
            (emotion1.arousal - emotion2.arousal) ** 2 +
            (emotion1.intensity - emotion2.intensity) ** 2
        )
        
        return distance
    
    def get_nearby_emotions(
        self, emotion_id: int, threshold: float = 0.5, max_results: int = 10
    ) -> List[Tuple[EmotionNode, float]]:
        """
        Find emotions near the given emotion in emotional space.
        
        Results are sorted by distance (closest first).
        
        Args:
            emotion_id: ID of the source emotion
            threshold: Maximum distance in emotional space
            max_results: Maximum number of results to return
            
        Returns:
            List of (EmotionNode, distance) tuples, sorted by distance
        """
        source = self.get_emotion(emotion_id)
        if not source:
            return []
        
        nearby: List[Tuple[EmotionNode, float]] = []
        for node in self.nodes.values():
            if node.id == emotion_id:
                continue
            
            distance = self.calculate_distance(emotion_id, node.id)
            if distance is not None and distance < threshold:
                nearby.append((node, distance))
        
        # Sort by distance and limit results
        nearby.sort(key=lambda x: x[1])
        return nearby[:max_results]
    
    def interpolate_emotions(
        self, emotion1_id: int, emotion2_id: int, t: float
    ) -> Optional[Dict[str, Any]]:
        """Interpolate between two emotions.
        
        Args:
            emotion1_id: ID of the first emotion
            emotion2_id: ID of the second emotion
            t: Interpolation factor (0.0 = emotion1, 1.0 = emotion2)
            
        Returns:
            Dictionary with interpolated VAD values and musical attributes,
            or None if either emotion not found
        """
        emotion1 = self.get_emotion(emotion1_id)
        emotion2 = self.get_emotion(emotion2_id)
        
        if not emotion1 or not emotion2:
            return None
        
        t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
        inv_t = 1.0 - t
        
        # Interpolate VAD dimensions
        valence = emotion1.valence * inv_t + emotion2.valence * t
        arousal = emotion1.arousal * inv_t + emotion2.arousal * t
        intensity = emotion1.intensity * inv_t + emotion2.intensity * t
        
        # Determine category (use emotion1's category if t < 0.5, else emotion2's)
        category = emotion1.category if t < 0.5 else emotion2.category
        
        # Calculate musical attributes for interpolated emotion
        musical_attrs = self._calculate_musical_attributes(
            valence, arousal, intensity, category
        )
        
        return {
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "intensity": round(intensity, 3),
            "category": category.value,
            "musical_attributes": musical_attrs,
        }


def main() -> None:
    """Demo/test the emotion thesaurus."""
    print("=" * 70)
    print("Emotion Thesaurus Demo")
    print("=" * 70)
    
    thesaurus = EmotionThesaurus()
    
    print(f"\n✓ Loaded {len(thesaurus.nodes)} emotion nodes")
    print(f"\nAvailable emotions by category:")
    
    # Group by category
    for category in EmotionCategory:
        emotions = thesaurus.get_emotions_by_category(category)
        if emotions:
            print(f"\n  {category.value.upper()}:")
            for node in emotions:
                print(f"    [{node.id:2d}] {node.name:15s} "
                      f"V:{node.valence:5.2f} A:{node.arousal:.2f} I:{node.intensity:.2f}")
    
    # Demo: Find emotion by name
    print("\n" + "=" * 70)
    print("Demo 1: Finding 'grief' emotion")
    print("=" * 70)
    grief = thesaurus.find_emotion_by_name("grief")
    if grief:
        print(f"Found: {grief.name}")
        print(f"  Category: {grief.category.value}")
        print(f"  Valence: {grief.valence:.2f} (negative)")
        print(f"  Arousal: {grief.arousal:.2f} (moderate-high)")
        print(f"  Intensity: {grief.intensity:.2f} (very high)")
        print(f"\n  Musical Attributes:")
        for key, value in grief.musical_attributes.items():
            print(f"    {key}: {value}")
    
    # Demo: Find nearby emotions
    if grief:
        print("\n" + "=" * 70)
        print(f"Demo 2: Finding emotions near '{grief.name}'")
        print("=" * 70)
        nearby = thesaurus.get_nearby_emotions(grief.id, threshold=0.8, max_results=5)
        print(f"Found {len(nearby)} nearby emotions:")
        for node, distance in nearby:
            print(f"  - {node.name:15s} (distance: {distance:.3f}, "
                  f"category: {node.category.value})")
    
    # Demo: Interpolation
    if grief:
        print("\n" + "=" * 70)
        print("Demo 3: Interpolating between 'grief' and 'euphoria'")
        print("=" * 70)
        euphoria = thesaurus.find_emotion_by_name("euphoria")
        if euphoria:
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = thesaurus.interpolate_emotions(grief.id, euphoria.id, t)
                if result:
                    print(f"\n  t={t:.2f}:")
                    print(f"    VAD: V={result['valence']:.2f}, "
                          f"A={result['arousal']:.2f}, I={result['intensity']:.2f}")
                    print(f"    Mode: {result['musical_attributes']['mode']}, "
                          f"Tempo: {result['musical_attributes']['tempo_modifier']}x")
    
    # Demo: Distance calculation
    if grief and euphoria:
        print("\n" + "=" * 70)
        print("Demo 4: Distance calculation")
        print("=" * 70)
        distance = thesaurus.calculate_distance(grief.id, euphoria.id)
        if distance is not None:
            print(f"Distance between 'grief' and 'euphoria': {distance:.3f}")
            print("(Larger values indicate more different emotions)")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
