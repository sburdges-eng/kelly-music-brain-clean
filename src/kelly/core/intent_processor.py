"""Three-phase intent processing: Wound → Emotion → Rule-breaks."""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from kelly.core.emotion_thesaurus import EmotionThesaurus, EmotionNode


class IntentPhase(Enum):
    """Three phases of intent processing."""
    WOUND = "wound"  # Initial trauma or trigger
    EMOTION = "emotion"  # Emotional response
    RULE_BREAK = "rule_break"  # Musical rule violations


@dataclass
class Wound:
    """Represents the initial wound or trigger."""
    description: str
    intensity: float  # 0.0 to 1.0
    source: str  # Internal or external
    timestamp: Optional[float] = None


@dataclass
class RuleBreak:
    """Represents intentional musical rule violations."""
    rule_type: str  # e.g., "harmony", "rhythm", "dynamics"
    severity: float  # 0.0 to 1.0
    description: str
    musical_impact: Dict[str, any]


class IntentProcessor:
    """
    Processes therapeutic intent through three phases:
    1. Wound identification
    2. Emotional mapping
    3. Musical rule-breaking for expression
    """
    
    def __init__(self) -> None:
        """Initialize the intent processor."""
        self.thesaurus = EmotionThesaurus()
        self.wound_history: List[Wound] = []
        self.rule_breaks: List[RuleBreak] = []
    
    def process_wound(self, wound: Wound) -> EmotionNode:
        """
        Phase 1: Process a wound and map it to an emotion.
        
        Args:
            wound: The wound to process
            
        Returns:
            The mapped emotion node
        """
        self.wound_history.append(wound)
        
        # Simple mapping based on wound characteristics
        # In full implementation, this would use ML/pattern matching
        if "loss" in wound.description.lower() or "grief" in wound.description.lower():
            emotion = self.thesaurus.find_emotion_by_name("grief")
        elif "anger" in wound.description.lower() or "rage" in wound.description.lower():
            emotion = self.thesaurus.find_emotion_by_name("rage")
        elif "fear" in wound.description.lower() or "anxiety" in wound.description.lower():
            emotion = self.thesaurus.find_emotion_by_name("anxiety")
        else:
            # Default to melancholy for unspecified wounds
            emotion = self.thesaurus.find_emotion_by_name("melancholy")
        
        return emotion if emotion else self.thesaurus.nodes[0]
    
    def emotion_to_rule_breaks(self, emotion: EmotionNode) -> List[RuleBreak]:
        """
        Phase 2-3: Convert emotion to musical rule-breaks.
        
        Args:
            emotion: The emotion to express
            
        Returns:
            List of rule breaks for musical expression
        """
        rule_breaks = []
        
        # High intensity emotions break more rules
        if emotion.intensity > 0.8:
            rule_breaks.append(RuleBreak(
                rule_type="dynamics",
                severity=emotion.intensity,
                description="Extreme dynamic contrasts",
                musical_impact={
                    "velocity_range": (10, 127),
                    "sudden_changes": True
                }
            ))
        
        # Negative valence introduces dissonance
        if emotion.valence < -0.5:
            rule_breaks.append(RuleBreak(
                rule_type="harmony",
                severity=abs(emotion.valence),
                description="Dissonant intervals and clusters",
                musical_impact={
                    "allow_dissonance": True,
                    "cluster_probability": abs(emotion.valence)
                }
            ))
        
        # High arousal breaks rhythmic conventions
        if emotion.arousal > 0.7:
            rule_breaks.append(RuleBreak(
                rule_type="rhythm",
                severity=emotion.arousal,
                description="Irregular rhythms and syncopation",
                musical_impact={
                    "syncopation_level": emotion.arousal,
                    "irregular_meters": True
                }
            ))
        
        self.rule_breaks.extend(rule_breaks)
        return rule_breaks
    
    def process_intent(self, wound: Wound) -> Dict[str, any]:
        """
        Complete three-phase intent processing.
        
        Args:
            wound: The initial wound/trigger
            
        Returns:
            Complete intent processing result
        """
        # Phase 1: Wound → Emotion
        emotion = self.process_wound(wound)
        
        # Phase 2-3: Emotion → Rule-breaks
        rule_breaks = self.emotion_to_rule_breaks(emotion)
        
        return {
            "wound": wound,
            "emotion": emotion,
            "rule_breaks": rule_breaks,
            "musical_params": self._compile_musical_params(emotion, rule_breaks)
        }
    
    def _compile_musical_params(
        self, emotion: EmotionNode, rule_breaks: List[RuleBreak]
    ) -> Dict[str, any]:
        """Compile final musical parameters from emotion and rule breaks."""
        params = emotion.musical_attributes.copy()
        
        for rb in rule_breaks:
            params.update(rb.musical_impact)
        
        return params
