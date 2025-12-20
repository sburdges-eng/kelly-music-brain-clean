"""Intent Processor - Three-phase emotional interrogation.

Phase 0: Core Wound/Desire (deep interrogation)
Phase 1: Emotional & Intent (validation)
Phase 2: Technical Constraints (implementation)
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

from kelly.core.emotion_thesaurus import EmotionThesaurus, EmotionNode, EmotionCategory


class WoundType(Enum):
    """Categories of emotional wounds."""
    LOSS = "loss"
    BETRAYAL = "betrayal"
    REJECTION = "rejection"
    FAILURE = "failure"
    TRAUMA = "trauma"
    LONELINESS = "loneliness"
    SHAME = "shame"
    FEAR = "fear"
    ANGER = "anger"
    LONGING = "longing"
    REGRET = "regret"
    UNSPECIFIED = "unspecified"


class RuleBreakType(Enum):
    """Types of intentional rule violations."""
    HARMONY = "harmony"
    RHYTHM = "rhythm"
    ARRANGEMENT = "arrangement"
    PRODUCTION = "production"
    MELODY = "melody"
    TEXTURE = "texture"


@dataclass
class Wound:
    """Represents the core emotional trigger."""
    description: str
    intensity: float = 0.7
    source: str = "user_input"
    wound_type: WoundType = WoundType.UNSPECIFIED
    context: Optional[str] = None
    
    def __post_init__(self):
        self.intensity = max(0.0, min(1.0, self.intensity))
        if self.wound_type == WoundType.UNSPECIFIED:
            self.wound_type = self._infer_wound_type()
    
    def _infer_wound_type(self) -> WoundType:
        """Infer wound type from description."""
        desc_lower = self.description.lower()
        keywords = {
            WoundType.LOSS: ["loss", "lost", "gone", "death", "died", "miss"],
            WoundType.BETRAYAL: ["betrayal", "betrayed", "cheated", "lied"],
            WoundType.REJECTION: ["reject", "rejected", "unwanted", "alone"],
            WoundType.FAILURE: ["failed", "failure", "couldn't", "wasn't enough"],
            WoundType.TRAUMA: ["trauma", "abuse", "hurt", "pain", "suffering"],
            WoundType.LONELINESS: ["lonely", "alone", "isolated", "empty"],
            WoundType.SHAME: ["shame", "ashamed", "embarrass", "humiliat"],
            WoundType.FEAR: ["afraid", "scared", "fear", "anxious", "worry"],
            WoundType.ANGER: ["angry", "rage", "furious", "hate"],
            WoundType.LONGING: ["miss", "want", "need", "yearn", "wish"],
            WoundType.REGRET: ["regret", "wish", "should have", "if only"],
        }
        for wound_type, words in keywords.items():
            if any(w in desc_lower for w in words):
                return wound_type
        return WoundType.UNSPECIFIED


@dataclass
class RuleBreak:
    """An intentional violation of musical convention."""
    rule_type: str
    description: str
    severity: float
    justification: str
    implementation_hint: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "rule_type": self.rule_type,
            "description": self.description,
            "severity": self.severity,
            "justification": self.justification,
            "implementation_hint": self.implementation_hint,
        }


# Rule break suggestions based on wound type
WOUND_RULE_BREAKS: Dict[WoundType, List[Dict]] = {
    WoundType.LOSS: [
        {"rule": "HARMONY_UnresolvedDissonance", "desc": "Never fully resolve", "severity": 0.7},
        {"rule": "RHYTHM_DroppedBeats", "desc": "Missing beats like missing pieces", "severity": 0.5},
        {"rule": "PRODUCTION_SilenceAsInstrument", "desc": "Emptiness speaks", "severity": 0.6},
    ],
    WoundType.BETRAYAL: [
        {"rule": "HARMONY_ModalInterchange", "desc": "Major to minor shifts", "severity": 0.6},
        {"rule": "MELODY_AvoidResolution", "desc": "Trust never restored", "severity": 0.7},
        {"rule": "PRODUCTION_Distortion", "desc": "Beauty corrupted", "severity": 0.5},
    ],
    WoundType.REJECTION: [
        {"rule": "HARMONY_AvoidTonicResolution", "desc": "Never arriving home", "severity": 0.6},
        {"rule": "ARRANGEMENT_BuriedVocals", "desc": "Voice not heard", "severity": 0.5},
        {"rule": "TEXTURE_Sparse", "desc": "Emptiness of isolation", "severity": 0.4},
    ],
    WoundType.FAILURE: [
        {"rule": "MELODY_AntiClimax", "desc": "Building to nothing", "severity": 0.7},
        {"rule": "RHYTHM_TempoFluctuation", "desc": "Losing momentum", "severity": 0.5},
        {"rule": "ARRANGEMENT_PrematureClimax", "desc": "Peaking too early", "severity": 0.6},
    ],
    WoundType.TRAUMA: [
        {"rule": "RHYTHM_MetricModulation", "desc": "Time distortion", "severity": 0.8},
        {"rule": "PRODUCTION_RoomNoise", "desc": "Environmental intrusion", "severity": 0.5},
        {"rule": "HARMONY_Polytonality", "desc": "Fractured reality", "severity": 0.7},
    ],
    WoundType.LONELINESS: [
        {"rule": "TEXTURE_Sparse", "desc": "Vast emptiness", "severity": 0.6},
        {"rule": "PRODUCTION_ExcessiveReverb", "desc": "Echo chamber of isolation", "severity": 0.5},
        {"rule": "MELODY_MonotoneDrone", "desc": "Unchanging solitude", "severity": 0.4},
    ],
    WoundType.SHAME: [
        {"rule": "ARRANGEMENT_BuriedVocals", "desc": "Hiding the self", "severity": 0.7},
        {"rule": "PRODUCTION_LoFiDegradation", "desc": "Self-degradation", "severity": 0.6},
        {"rule": "MELODY_FragmentedPhrases", "desc": "Unable to complete", "severity": 0.5},
    ],
    WoundType.FEAR: [
        {"rule": "RHYTHM_TempoFluctuation", "desc": "Heartbeat racing", "severity": 0.6},
        {"rule": "HARMONY_AvoidTonicResolution", "desc": "No safe ground", "severity": 0.7},
        {"rule": "PRODUCTION_RoomNoise", "desc": "Threat in environment", "severity": 0.4},
    ],
    WoundType.ANGER: [
        {"rule": "PRODUCTION_Distortion", "desc": "Raw aggression", "severity": 0.8},
        {"rule": "HARMONY_ParallelMotion", "desc": "Relentless force", "severity": 0.6},
        {"rule": "RHYTHM_PolyrhythmicLayers", "desc": "Chaotic energy", "severity": 0.7},
    ],
    WoundType.LONGING: [
        {"rule": "MELODY_AvoidResolution", "desc": "Always reaching", "severity": 0.6},
        {"rule": "HARMONY_ModalInterchange", "desc": "Between hope and loss", "severity": 0.5},
        {"rule": "PRODUCTION_ExcessiveReverb", "desc": "Distance and memory", "severity": 0.4},
    ],
    WoundType.REGRET: [
        {"rule": "MELODY_ExcessiveRepetition", "desc": "Obsessive replay", "severity": 0.6},
        {"rule": "HARMONY_UnresolvedDissonance", "desc": "Can't undo", "severity": 0.5},
        {"rule": "RHYTHM_DroppedBeats", "desc": "Moments missed", "severity": 0.5},
    ],
    WoundType.UNSPECIFIED: [
        {"rule": "HARMONY_ModalInterchange", "desc": "Emotional complexity", "severity": 0.5},
        {"rule": "RHYTHM_ConstantDisplacement", "desc": "Off-center feeling", "severity": 0.4},
    ],
}


@dataclass
class IntentResult:
    """Complete result from intent processing."""
    wound: Wound
    emotion: EmotionNode
    rule_breaks: List[RuleBreak]
    musical_params: Dict[str, Any]
    narrative_arc: str
    imagery: List[str]
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "wound": {
                "description": self.wound.description,
                "intensity": self.wound.intensity,
                "type": self.wound.wound_type.value,
            },
            "emotion": {
                "name": self.emotion.name,
                "category": self.emotion.category.value,
                "valence": self.emotion.valence,
                "arousal": self.emotion.arousal,
            },
            "rule_breaks": [rb.to_dict() for rb in self.rule_breaks],
            "musical_params": self.musical_params,
            "narrative_arc": self.narrative_arc,
            "imagery": self.imagery,
            "confidence": self.confidence,
        }


class IntentProcessor:
    """Processes emotional wounds into musical intent.
    
    Three-phase processing:
    1. Wound identification and categorization
    2. Emotion mapping via thesaurus
    3. Rule-break generation for authentic expression
    
    Usage:
        processor = IntentProcessor()
        wound = Wound("the loss of my best friend", intensity=0.9)
        result = processor.process_intent(wound)
    """
    
    def __init__(self, thesaurus: Optional[EmotionThesaurus] = None):
        self.thesaurus = thesaurus or EmotionThesaurus()
    
    def process_intent(self, wound: Wound) -> IntentResult:
        """Process a wound through all three phases."""
        # Phase 1: Map to emotion
        emotion = self._map_to_emotion(wound)
        
        # Phase 2: Generate rule breaks
        rule_breaks = self._generate_rule_breaks(wound, emotion)
        
        # Phase 3: Compile musical parameters
        musical_params = self._compile_musical_params(wound, emotion, rule_breaks)
        
        # Generate narrative elements
        narrative_arc = self._generate_narrative_arc(wound, emotion)
        imagery = self._generate_imagery(wound, emotion)
        
        # Calculate confidence
        confidence = self._calculate_confidence(wound, emotion)
        
        return IntentResult(
            wound=wound,
            emotion=emotion,
            rule_breaks=rule_breaks,
            musical_params=musical_params,
            narrative_arc=narrative_arc,
            imagery=imagery,
            confidence=confidence,
        )
    
    def _map_to_emotion(self, wound: Wound) -> EmotionNode:
        """Map wound description to emotion node."""
        desc_lower = wound.description.lower()
        
        # Direct emotion name match
        for emotion_name in self.thesaurus.list_all():
            if emotion_name in desc_lower:
                node = self.thesaurus.get_emotion(emotion_name)
                if node:
                    return node
        
        # Wound type to emotion mapping
        wound_emotion_map = {
            WoundType.LOSS: "grief",
            WoundType.BETRAYAL: "anger",
            WoundType.REJECTION: "sadness",
            WoundType.FAILURE: "despair",
            WoundType.TRAUMA: "fear",
            WoundType.LONELINESS: "melancholy",
            WoundType.SHAME: "anguish",
            WoundType.FEAR: "anxiety",
            WoundType.ANGER: "rage" if wound.intensity > 0.7 else "anger",
            WoundType.LONGING: "wistfulness",
            WoundType.REGRET: "sorrow",
        }
        
        emotion_name = wound_emotion_map.get(wound.wound_type, "sadness")
        
        # Adjust for intensity
        category_emotions = {
            "grief": EmotionCategory.SADNESS,
            "anger": EmotionCategory.ANGER,
            "anxiety": EmotionCategory.FEAR,
        }
        
        if emotion_name in category_emotions:
            # Find emotion at appropriate intensity level
            cat = category_emotions[emotion_name]
            emotions = self.thesaurus.find_by_category(cat)
            intensity_matches = sorted(
                emotions, 
                key=lambda e: abs(e.intensity - wound.intensity)
            )
            if intensity_matches:
                return intensity_matches[0]
        
        return self.thesaurus.get_emotion(emotion_name) or self.thesaurus.get_emotion("sadness")
    
    def _generate_rule_breaks(
        self, 
        wound: Wound, 
        emotion: EmotionNode
    ) -> List[RuleBreak]:
        """Generate appropriate rule breaks for the wound/emotion."""
        rule_breaks = []
        
        # Get wound-specific rule breaks
        wound_rules = WOUND_RULE_BREAKS.get(wound.wound_type, [])
        
        for rule_data in wound_rules:
            severity = rule_data["severity"] * wound.intensity
            rule_break = RuleBreak(
                rule_type=rule_data["rule"],
                description=rule_data["desc"],
                severity=severity,
                justification=f"Expressing {wound.wound_type.value} through {rule_data['desc'].lower()}",
            )
            rule_breaks.append(rule_break)
        
        # Add emotion-based rule breaks
        for rule_str in emotion.musical_mapping.rule_breaks:
            if not any(rb.rule_type == rule_str for rb in rule_breaks):
                rule_breaks.append(RuleBreak(
                    rule_type=rule_str,
                    description=f"{emotion.name} expression",
                    severity=emotion.intensity * 0.6,
                    justification=f"Emotional authenticity for {emotion.name}",
                ))
        
        return rule_breaks
    
    def _compile_musical_params(
        self,
        wound: Wound,
        emotion: EmotionNode,
        rule_breaks: List[RuleBreak]
    ) -> Dict[str, Any]:
        """Compile all musical parameters from emotion mapping."""
        mapping = emotion.musical_mapping
        
        return {
            "mode": mapping.mode,
            "tempo_modifier": mapping.tempo_modifier,
            "velocity_range": mapping.dynamic_range,
            "harmonic_complexity": mapping.harmonic_complexity,
            "dissonance": mapping.dissonance_tolerance,
            "rhythm_regularity": mapping.rhythm_regularity,
            "articulation": mapping.articulation,
            "register": mapping.register_preference,
            "space_density": mapping.space_density,
            "allow_dissonance": mapping.dissonance_tolerance > 0.3,
            "rule_breaks": [rb.rule_type for rb in rule_breaks],
            "intensity": wound.intensity,
            "valence": emotion.valence,
            "arousal": emotion.arousal,
        }
    
    def _generate_narrative_arc(self, wound: Wound, emotion: EmotionNode) -> str:
        """Generate narrative arc suggestion."""
        arcs = {
            WoundType.LOSS: "descent_with_acceptance",
            WoundType.BETRAYAL: "confrontation_to_release",
            WoundType.REJECTION: "isolation_to_self_acceptance",
            WoundType.FAILURE: "defeat_to_resilience",
            WoundType.TRAUMA: "fragmentation_to_integration",
            WoundType.LONELINESS: "emptiness_to_connection",
            WoundType.SHAME: "hiding_to_emergence",
            WoundType.FEAR: "threat_to_safety",
            WoundType.ANGER: "eruption_to_exhaustion",
            WoundType.LONGING: "reaching_without_grasping",
            WoundType.REGRET: "replay_to_release",
        }
        return arcs.get(wound.wound_type, "emotional_journey")
    
    def _generate_imagery(self, wound: Wound, emotion: EmotionNode) -> List[str]:
        """Generate evocative imagery for the emotional state."""
        imagery_bank = {
            EmotionCategory.SADNESS: [
                "rain on empty streets", "fading photographs", 
                "wilting flowers", "distant horizon", "grey light through curtains"
            ],
            EmotionCategory.ANGER: [
                "breaking glass", "thunderstorm", "clenched fists",
                "fire spreading", "shattering mirrors"
            ],
            EmotionCategory.FEAR: [
                "shadows lengthening", "footsteps in darkness",
                "door slowly opening", "cold breath on neck", "falling sensation"
            ],
            EmotionCategory.JOY: [
                "sunlight through leaves", "laughter echoing",
                "warm embrace", "dancing in rain", "first light of dawn"
            ],
            EmotionCategory.SURPRISE: [
                "sudden silence", "flash of light",
                "held breath", "world shifting", "time stopping"
            ],
            EmotionCategory.DISGUST: [
                "bitter taste", "recoiling touch",
                "corrupted beauty", "rot beneath surface", "broken trust"
            ],
        }
        return imagery_bank.get(emotion.category, ["undefined space"])[:3]
    
    def _calculate_confidence(self, wound: Wound, emotion: EmotionNode) -> float:
        """Calculate confidence in the emotion mapping."""
        # Higher confidence if wound type matches emotion category
        type_match = {
            WoundType.LOSS: EmotionCategory.SADNESS,
            WoundType.BETRAYAL: EmotionCategory.ANGER,
            WoundType.REJECTION: EmotionCategory.SADNESS,
            WoundType.FAILURE: EmotionCategory.SADNESS,
            WoundType.TRAUMA: EmotionCategory.FEAR,
            WoundType.LONELINESS: EmotionCategory.SADNESS,
            WoundType.SHAME: EmotionCategory.DISGUST,
            WoundType.FEAR: EmotionCategory.FEAR,
            WoundType.ANGER: EmotionCategory.ANGER,
        }
        
        expected_cat = type_match.get(wound.wound_type)
        if expected_cat == emotion.category:
            return 0.85
        elif expected_cat is None:
            return 0.6
        else:
            return 0.5


def process_wound(description: str, intensity: float = 0.7) -> Dict:
    """Quick helper to process a wound description."""
    processor = IntentProcessor()
    wound = Wound(description=description, intensity=intensity)
    result = processor.process_intent(wound)
    return result.to_dict()
