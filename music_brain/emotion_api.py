"""
Music Brain Emotion API - Clean interface for emotion-to-music generation.

Provides both declarative and fluent API styles for mapping emotional intent
to musical parameters and DAW mixer automation.

Philosophy: "Interrogate Before Generate" - Emotional intent drives technical choices.

Usage (Declarative):
    brain = MusicBrain()
    music = brain.generate_from_intent(intent)
    brain.export_to_logic(music, "output.mid")

Usage (Fluent):
    brain = MusicBrain()
    result = (brain.process("grief and loss")
                   .map_to_mixer()
                   .export_logic("output_automation.json"))
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from music_brain.data.emotional_mapping import (
    Arousal,
    EmotionalState,
    MusicalParameters,
    TimingFeel,
    Valence,
    get_parameters_for_state,
)
from music_brain.daw.mixer_params import (
    EmotionMapper,
    MixerParameters,
    export_mixer_settings,
    export_to_logic_automation,
)

# Import existing modules
from music_brain.session.intent_schema import (
    CompleteSongIntent,
    SongIntent,
    SongRoot,
    SystemDirective,
    TechnicalConstraints,
    get_affect_mapping,
    suggest_rule_break,
)

# Intent examples for reference
INTENT_EXAMPLES = {
    "therapeutic": "Create calming music for anxiety",
    "workout": "Energetic, driving beat for running",
    "grief_processing": "Melancholic, introspective piece for loss",
    "anger_release": "Cathartic, heavy track for rage",
    "misdirection": "Surface happy, undertow sad (emotional contrast)",
    "nostalgia": "Warm, lo-fi piece for memory processing",
    "hope": "Uplifting, building toward resolution",
    "tension": "Building suspense and unease",
}


@dataclass
class GeneratedMusic:
    """Complete music generation result."""
    emotional_state: EmotionalState
    musical_params: MusicalParameters
    mixer_params: MixerParameters
    intent: CompleteSongIntent | None = None
    midi_path: str | None = None
    automation_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "emotional_state": {
                "valence": self.emotional_state.valence.value if isinstance(self.emotional_state.valence, Valence) else self.emotional_state.valence,
                "arousal": self.emotional_state.arousal.value if isinstance(self.emotional_state.arousal, Arousal) else self.emotional_state.arousal,
                "primary_emotion": self.emotional_state.primary_emotion,
                "secondary_emotions": self.emotional_state.secondary_emotions,
            },
            "musical_params": {
                "tempo_suggested": self.musical_params.tempo_suggested,
                "tempo_range": (self.musical_params.tempo_min, self.musical_params.tempo_max),
                "dissonance": self.musical_params.dissonance,
                "timing_feel": self.musical_params.timing_feel.value,
                "density_suggested": self.musical_params.density_suggested,
                "space_probability": self.musical_params.space_probability,
            },
            "mixer_params": self.mixer_params.to_dict(),
            "paths": {
                "midi": self.midi_path,
                "automation": self.automation_path,
            },
        }

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            "=" * 60,
            "GENERATED MUSIC SUMMARY",
            "=" * 60,
            "",
            f"Primary Emotion: {self.emotional_state.primary_emotion}",
            f"Valence: {self.emotional_state.valence}",
            f"Arousal: {self.emotional_state.arousal}",
            "",
            "Musical Parameters:",
            f"  Tempo: {self.musical_params.tempo_suggested} BPM",
            f"  Timing Feel: {self.musical_params.timing_feel.value}",
            f"  Dissonance: {self.musical_params.dissonance:.0%}",
            f"  Density: {self.musical_params.density_suggested}",
            "",
            "Mixer Settings:",
            f"  Description: {self.mixer_params.description}",
            f"  Reverb: {self.mixer_params.reverb_mix:.0%} mix, {self.mixer_params.reverb_decay:.1f}s decay",
            f"  Compression: {self.mixer_params.compression_ratio:.1f}:1",
            f"  Saturation: {self.mixer_params.saturation:.0%} ({self.mixer_params.saturation_type})",
            "",
            "Output Files:",
            f"  MIDI: {self.midi_path or 'Not generated'}",
            f"  Automation: {self.automation_path or 'Not generated'}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


class MusicBrain:
    """
    Main Music Brain API for emotion-to-music generation.

    Provides a clean, consistent interface for all emotion-to-music operations,
    making it easier to integrate with desktop apps, web services, or CLI tools.
    """

    def __init__(self):
        self.emotion_mapper = EmotionMapper()
        self._emotion_keywords = self._build_emotion_keywords()

    def _build_emotion_keywords(self) -> dict[str, tuple[float, float, str]]:
        """Build emotion keyword to (valence, arousal, emotion) mapping."""
        return {
            # Negative valence, low arousal
            "grief": (-0.8, 0.3, "grief"),
            "sad": (-0.6, 0.3, "grief"),
            "loss": (-0.7, 0.3, "grief"),
            "mourning": (-0.8, 0.2, "grief"),
            "melancholy": (-0.5, 0.2, "grief"),
            "sorrow": (-0.7, 0.3, "grief"),

            # Negative valence, high arousal
            "anxiety": (-0.6, 0.8, "anxiety"),
            "anxious": (-0.6, 0.8, "anxiety"),
            "nervous": (-0.5, 0.7, "anxiety"),
            "panic": (-0.7, 0.9, "anxiety"),
            "fear": (-0.6, 0.8, "anxiety"),
            "worry": (-0.5, 0.6, "anxiety"),

            # Negative valence, high arousal (anger)
            "anger": (-0.7, 0.9, "anger"),
            "angry": (-0.7, 0.9, "anger"),
            "rage": (-0.9, 0.95, "anger"),
            "fury": (-0.8, 0.95, "anger"),
            "frustration": (-0.5, 0.7, "anger"),

            # Positive valence, low arousal
            "calm": (0.3, 0.2, "calm"),
            "peaceful": (0.4, 0.2, "calm"),
            "serene": (0.4, 0.15, "calm"),
            "relaxed": (0.3, 0.25, "calm"),
            "content": (0.4, 0.3, "calm"),

            # Positive valence, high arousal
            "hope": (0.6, 0.6, "hope"),
            "hopeful": (0.6, 0.6, "hope"),
            "joy": (0.8, 0.7, "hope"),
            "happy": (0.7, 0.6, "hope"),
            "excited": (0.7, 0.8, "hope"),
            "euphoria": (0.9, 0.9, "hope"),

            # Mixed/complex
            "nostalgia": (-0.2, 0.3, "nostalgia"),
            "nostalgic": (-0.2, 0.3, "nostalgia"),
            "bittersweet": (-0.1, 0.4, "nostalgia"),
            "wistful": (-0.2, 0.3, "nostalgia"),

            "tension": (-0.4, 0.7, "tension"),
            "suspense": (-0.4, 0.7, "tension"),
            "building": (-0.3, 0.6, "tension"),
            "uneasy": (-0.4, 0.6, "tension"),

            "catharsis": (0.2, 0.8, "catharsis"),
            "release": (0.3, 0.7, "catharsis"),
            "breakthrough": (0.4, 0.8, "catharsis"),

            "dissociation": (-0.3, 0.2, "dissociation"),
            "disconnected": (-0.3, 0.2, "dissociation"),
            "numb": (-0.4, 0.1, "dissociation"),
            "detached": (-0.3, 0.2, "dissociation"),

            "intimacy": (0.5, 0.3, "intimacy"),
            "intimate": (0.5, 0.3, "intimacy"),
            "vulnerable": (0.1, 0.4, "intimacy"),
            "tender": (0.5, 0.3, "intimacy"),
        }

    # ========== DECLARATIVE API ==========

    def generate_from_intent(
        self,
        intent: CompleteSongIntent
    ) -> GeneratedMusic:
        """
        Generate complete music from intent (simple, one-step).

        Args:
            intent: Complete song intent with all phases

        Returns:
            GeneratedMusic with all parameters
        """
        # Extract emotion from intent
        mood = intent.song_intent.mood_primary.lower() if intent.song_intent.mood_primary else "neutral"

        # Map mood to valence/arousal
        if mood in self._emotion_keywords:
            valence, arousal, emotion_key = self._emotion_keywords[mood]
        else:
            # Try to find partial match
            emotion_key = mood
            valence, arousal = -0.5, 0.4  # Default for unknown
            for keyword, (v, a, e) in self._emotion_keywords.items():
                if keyword in mood:
                    valence, arousal, emotion_key = v, a, e
                    break

        # Create emotional state
        emotional_state = EmotionalState(
            valence=Valence(int(valence * 2)),  # Convert to enum
            arousal=Arousal(int((arousal - 0.5) * 4)),  # Convert to enum
            primary_emotion=emotion_key
        )

        # Get musical parameters
        musical_params = get_parameters_for_state(emotional_state)

        # Override with technical constraints from intent
        if intent.technical_constraints.technical_tempo_range:
            tempo_min, tempo_max = intent.technical_constraints.technical_tempo_range
            musical_params.tempo_min = tempo_min
            musical_params.tempo_max = tempo_max
            musical_params.tempo_suggested = (tempo_min + tempo_max) // 2

        # Map to mixer parameters
        mixer_params = self.emotion_mapper.map_emotion_to_mixer(
            emotional_state,
            musical_params
        )

        return GeneratedMusic(
            emotional_state=emotional_state,
            musical_params=musical_params,
            mixer_params=mixer_params,
            intent=intent
        )

    def generate_from_text(
        self,
        emotional_text: str
    ) -> GeneratedMusic:
        """
        Generate music from emotional text description.

        Args:
            emotional_text: Natural language emotional description

        Returns:
            GeneratedMusic with all parameters
        """
        # Parse text for emotion keywords
        text_lower = emotional_text.lower()

        valence = 0.0
        arousal = 0.5
        primary_emotion = "neutral"

        # Find matching emotion
        for keyword, (v, a, emotion) in self._emotion_keywords.items():
            if keyword in text_lower:
                valence, arousal = v, a
                primary_emotion = emotion
                break

        # Create emotional state
        emotional_state = EmotionalState(
            valence=Valence(max(-2, min(2, int(valence * 2)))),
            arousal=Arousal(max(-2, min(2, int((arousal - 0.5) * 4)))),
            primary_emotion=primary_emotion
        )

        # Get musical parameters
        musical_params = get_parameters_for_state(emotional_state)

        # Map to mixer parameters
        mixer_params = self.emotion_mapper.map_emotion_to_mixer(
            emotional_state,
            musical_params
        )

        return GeneratedMusic(
            emotional_state=emotional_state,
            musical_params=musical_params,
            mixer_params=mixer_params
        )

    def export_to_logic(
        self,
        music: GeneratedMusic,
        output_base: str
    ) -> dict[str, str]:
        """
        Export to Logic Pro format.

        Args:
            music: Generated music
            output_base: Base filename (without extension)

        Returns:
            Dict with paths to created files
        """
        output_base = Path(output_base).stem

        # Export mixer automation
        automation_path = f"{output_base}_automation.json"
        export_to_logic_automation(music.mixer_params, automation_path)
        music.automation_path = automation_path

        result = {
            "automation": automation_path,
        }

        return result

    def create_intent(
        self,
        title: str,
        core_event: str,
        mood_primary: str,
        technical_key: str = "C",
        technical_mode: str = "major",
        tempo_range: tuple[int, int] = (80, 120),
        rule_to_break: str = "",
        rule_justification: str = "",
        **kwargs
    ) -> CompleteSongIntent:
        """
        Create a CompleteSongIntent from parameters.

        Args:
            title: Song title
            core_event: The inciting moment/realization
            mood_primary: Primary emotion
            technical_key: Musical key
            technical_mode: Mode (major, minor, etc.)
            tempo_range: (min, max) BPM
            rule_to_break: Optional rule to break
            rule_justification: Why break the rule
            **kwargs: Additional intent fields

        Returns:
            CompleteSongIntent
        """
        return CompleteSongIntent(
            title=title,
            song_root=SongRoot(
                core_event=core_event,
                core_resistance=kwargs.get("core_resistance", ""),
                core_longing=kwargs.get("core_longing", ""),
                core_stakes=kwargs.get("core_stakes", ""),
                core_transformation=kwargs.get("core_transformation", ""),
            ),
            song_intent=SongIntent(
                mood_primary=mood_primary,
                mood_secondary_tension=kwargs.get("mood_secondary_tension", 0.5),
                imagery_texture=kwargs.get("imagery_texture", ""),
                vulnerability_scale=kwargs.get("vulnerability_scale", "Medium"),
                narrative_arc=kwargs.get("narrative_arc", ""),
            ),
            technical_constraints=TechnicalConstraints(
                technical_genre=kwargs.get("technical_genre", ""),
                technical_tempo_range=tempo_range,
                technical_key=technical_key,
                technical_mode=technical_mode,
                technical_groove_feel=kwargs.get("technical_groove_feel", ""),
                technical_rule_to_break=rule_to_break,
                rule_breaking_justification=rule_justification,
            ),
            system_directive=SystemDirective(
                output_target=kwargs.get("output_target", ""),
                output_feedback_loop=kwargs.get("output_feedback_loop", ""),
            ),
        )

    def suggest_rules(self, emotion: str) -> list[dict]:
        """
        Get rule-breaking suggestions for an emotion.

        Args:
            emotion: Target emotion

        Returns:
            List of rule-breaking suggestions
        """
        return suggest_rule_break(emotion)

    def get_affect_mapping(self, emotion: str) -> dict | None:
        """
        Get musical parameter suggestions for an emotion.

        Args:
            emotion: Target emotion

        Returns:
            Dict with modes, tempo_range, density suggestions
        """
        return get_affect_mapping(emotion)

    def list_mixer_presets(self) -> list[str]:
        """List all available mixer presets."""
        return self.emotion_mapper.list_presets()

    def get_mixer_preset(self, emotion: str) -> MixerParameters | None:
        """Get a specific mixer preset."""
        return self.emotion_mapper.get_preset(emotion)

    # ========== FLUENT API ==========

    def process(self, emotional_text: str) -> 'FluentChain':
        """Start a fluent processing chain."""
        return FluentChain(emotional_text, self)


class FluentChain:
    """
    Fluent API for step-by-step control over emotion-to-music mapping.

    Usage:
        brain = MusicBrain()
        result = (brain.process("grief and loss")
                       .map_to_emotion()
                       .map_to_music()
                       .map_to_mixer()
                       .export_logic("output_automation.json"))
    """

    def __init__(self, emotional_text: str, brain: MusicBrain):
        self.brain = brain
        self.emotional_text = emotional_text
        self.emotional_state: EmotionalState | None = None
        self.musical_params: MusicalParameters | None = None
        self.mixer_params: MixerParameters | None = None
        self._paths: dict[str, str] = {}

    def map_to_emotion(self) -> 'FluentChain':
        """Map text to emotional state."""
        text_lower = self.emotional_text.lower()

        valence = 0.0
        arousal = 0.5
        primary_emotion = "neutral"

        for keyword, (v, a, emotion) in self.brain._emotion_keywords.items():
            if keyword in text_lower:
                valence, arousal = v, a
                primary_emotion = emotion
                break

        self.emotional_state = EmotionalState(
            valence=Valence(max(-2, min(2, int(valence * 2)))),
            arousal=Arousal(max(-2, min(2, int((arousal - 0.5) * 4)))),
            primary_emotion=primary_emotion
        )
        return self

    def map_to_music(self) -> 'FluentChain':
        """Map emotion to musical parameters."""
        if not self.emotional_state:
            self.map_to_emotion()

        self.musical_params = get_parameters_for_state(self.emotional_state)
        return self

    def map_to_mixer(self) -> 'FluentChain':
        """Map to mixer parameters."""
        if not self.musical_params:
            self.map_to_music()

        self.mixer_params = self.brain.emotion_mapper.map_emotion_to_mixer(
            self.emotional_state,
            self.musical_params
        )
        return self

    def with_tempo(self, tempo: int) -> 'FluentChain':
        """Override tempo."""
        if not self.musical_params:
            self.map_to_music()
        self.musical_params.tempo_suggested = tempo
        return self

    def with_dissonance(self, dissonance: float) -> 'FluentChain':
        """Override dissonance level (0.0-1.0)."""
        if not self.musical_params:
            self.map_to_music()
        self.musical_params.dissonance = max(0.0, min(1.0, dissonance))
        return self

    def with_timing(self, feel: str) -> 'FluentChain':
        """Override timing feel (ahead, on, behind)."""
        if not self.musical_params:
            self.map_to_music()
        feel_map = {
            "ahead": TimingFeel.AHEAD,
            "on": TimingFeel.ON,
            "behind": TimingFeel.BEHIND,
        }
        if feel.lower() in feel_map:
            self.musical_params.timing_feel = feel_map[feel.lower()]
        return self

    def export_logic(self, output_path: str) -> dict[str, str]:
        """Export to Logic Pro and return paths."""
        if not self.mixer_params:
            self.map_to_mixer()

        automation_path = export_to_logic_automation(
            self.mixer_params,
            output_path
        )
        self._paths["automation"] = automation_path

        return {
            "automation": automation_path,
            "emotional_state": str(self.emotional_state.primary_emotion) if self.emotional_state else "neutral",
            "tempo": str(self.musical_params.tempo_suggested) if self.musical_params else "unknown",
        }

    def export_json(self, output_path: str) -> str:
        """Export full settings to JSON."""
        if not self.mixer_params:
            self.map_to_mixer()

        export_mixer_settings(self.mixer_params, output_path, format="json")
        self._paths["json"] = output_path
        return output_path

    def get(self) -> dict[str, Any]:
        """Get current state."""
        return {
            "emotional_text": self.emotional_text,
            "emotional_state": {
                "primary_emotion": self.emotional_state.primary_emotion if self.emotional_state else None,
                "valence": self.emotional_state.valence if self.emotional_state else None,
                "arousal": self.emotional_state.arousal if self.emotional_state else None,
            } if self.emotional_state else None,
            "musical_params": {
                "tempo": self.musical_params.tempo_suggested if self.musical_params else None,
                "dissonance": self.musical_params.dissonance if self.musical_params else None,
                "timing_feel": self.musical_params.timing_feel.value if self.musical_params else None,
            } if self.musical_params else None,
            "mixer_params": self.mixer_params.to_dict() if self.mixer_params else None,
            "paths": self._paths,
        }

    def describe(self) -> str:
        """Get human-readable description of current state."""
        lines = []

        if self.emotional_state:
            lines.extend([
                f"Emotion: {self.emotional_state.primary_emotion}",
                f"Valence: {self.emotional_state.valence}",
                f"Arousal: {self.emotional_state.arousal}",
            ])

        if self.musical_params:
            lines.extend([
                "",
                f"Tempo: {self.musical_params.tempo_suggested} BPM",
                f"Timing: {self.musical_params.timing_feel.value}",
                f"Dissonance: {self.musical_params.dissonance:.0%}",
            ])

        if self.mixer_params:
            lines.extend([
                "",
                f"Mixer: {self.mixer_params.description}",
                f"Reverb: {self.mixer_params.reverb_mix:.0%}",
                f"Compression: {self.mixer_params.compression_ratio:.1f}:1",
            ])

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_generate(emotional_text: str) -> GeneratedMusic:
    """Quick helper to generate music from text."""
    brain = MusicBrain()
    return brain.generate_from_text(emotional_text)


def quick_export(emotional_text: str, output_path: str) -> dict[str, str]:
    """Quick helper to generate and export to Logic Pro."""
    brain = MusicBrain()
    music = brain.generate_from_text(emotional_text)
    return brain.export_to_logic(music, output_path)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MUSIC BRAIN EMOTION API - EXAMPLES")
    print("=" * 70)

    brain = MusicBrain()

    # Example 1: Simple text-based generation
    print("\n[Example 1: Text -> Music]")
    music = brain.generate_from_text("grief and loss processing")
    print(f"  Emotion: {music.emotional_state.primary_emotion}")
    print(f"  Tempo: {music.musical_params.tempo_suggested} BPM")
    print(f"  Reverb: {music.mixer_params.reverb_mix:.0%}")
    print(f"  Description: {music.mixer_params.description}")

    # Example 2: Fluent API
    print("\n[Example 2: Fluent Chain]")
    result = (brain.process("anxiety and tension")
                   .map_to_emotion()
                   .map_to_music()
                   .with_tempo(110)
                   .map_to_mixer()
                   .get())
    print(f"  Emotion: {result['emotional_state']['primary_emotion']}")
    print(f"  Tempo: {result['musical_params']['tempo']}")
    print(f"  Compression: {result['mixer_params']['compression']['ratio']:.1f}:1")

    # Example 3: From intent
    print("\n[Example 3: From Intent]")
    intent = brain.create_intent(
        title="Test Song",
        core_event="Processing loss",
        mood_primary="grief",
        technical_key="F",
        technical_mode="major",
        tempo_range=(78, 86),
        rule_to_break="HARMONY_ModalInterchange",
        rule_justification="Bbm creates bittersweet hope"
    )
    music_from_intent = brain.generate_from_intent(intent)
    print(f"  Title: {intent.title}")
    print(f"  Emotion: {music_from_intent.emotional_state.primary_emotion}")
    print(f"  Tempo: {music_from_intent.musical_params.tempo_suggested} BPM")

    # Example 4: Export
    print("\n[Example 4: Export to Logic Pro]")
    paths = brain.export_to_logic(music, "example_grief")
    print(f"  Automation file: {paths['automation']}")

    # List available presets
    print("\n[Available Mixer Presets]")
    print(f"  {', '.join(brain.list_mixer_presets())}")

    print("\n" + "=" * 70)
