"""Tests for Kelly core modules."""

import pytest
from kelly.core.emotion_thesaurus import EmotionThesaurus, EmotionCategory
from kelly.core.intent_processor import IntentProcessor, Wound, WoundType
from kelly.core.midi_generator import MidiGenerator, note_to_midi, get_scale_pitches
from kelly.core.emotional_mapping import (
    EmotionalState, get_parameters_for_state, EMOTIONAL_PRESETS
)


class TestEmotionThesaurus:
    def test_thesaurus_initialization(self):
        thesaurus = EmotionThesaurus()
        assert len(thesaurus.nodes) > 0
        assert len(thesaurus.name_index) > 0
    
    def test_get_emotion_by_name(self):
        thesaurus = EmotionThesaurus()
        grief = thesaurus.get_emotion("grief")
        assert grief is not None
        assert grief.name == "grief"
        assert grief.category == EmotionCategory.SADNESS
    
    def test_get_emotion_case_insensitive(self):
        thesaurus = EmotionThesaurus()
        assert thesaurus.get_emotion("GRIEF") is not None
        assert thesaurus.get_emotion("Grief") is not None
    
    def test_unknown_emotion_returns_none(self):
        thesaurus = EmotionThesaurus()
        assert thesaurus.get_emotion("nonexistent") is None
    
    def test_find_by_category(self):
        thesaurus = EmotionThesaurus()
        sadness_emotions = thesaurus.find_by_category(EmotionCategory.SADNESS)
        assert len(sadness_emotions) > 0
        assert all(e.category == EmotionCategory.SADNESS for e in sadness_emotions)
    
    def test_find_similar(self):
        thesaurus = EmotionThesaurus()
        similar = thesaurus.find_similar("grief", n=3)
        assert len(similar) <= 3
    
    def test_emotion_has_musical_mapping(self):
        thesaurus = EmotionThesaurus()
        grief = thesaurus.get_emotion("grief")
        assert grief.musical_mapping is not None
        assert grief.musical_mapping.mode in ["minor", "major", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian", "harmonic_minor", "melodic_minor"]
    
    def test_list_all_emotions(self):
        thesaurus = EmotionThesaurus()
        all_emotions = thesaurus.list_all()
        assert "grief" in all_emotions
        assert "joy" in all_emotions
        assert "anger" in all_emotions


class TestIntentProcessor:
    def test_process_grief_wound(self):
        processor = IntentProcessor()
        wound = Wound("the loss of my best friend", intensity=0.9)
        result = processor.process_intent(wound)
        
        assert result.emotion is not None
        assert result.emotion.category == EmotionCategory.SADNESS
        assert len(result.rule_breaks) > 0
    
    def test_wound_type_inference(self):
        wound = Wound("feeling betrayed by someone I trusted")
        assert wound.wound_type == WoundType.BETRAYAL
        
        wound2 = Wound("fear of the unknown")
        assert wound2.wound_type == WoundType.FEAR
    
    def test_rule_breaks_generated(self):
        processor = IntentProcessor()
        wound = Wound("overwhelming anger", intensity=0.8)
        result = processor.process_intent(wound)
        
        assert len(result.rule_breaks) > 0
        rule_types = [rb.rule_type for rb in result.rule_breaks]
        assert any("HARMONY" in rt or "RHYTHM" in rt or "PRODUCTION" in rt for rt in rule_types)
    
    def test_musical_params_compiled(self):
        processor = IntentProcessor()
        wound = Wound("feeling anxious", intensity=0.6)
        result = processor.process_intent(wound)
        
        assert "mode" in result.musical_params
        assert "tempo_modifier" in result.musical_params
        assert "velocity_range" in result.musical_params
    
    def test_intensity_clamping(self):
        wound = Wound("test", intensity=1.5)
        assert wound.intensity == 1.0
        
        wound2 = Wound("test", intensity=-0.5)
        assert wound2.intensity == 0.0


class TestMidiGenerator:
    def test_note_to_midi(self):
        assert note_to_midi("C", 4) == 60
        assert note_to_midi("A", 4) == 69
        assert note_to_midi("C#", 4) == 61
    
    def test_get_scale_pitches(self):
        c_major = get_scale_pitches("C", "major", 4)
        assert len(c_major) == 7
        assert c_major[0] == 60  # C4
    
    def test_generator_initialization(self):
        gen = MidiGenerator(tempo=82, key="F", mode="minor")
        assert gen.tempo == 82
        assert gen.key == "F"
        assert gen.mode == "minor"
    
    def test_generate_chord_progression(self):
        gen = MidiGenerator()
        progression = gen.generate_chord_progression(bars=4)
        assert len(progression) == 4
        assert all(len(v.pitches) >= 3 for v in progression)
    
    def test_chords_to_notes(self):
        gen = MidiGenerator()
        progression = gen.generate_chord_progression(bars=2)
        notes = gen.chords_to_notes(progression)
        assert len(notes) > 0
        assert all(hasattr(n, 'pitch') for n in notes)
    
    def test_generate_melody(self):
        gen = MidiGenerator()
        melody = gen.generate_melody(bars=2, density=0.5)
        assert len(melody) > 0
    
    def test_generate_bass(self):
        gen = MidiGenerator()
        progression = gen.generate_chord_progression(bars=2)
        bass = gen.generate_bass(progression, pattern="root_fifth")
        assert len(bass) > 0


class TestEmotionalMapping:
    def test_preset_exists(self):
        assert "grief" in EMOTIONAL_PRESETS
        assert "joy" in EMOTIONAL_PRESETS
        assert "anger" in EMOTIONAL_PRESETS
    
    def test_get_parameters_for_state(self):
        state = EmotionalState(
            valence=-0.7,
            arousal=0.3,
            primary_emotion="grief"
        )
        params = get_parameters_for_state(state)
        
        assert params.tempo_suggested > 0
        assert params.velocity_min < params.velocity_max
        assert 0 <= params.dissonance <= 1
    
    def test_emotional_state_quadrant(self):
        state = EmotionalState(valence=0.5, arousal=0.7, primary_emotion="joy")
        assert state.quadrant() == "excited_positive"
        
        state2 = EmotionalState(valence=-0.5, arousal=0.2, primary_emotion="sadness")
        assert state2.quadrant() == "calm_negative"


class TestEngines:
    def test_groove_engine(self):
        from kelly.engines.groove_engine import GrooveEngine
        engine = GrooveEngine()
        notes = [{"tick": 0, "velocity": 80, "pitch": 60, "duration": 480}]
        grooved = engine.apply_groove(notes, emotion="grief")
        assert len(grooved) >= 1
    
    def test_bass_engine(self):
        from kelly.engines.bass_engine import BassEngine, BassConfig
        engine = BassEngine()
        config = BassConfig(emotion="grief", chord_progression=["F", "C", "Dm"], bars=2)
        output = engine.generate(config)
        assert len(output.notes) > 0
    
    def test_melody_engine(self):
        from kelly.engines.melody_engine import MelodyEngine, MelodyConfig
        engine = MelodyEngine()
        config = MelodyConfig(emotion="hope", bars=2)
        output = engine.generate(config)
        assert len(output.notes) > 0
    
    def test_rhythm_engine(self):
        from kelly.engines.rhythm_engine import RhythmEngine, RhythmConfig
        engine = RhythmEngine()
        config = RhythmConfig(emotion="anger", bars=2)
        output = engine.generate(config)
        assert len(output.hits) > 0
    
    def test_dynamics_engine(self):
        from kelly.engines.dynamics_engine import DynamicsEngine
        engine = DynamicsEngine()
        curve = engine.generate_curve("grief", duration_ticks=1920)
        assert len(curve.points) > 0
    
    def test_arrangement_engine(self):
        from kelly.engines.arrangement_engine import ArrangementEngine
        engine = ArrangementEngine()
        plan = engine.generate_arrangement("grief", duration_minutes=2.0)
        assert len(plan.sections) > 0
        assert plan.total_bars > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
