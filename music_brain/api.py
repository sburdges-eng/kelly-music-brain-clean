"""
DAiW API Wrapper - Clean interface for desktop app and future REST API.

This module provides a simplified, consistent API surface for all music_brain
functionality, making it easier to integrate with desktop apps, web services,
or other interfaces.
"""
from typing import Dict, List, Optional, Any, Tuple
import sys
import logging

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FASTAPI_AVAILABLE = False
from pathlib import Path
import tempfile
import os

import numpy as np

# Core imports
from music_brain.audio import (
    AudioAnalyzer,
    AudioAnalysis,
    analyze_feel,
    AudioFeatures,
)
from music_brain.harmony import (
    HarmonyGenerator,
    HarmonyResult,
    generate_midi_from_harmony,
)
from music_brain.groove import (
    extract_groove,
    apply_groove,
    GrooveTemplate,
    humanize_midi_file,
    GrooveSettings,
    settings_from_preset,
    list_presets,
    get_preset,
)
from music_brain.structure import (
    analyze_chords,
    detect_sections,
    ChordProgression,
)
from music_brain.structure.progression import (
    diagnose_progression,
    generate_reharmonizations,
)
from music_brain.structure.comprehensive_engine import (
    TherapySession,
    render_plan_to_midi,
    HarmonyPlan,
)
from music_brain.session.intent_schema import (
    CompleteSongIntent,
    suggest_rule_break,
    validate_intent,
    list_all_rules,
)
from music_brain.session.intent_processor import process_intent
from music_brain.data.emotional_mapping import EMOTIONAL_PRESETS
from music_brain.voice import (
    AutoTuneProcessor,
    AutoTuneSettings,
    get_auto_tune_preset,
    VoiceModulator,
    ModulationSettings,
    get_modulation_preset,
    VoiceSynthesizer,
    SynthConfig,
    get_voice_profile,
)


class DAiWAPI:
    """
    Unified API wrapper for DAiW functionality.
    
    Provides a clean, consistent interface for all music_brain operations,
    making it easier to integrate with desktop apps, web services, or CLI tools.
    """
    
    def __init__(self):
        self.harmony_generator = HarmonyGenerator()
        self.auto_tune_processor = AutoTuneProcessor()
        self.voice_modulator = VoiceModulator()
        self.voice_synthesizer = VoiceSynthesizer()
    
    # ========== Harmony Generation ==========
    
    def generate_harmony_from_intent(
        self,
        intent: CompleteSongIntent,
        output_midi: Optional[str] = None,
        tempo_bpm: int = 82
    ) -> Dict[str, Any]:
        """
        Generate harmony from a CompleteSongIntent.
        
        Args:
            intent: CompleteSongIntent object
            output_midi: Optional path to save MIDI file
            tempo_bpm: Tempo for MIDI output
            
        Returns:
            Dict with harmony result and optional MIDI path
        """
        harmony = self.harmony_generator.generate_from_intent(intent)
        
        result = {
            "harmony": {
                "chords": harmony.chords,
                "key": harmony.key,
                "mode": harmony.mode,
                "rule_break_applied": harmony.rule_break_applied,
                "emotional_justification": harmony.emotional_justification,
            },
            "voicings": [
                {
                    "root": v.root,
                    "notes": v.notes,
                    "duration_beats": v.duration_beats,
                    "roman_numeral": v.roman_numeral,
                }
                for v in harmony.voicings
            ],
        }
        
        if output_midi:
            generate_midi_from_harmony(harmony, output_midi, tempo_bpm=tempo_bpm)
            result["midi_path"] = output_midi
        
        return result
    
    def generate_basic_progression(
        self,
        key: str = "C",
        mode: str = "major",
        pattern: str = "I-V-vi-IV",
        output_midi: Optional[str] = None,
        tempo_bpm: int = 82
    ) -> Dict[str, Any]:
        """
        Generate a basic chord progression.
        
        Args:
            key: Musical key (e.g., "C", "F", "Bb")
            mode: Mode (major, minor, dorian, etc.)
            pattern: Roman numeral pattern (e.g., "I-V-vi-IV")
            output_midi: Optional path to save MIDI file
            tempo_bpm: Tempo for MIDI output
            
        Returns:
            Dict with harmony result
        """
        harmony = self.harmony_generator.generate_basic_progression(
            key=key,
            mode=mode,
            pattern=pattern
        )
        
        result = {
            "harmony": {
                "chords": harmony.chords,
                "key": harmony.key,
                "mode": harmony.mode,
                "rule_break_applied": harmony.rule_break_applied,
                "emotional_justification": harmony.emotional_justification,
            },
        }
        
        if output_midi:
            generate_midi_from_harmony(harmony, output_midi, tempo_bpm=tempo_bpm)
            result["midi_path"] = output_midi
        
        return result
    
    # ========== Groove Operations ==========
    
    def extract_groove_from_midi(
        self,
        midi_path: str
    ) -> Dict[str, Any]:
        """
        Extract groove pattern from a MIDI file.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            Dict with groove analysis data
        """
        groove = extract_groove(midi_path)
        return groove.to_dict()
    
    def apply_groove_to_midi(
        self,
        midi_path: str,
        genre: str = "funk",
        intensity: float = 0.5,
        output_path: Optional[str] = None
    ) -> str:
        """
        Apply a genre groove template to a MIDI file.
        
        Args:
            midi_path: Path to input MIDI file
            genre: Genre template (funk, jazz, rock, etc.)
            intensity: Groove intensity (0.0-1.0)
            output_path: Optional output path (auto-generated if None)
            
        Returns:
            Path to output MIDI file
        """
        if output_path is None:
            output_path = str(Path(midi_path).with_suffix('.grooved.mid'))
        
        apply_groove(midi_path, genre=genre, output=output_path, intensity=intensity)
        return output_path
    
    def humanize_drums(
        self,
        midi_path: str,
        complexity: float = 0.5,
        vulnerability: float = 0.5,
        preset: Optional[str] = None,
        drum_channel: int = 9,
        enable_ghost_notes: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply humanization to drum track in MIDI file.
        
        Args:
            midi_path: Path to input MIDI file
            complexity: Timing chaos (0.0-1.0)
            vulnerability: Dynamic fragility (0.0-1.0)
            preset: Optional preset name (overrides complexity/vulnerability)
            drum_channel: MIDI channel for drums (default 9 = channel 10)
            enable_ghost_notes: Whether to add ghost notes
            output_path: Optional output path
            
        Returns:
            Dict with result info and output path
        """
        if preset:
            settings = settings_from_preset(preset)
            complexity = settings.complexity
            vulnerability = settings.vulnerability
        else:
            settings = GrooveSettings(
                complexity=complexity,
                vulnerability=vulnerability,
                enable_ghost_notes=enable_ghost_notes
            )
        
        if output_path is None:
            output_path = str(Path(midi_path).with_suffix('.humanized.mid'))
        
        result_path = humanize_midi_file(
            input_path=midi_path,
            output_path=output_path,
            complexity=complexity,
            vulnerability=vulnerability,
            drum_channel=drum_channel,
            settings=settings,
        )
        
        return {
            "output_path": result_path,
            "complexity": complexity,
            "vulnerability": vulnerability,
            "preset_used": preset,
        }
    
    # ========== Chord Analysis ==========
    
    def analyze_midi_chords(
        self,
        midi_path: str,
        include_sections: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze chord progression in a MIDI file.
        
        Args:
            midi_path: Path to MIDI file
            include_sections: Whether to also detect sections
            
        Returns:
            Dict with chord analysis and optional sections
        """
        progression = analyze_chords(midi_path)
        
        result = {
            "key": progression.key,
            "chords": progression.chords,
            "roman_numerals": progression.roman_numerals,
            "borrowed_chords": progression.borrowed_chords,
        }
        
        if include_sections:
            sections = detect_sections(midi_path)
            result["sections"] = [
                {
                    "name": s.name,
                    "start_bar": s.start_bar,
                    "end_bar": s.end_bar,
                    "energy": s.energy,
                }
                for s in sections
            ]
        
        return result
    
    def diagnose_progression(
        self,
        progression: str
    ) -> Dict[str, Any]:
        """
        Diagnose issues in a chord progression string.
        
        Args:
            progression: Chord progression (e.g., "F-C-Am-Dm")
            
        Returns:
            Dict with diagnosis results
        """
        return diagnose_progression(progression)
    
    def suggest_reharmonizations(
        self,
        progression: str,
        style: str = "jazz",
        count: int = 3
    ) -> List[Dict[str, str]]:
        """
        Generate reharmonization suggestions.
        
        Args:
            progression: Chord progression string
            style: Reharmonization style (jazz, pop, rnb, etc.)
            count: Number of suggestions
            
        Returns:
            List of reharmonization suggestions
        """
        return generate_reharmonizations(progression, style=style, count=count)
    
    # ========== Audio Analysis ==========
    
    def analyze_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze an audio file, returning tempo, key, spectrum, and chords.
        """
        analyzer = AudioAnalyzer()
        return analyzer.analyze_file(audio_path).to_dict()
    
    def analyze_audio_waveform(self, samples: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        analyzer = AudioAnalyzer(sample_rate=sample_rate)
        return analyzer.analyze_waveform(samples, sample_rate).to_dict()
    
    def detect_audio_bpm(self, samples: np.ndarray, sample_rate: int) -> float:
        analyzer = AudioAnalyzer(sample_rate=sample_rate)
        bpm, _ = analyzer.detect_bpm(samples, sample_rate)
        return bpm
    
    def detect_audio_key(self, samples: np.ndarray, sample_rate: int) -> Tuple[str, str]:
        analyzer = AudioAnalyzer(sample_rate=sample_rate)
        return analyzer.detect_key(samples, sample_rate)
    
    # ========== Voice Processing ==========
    
    def auto_tune_vocals(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        preset: str = "transparent",
        key: Optional[str] = None,
        mode: str = "major",
    ) -> str:
        settings = get_auto_tune_preset(preset)
        processor = AutoTuneProcessor(settings)
        return processor.process_file(input_path, output_path, key, mode)
    
    def modulate_voice(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        preset: str = "intimate_whisper",
    ) -> str:
        settings = get_modulation_preset(preset)
        modulator = VoiceModulator(settings)
        return modulator.process_file(input_path, output_path)
    
    def synthesize_voice(
        self,
        lyrics: str,
        melody_midi: List[int],
        tempo_bpm: int = 82,
        output_path: str = "guide_vocal.wav",
        profile: str = "guide_vulnerable",
    ) -> str:
        config = get_voice_profile(profile)
        synthesizer = VoiceSynthesizer(config)
        return synthesizer.synthesize_guide(
            lyrics=lyrics,
            melody_midi=melody_midi,
            tempo_bpm=tempo_bpm,
            output_path=output_path,
        )
    
    def speak_text_prompt(
        self,
        text: str,
        output_path: str = "spoken_prompt.wav",
        profile: str = "guide_confident",
        tempo_bpm: int = 80,
    ) -> str:
        config = get_voice_profile(profile)
        synthesizer = VoiceSynthesizer(config)
        return synthesizer.speak_text(
            text=text,
            output_path=output_path,
            profile=profile,
            tempo_bpm=tempo_bpm,
        )
    
    # ========== Therapy Session ==========
    
    def therapy_session(
        self,
        text: str,
        motivation: int = 7,
        chaos_tolerance: float = 0.5,
        output_midi: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process emotional text through therapy session and generate MIDI.
        
        Args:
            text: Emotional text input
            motivation: Motivation level (1-10)
            chaos_tolerance: Chaos tolerance (0.0-1.0)
            output_midi: Optional path to save MIDI file
            
        Returns:
            Dict with analysis and plan, plus optional MIDI path
        """
        session = TherapySession()
        affect = session.process_core_input(text)
        session.set_scales(motivation, chaos_tolerance)
        plan = session.generate_plan()
        
        result = {
            "affect": {
                "primary": affect,
                "secondary": session.state.affect_result.secondary if session.state.affect_result else None,
                "intensity": session.state.affect_result.intensity if session.state.affect_result else 0.0,
            },
            "plan": {
                "root_note": plan.root_note,
                "mode": plan.mode,
                "tempo_bpm": plan.tempo_bpm,
                "length_bars": plan.length_bars,
                "chord_symbols": plan.chord_symbols,
                "complexity": plan.complexity,
            },
        }
        
        if output_midi:
            midi_path = render_plan_to_midi(plan, output_midi)
            result["midi_path"] = midi_path
        
        return result
    
    # ========== Intent Processing ==========
    
    def process_song_intent(
        self,
        intent: CompleteSongIntent,
        output_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a CompleteSongIntent and generate all musical elements.
        
        Args:
            intent: CompleteSongIntent object
            output_json: Optional path to save results as JSON
            
        Returns:
            Dict with all generated elements
        """
        result = process_intent(intent)
        
        # Convert to serializable format
        output = {
            "intent_summary": result['intent_summary'],
            "harmony": {
                "chords": result['harmony'].chords,
                "roman_numerals": result['harmony'].roman_numerals,
                "rule_broken": result['harmony'].rule_broken,
                "rule_effect": result['harmony'].rule_effect,
            },
            "groove": {
                "pattern_name": result['groove'].pattern_name,
                "tempo_bpm": result['groove'].tempo_bpm,
                "swing_factor": result['groove'].swing_factor,
                "rule_broken": result['groove'].rule_broken,
                "rule_effect": result['groove'].rule_effect,
            },
            "arrangement": {
                "sections": result['arrangement'].sections,
                "dynamic_arc": result['arrangement'].dynamic_arc,
                "rule_broken": result['arrangement'].rule_broken,
            },
            "production": {
                "vocal_treatment": result['production'].vocal_treatment,
                "eq_notes": result['production'].eq_notes,
                "dynamics_notes": result['production'].dynamics_notes,
                "rule_broken": result['production'].rule_broken,
            },
        }
        
        if output_json:
            import json
            with open(output_json, 'w') as f:
                json.dump(output, f, indent=2)
        
        return output
    
    def suggest_rule_breaks(
        self,
        emotion: str
    ) -> List[Dict[str, str]]:
        """
        Get rule-breaking suggestions for an emotion.
        
        Args:
            emotion: Target emotion (e.g., "grief", "anger")
            
        Returns:
            List of rule-breaking suggestions
        """
        return suggest_rule_break(emotion)
    
    def list_available_rules(self) -> Dict[str, List[str]]:
        """
        List all available rule-breaking options.
        
        Returns:
            Dict mapping categories to lists of rules
        """
        return list_all_rules()
    
    def validate_song_intent(
        self,
        intent: CompleteSongIntent
    ) -> List[str]:
        """
        Validate a CompleteSongIntent.
        
        Args:
            intent: CompleteSongIntent to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        return validate_intent(intent)
    
    # ========== Preset Management ==========
    
    def list_humanization_presets(self) -> List[str]:
        """List available humanization presets."""
        return list_presets()
    
    def get_humanization_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """Get information about a humanization preset."""
        return get_preset(preset_name)


# Convenience instance
api = DAiWAPI()

__all__ = ['DAiWAPI', 'api']


# ---------- Minimal HTTP API (FastAPI) ----------
# This provides the server that `python -m music_brain.api` is expected to start.

if FASTAPI_AVAILABLE:
    class TechnicalIntent(BaseModel):
        key: Optional[str] = None
        bpm: Optional[int] = None
        progression: Optional[List[str]] = None
        genre: Optional[str] = None

    class EmotionalIntent(BaseModel):
        core_wound: Optional[str] = None
        core_desire: Optional[str] = None
        emotional_intent: str
        technical: Optional[TechnicalIntent] = None

    class GenerateRequest(BaseModel):
        intent: EmotionalIntent
        output_format: Optional[str] = None

    class InterrogateRequest(BaseModel):
        message: str
        session_id: Optional[str] = None
        context: Optional[Dict[str, Any]] = None

    app = FastAPI(title="Music Brain API", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/emotions")
    async def list_emotions():
        try:
            return sorted(EMOTIONAL_PRESETS.keys())
        except Exception as exc:  # pragma: no cover
            logging.exception("Failed to list emotions")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/generate")
    async def generate_music(request: GenerateRequest):
        try:
            # Map the simple intent into the therapy session pipeline
            chaos = 0.5
            motivation = 7
            if request.intent.technical and request.intent.technical.bpm:
                # Use bpm as a proxy for motivation scaling (soft heuristic)
                motivation = max(1, min(10, int(request.intent.technical.bpm / 20)))
            result = api.therapy_session(
                text=request.intent.emotional_intent,
                motivation=motivation,
                chaos_tolerance=chaos,
                output_midi=None,
            )
            return {"status": "success", "result": result}
        except Exception as exc:
            logging.exception("generate failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/interrogate")
    async def interrogate(request: InterrogateRequest):
        # Placeholder: echo back the message with a simple tip
        try:
            return {
                "status": "success",
                "reply": f"Noted: {request.message}. Consider clarifying the desired mood or groove.",
                "session_id": request.session_id,
            }
        except Exception as exc:  # pragma: no cover
            logging.exception("interrogate failed")
            raise HTTPException(status_code=500, detail=str(exc))


def _main():
    """Entry point for `python -m music_brain.api`."""
    if not FASTAPI_AVAILABLE:
        print(
            "FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    uvicorn.run("music_brain.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    _main()

