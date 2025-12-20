"""
Music Brain API Server
Provides endpoints for emotional music generation
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from music_brain.auth import (
    get_current_active_user,
    User,
    Token,
    create_access_token,
    TokenData,
    security,
)
from music_brain.middleware import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
)
from music_brain.metrics import metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("music_brain.api")

# Create FastAPI app
app = FastAPI(
    title="Music Brain API",
    version="1.0.0",
    description="Intelligent music analysis, generation, and production toolkit",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware, logger=logger)

# Add rate limiting middleware (configurable via environment)
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
if RATE_LIMIT_ENABLED:
    requests_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    requests_per_hour = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    requests_per_day = int(os.getenv("RATE_LIMIT_PER_DAY", "10000"))
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        requests_per_day=requests_per_day,
    )
    logger.info(
        f"Rate limiting enabled: {requests_per_minute}/min, "
        f"{requests_per_hour}/hour, {requests_per_day}/day"
    )

# Load emotion thesaurus
EMOTION_THESAURUS_PATH = Path(__file__).parent.parent / "emotion_thesaurus"


class EmotionalIntent(BaseModel):
    core_wound: Optional[str] = None
    core_desire: Optional[str] = None
    emotional_intent: str
    technical: Optional[Dict[str, Any]] = None


class GenerateRequest(BaseModel):
    intent: EmotionalIntent
    output_format: str = "midi"


class InterrogateRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


# =============================================================================
# Voice Processing Request/Response Models
# =============================================================================

class AutoTuneSettingsModel(BaseModel):
    """Auto-tune settings configuration."""
    correction_strength: float = 0.8
    correction_speed_ms: float = 50.0
    humanize: float = 0.1
    vibrato_preservation: float = 0.7
    preserve_formants: bool = True
    snap_threshold_cents: float = 50.0
    natural_transitions: bool = True


class AutoTuneRequest(BaseModel):
    """Request for auto-tune processing."""
    input_path: str
    output_path: Optional[str] = None
    key: Optional[str] = None
    mode: str = "major"
    preset: Optional[str] = None  # Use preset instead of custom settings
    settings: Optional[AutoTuneSettingsModel] = None


class AutoTuneResponse(BaseModel):
    """Response from auto-tune processing."""
    success: bool
    output_path: Optional[str] = None
    key: Optional[str] = None
    mode: str = "major"
    preset_used: Optional[str] = None
    message: str


class ModulationSettingsModel(BaseModel):
    """Voice modulation settings configuration."""
    pitch_shift: float = 0.0
    formant_shift: float = 0.0
    breathiness: float = 0.0
    whisper_mix: float = 0.0
    robotic: float = 0.0
    warmth: float = 0.5
    presence: float = 0.5
    reverb: float = 0.1
    compression: float = 0.3


class ModulationRequest(BaseModel):
    """Request for voice modulation."""
    input_path: str
    output_path: Optional[str] = None
    preset: Optional[str] = None  # Use preset instead of custom settings
    settings: Optional[ModulationSettingsModel] = None


class ModulationResponse(BaseModel):
    """Response from voice modulation."""
    success: bool
    output_path: Optional[str] = None
    preset_used: Optional[str] = None
    message: str


class SynthesisRequest(BaseModel):
    """Request for voice synthesis."""
    text: str
    output_path: str = "synthesis_output.wav"
    profile: str = "narrator_neutral"
    tempo_bpm: Optional[int] = None  # For guide vocal synthesis
    melody_midi: Optional[List[int]] = None  # For melodic synthesis


class SynthesisResponse(BaseModel):
    """Response from voice synthesis."""
    success: bool
    output_path: Optional[str] = None
    profile_used: str
    message: str


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint providing API information."""
    return {
        "service": "Music Brain API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/generate", "/interrogate", "/emotions"],
    }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    metrics.increment("health_check")
    return {
        "status": "healthy",
        "service": "Music Brain API",
        "version": "1.0.0",
    }


@app.get("/metrics")
async def get_metrics(
    current_user: User = Security(get_current_active_user)
) -> Dict[str, Any]:
    """Get metrics summary (requires authentication)."""
    metrics.increment("metrics_requested")
    return metrics.get_metrics_summary()

@app.post("/generate")
async def generate_music(request: GenerateRequest) -> Dict[str, Any]:
    """Generate music from emotional intent.

    Args:
        request: GenerateRequest containing emotional intent and technical parameters

    Returns:
        Dictionary containing success status, intent, song data, and message

    Raises:
        HTTPException: If generation fails or parameters are invalid
    """
    import traceback

    from music_brain.session.generator import SongGenerator

    metrics.increment("generate_requested", tags={"endpoint": "generate"})
    
    with metrics.timer("generate_duration", tags={"endpoint": "generate"}):
        try:
            # Extract parameters from intent
            intent = request.intent
            technical = intent.technical or {}

            # Parse key and mode from technical.key (e.g., "F major" or "C minor")
            key = "C"
            mode = "major"
            if technical.get("key"):
                key_str = str(technical["key"]).strip()
                if " " in key_str:
                    parts = key_str.split(" ", 1)
                    key = parts[0]
                    mode = parts[1].lower() if len(parts) > 1 else "major"
                else:
                    key = key_str

            # Extract other parameters
            tempo = technical.get("bpm")
            genre = technical.get("genre")
            mood = intent.emotional_intent or None

            # Generate song
            generator = SongGenerator()
            song = generator.generate(
                key=key, mode=mode, mood=mood, genre=genre, tempo=float(tempo) if tempo else None
            )

            # Convert to dict for JSON response using the song's to_dict method
            result = {
                "success": True,
                "intent": intent.model_dump() if hasattr(intent, "model_dump") else intent.dict(),
                "song": song.to_dict(),
                "message": (
                    f"Generated {len(song.sections)} sections, {song.total_bars} bars "
                    f"at {song.tempo_bpm} BPM"
                ),
            }

            metrics.increment("generate_success", tags={"endpoint": "generate"})
            return result

        except Exception as e:
            metrics.increment("generate_error", tags={"endpoint": "generate", "error": type(e).__name__})
            logger.error(f"Error generating music: {str(e)}", exc_info=True)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/interrogate")
async def interrogate(request: InterrogateRequest) -> Dict[str, Any]:
    """Conversational music creation endpoint.

    Args:
        request: InterrogateRequest with message and optional session context

    Returns:
        Dictionary containing success status and response data

    Raises:
        HTTPException: If interrogation fails
    """
    metrics.increment("interrogate_requested", tags={"endpoint": "interrogate"})
    
    with metrics.timer("interrogate_duration", tags={"endpoint": "interrogate"}):
        try:
            from music_brain.session.interrogator import (
                SongInterrogator,
                SongPhase,
                quick_interrogate,
            )
            
            # Initialize interrogator
            interrogator = SongInterrogator()
            
            # If context provided, update interrogator's context
            if request.context:
                for key, value in request.context.items():
                    if hasattr(interrogator.context, key):
                        setattr(interrogator.context, key, value)
            
            # Determine phase from message or use current phase
            message_lower = request.message.lower()
            current_phase = interrogator.current_phase
            
            # Try to detect phase from message keywords
            phase_keywords = {
                SongPhase.INTENT: ["intent", "need", "want", "purpose", "goal"],
                SongPhase.EMOTION: ["emotion", "feel", "feeling", "mood", "vulnerable"],
                SongPhase.STORY: ["story", "about", "subject", "character", "narrative"],
                SongPhase.SOUND: ["sound", "tempo", "instrument", "reference", "texture"],
                SongPhase.STRUCTURE: ["structure", "build", "peak", "chorus", "verse"],
                SongPhase.LYRICS: ["lyrics", "line", "metaphor", "word", "phrase"],
                SongPhase.PRODUCTION: ["production", "polished", "raw", "mix", "style"],
            }
            
            for phase, keywords in phase_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    current_phase = phase
                    break
            
            # Get relevant questions for the detected phase
            questions = interrogator.quick_questions(current_phase, count=3)
            
            # Store the message as context
            if current_phase == SongPhase.INTENT:
                interrogator.context.core_emotion = request.message
            elif current_phase == SongPhase.STORY:
                interrogator.context.subject = request.message
            
            # Build response with questions and context
            result = {
                "success": True,
                "response": {
                    "message": f"Exploring {current_phase.value} - here are some questions to consider",
                    "questions": questions,
                    "phase": current_phase.value,
                    "context": {
                        "core_emotion": interrogator.context.core_emotion,
                        "subject": interrogator.context.subject,
                        "tempo_feel": interrogator.context.tempo_feel,
                        "production_style": interrogator.context.production_style,
                    },
                },
            }
            
            metrics.increment("interrogate_success", tags={"endpoint": "interrogate"})
            return result

        except Exception as e:
            metrics.increment("interrogate_error", tags={"endpoint": "interrogate", "error": type(e).__name__})
            logger.error(f"Error in interrogation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions")
async def get_emotions() -> Dict[str, Any]:
    """Get the full 6x6x6 emotion thesaurus.

    Returns:
        Dictionary containing success status, emotions data, and total node count

    Raises:
        HTTPException: If emotion data cannot be loaded
    """
    metrics.increment("emotions_requested", tags={"endpoint": "emotions"})
    
    with metrics.timer("emotions_duration", tags={"endpoint": "emotions"}):
        try:
            emotions = {}

            for emotion_file in EMOTION_THESAURUS_PATH.glob("*.json"):
                if emotion_file.stem not in ["metadata", "blends"]:
                    with open(emotion_file) as f:
                        emotions[emotion_file.stem] = json.load(f)

            blends_path = EMOTION_THESAURUS_PATH / "blends.json"
            if blends_path.exists():
                with open(blends_path) as f:
                    emotions["blends"] = json.load(f)

            metrics.increment("emotions_success", tags={"endpoint": "emotions"})
            return {"success": True, "emotions": emotions, "total_nodes": 216}

        except Exception as e:
            metrics.increment("emotions_error", tags={"endpoint": "emotions", "error": type(e).__name__})
            logger.error(f"Error loading emotions: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions/{base_emotion}")
async def get_emotion_category(base_emotion: str) -> Dict[str, Any]:
    """Get specific emotion category.

    Args:
        base_emotion: Name of the emotion category to retrieve

    Returns:
        Dictionary containing success status, emotion name, and emotion data

    Raises:
        HTTPException: If emotion category not found or cannot be loaded
    """
    metrics.increment("emotion_category_requested", tags={"endpoint": "emotions", "emotion": base_emotion})
    
    emotion_file = EMOTION_THESAURUS_PATH / f"{base_emotion}.json"

    if not emotion_file.exists():
        metrics.increment("emotion_category_not_found", tags={"endpoint": "emotions", "emotion": base_emotion})
        raise HTTPException(status_code=404, detail=f"Emotion '{base_emotion}' not found")

    with metrics.timer("emotion_category_duration", tags={"endpoint": "emotions", "emotion": base_emotion}):
        try:
            with open(emotion_file) as f:
                data = json.load(f)

            metrics.increment("emotion_category_success", tags={"endpoint": "emotions", "emotion": base_emotion})
            return {"success": True, "emotion": base_emotion, "data": data}

        except Exception as e:
            metrics.increment("emotion_category_error", tags={"endpoint": "emotions", "emotion": base_emotion, "error": type(e).__name__})
            logger.error(f"Error loading emotion category {base_emotion}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Voice Processing API Endpoints
# =============================================================================

@app.post("/voice/autotune", response_model=AutoTuneResponse)
async def process_autotune(request: AutoTuneRequest) -> AutoTuneResponse:
    """Process audio with pitch correction (auto-tune).

    Args:
        request: AutoTuneRequest with input path, key, mode, and settings

    Returns:
        AutoTuneResponse with output path and processing details
    """
    metrics.increment("autotune_requested", tags={"endpoint": "voice/autotune"})
    
    with metrics.timer("autotune_duration", tags={"endpoint": "voice/autotune"}):
        try:
            from music_brain.voice import AutoTuneProcessor, AutoTuneSettings, get_auto_tune_preset
            
            # Get settings from preset or custom
            if request.preset:
                settings = get_auto_tune_preset(request.preset)
                preset_used = request.preset
            elif request.settings:
                settings = AutoTuneSettings(**request.settings.model_dump())
                preset_used = None
            else:
                settings = AutoTuneSettings()
                preset_used = "default"
            
            # Process audio
            processor = AutoTuneProcessor(settings)
            output_path = processor.process_file(
                input_path=request.input_path,
                output_path=request.output_path,
                key=request.key,
                mode=request.mode,
            )
            
            metrics.increment("autotune_success", tags={"endpoint": "voice/autotune"})
            return AutoTuneResponse(
                success=True,
                output_path=output_path,
                key=request.key,
                mode=request.mode,
                preset_used=preset_used,
                message=f"Audio processed successfully with auto-tune ({preset_used or 'custom'} settings)",
            )
            
        except FileNotFoundError as e:
            metrics.increment("autotune_error", tags={"endpoint": "voice/autotune", "error": "file_not_found"})
            raise HTTPException(status_code=404, detail=f"Input file not found: {str(e)}")
        except ValueError as e:
            metrics.increment("autotune_error", tags={"endpoint": "voice/autotune", "error": "invalid_preset"})
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            metrics.increment("autotune_error", tags={"endpoint": "voice/autotune", "error": type(e).__name__})
            logger.error(f"Error in auto-tune processing: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/modulate", response_model=ModulationResponse)
async def process_modulation(request: ModulationRequest) -> ModulationResponse:
    """Apply voice character modulation to audio.

    Args:
        request: ModulationRequest with input path and modulation settings

    Returns:
        ModulationResponse with output path and processing details
    """
    metrics.increment("modulation_requested", tags={"endpoint": "voice/modulate"})
    
    with metrics.timer("modulation_duration", tags={"endpoint": "voice/modulate"}):
        try:
            from music_brain.voice import VoiceModulator, ModulationSettings, get_modulation_preset
            
            # Get settings from preset or custom
            if request.preset:
                settings = get_modulation_preset(request.preset)
                preset_used = request.preset
            elif request.settings:
                settings = ModulationSettings(**request.settings.model_dump())
                preset_used = None
            else:
                settings = ModulationSettings()
                preset_used = "neutral"
            
            # Process audio
            modulator = VoiceModulator(settings)
            output_path = modulator.process_file(
                input_path=request.input_path,
                output_path=request.output_path,
            )
            
            metrics.increment("modulation_success", tags={"endpoint": "voice/modulate"})
            return ModulationResponse(
                success=True,
                output_path=output_path,
                preset_used=preset_used,
                message=f"Voice modulation applied ({preset_used or 'custom'} settings)",
            )
            
        except FileNotFoundError as e:
            metrics.increment("modulation_error", tags={"endpoint": "voice/modulate", "error": "file_not_found"})
            raise HTTPException(status_code=404, detail=f"Input file not found: {str(e)}")
        except ValueError as e:
            metrics.increment("modulation_error", tags={"endpoint": "voice/modulate", "error": "invalid_preset"})
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            metrics.increment("modulation_error", tags={"endpoint": "voice/modulate", "error": type(e).__name__})
            logger.error(f"Error in voice modulation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/synthesize", response_model=SynthesisResponse)
async def synthesize_voice(request: SynthesisRequest) -> SynthesisResponse:
    """Synthesize voice from text (TTS) or create guide vocals.

    Args:
        request: SynthesisRequest with text, profile, and optional melody info

    Returns:
        SynthesisResponse with output path and synthesis details
    """
    metrics.increment("synthesis_requested", tags={"endpoint": "voice/synthesize"})
    
    with metrics.timer("synthesis_duration", tags={"endpoint": "voice/synthesize"}):
        try:
            # Use TTS-backed synthesizer (pyttsx3) with profile support
            from music_brain.voice.synth import VoiceSynthesizer

            synthesizer = VoiceSynthesizer()
            
            # Check if TTS is available
            if not synthesizer.is_available:
                return SynthesisResponse(
                    success=False,
                    output_path=None,
                    profile_used=request.profile,
                    message="Voice synthesis not available. Install pyttsx3: pip install pyttsx3",
                )
            
            # Use guide synthesis if tempo provided
            if request.tempo_bpm:
                output_path = synthesizer.synthesize_guide(
                    lyrics=request.text,
                    melody_midi=request.melody_midi,
                    tempo_bpm=request.tempo_bpm,
                    output_path=request.output_path,
                    profile=request.profile,
                )
            else:
                output_path = synthesizer.speak_text(
                    text=request.text,
                    output_path=request.output_path,
                    profile=request.profile,
                )
            
            if output_path:
                metrics.increment("synthesis_success", tags={"endpoint": "voice/synthesize"})
                return SynthesisResponse(
                    success=True,
                    output_path=output_path,
                    profile_used=request.profile,
                    message=f"Voice synthesized using '{request.profile}' profile",
                )
            else:
                return SynthesisResponse(
                    success=False,
                    output_path=None,
                    profile_used=request.profile,
                    message="Synthesis failed - check TTS configuration",
                )
            
        except KeyError as e:
            metrics.increment("synthesis_error", tags={"endpoint": "voice/synthesize", "error": "invalid_profile"})
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            metrics.increment("synthesis_error", tags={"endpoint": "voice/synthesize", "error": type(e).__name__})
            logger.error(f"Error in voice synthesis: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/voice/autotune/presets")
async def list_autotune_presets() -> Dict[str, Any]:
    """List available auto-tune presets.

    Returns:
        Dictionary with preset names and their configurations
    """
    metrics.increment("autotune_presets_requested", tags={"endpoint": "voice/autotune/presets"})
    
    try:
        from music_brain.voice.auto_tune import AUTO_TUNE_PRESETS
        
        presets = {
            name: settings.to_dict() 
            for name, settings in AUTO_TUNE_PRESETS.items()
        }
        
        return {
            "success": True,
            "presets": presets,
            "count": len(presets),
        }
    except Exception as e:
        logger.error(f"Error listing auto-tune presets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voice/modulation/presets")
async def list_modulation_presets() -> Dict[str, Any]:
    """List available voice modulation presets.

    Returns:
        Dictionary with preset names and their configurations
    """
    metrics.increment("modulation_presets_requested", tags={"endpoint": "voice/modulation/presets"})
    
    try:
        from music_brain.voice.modulator import MODULATION_PRESETS
        
        presets = {
            name: settings.to_dict() 
            for name, settings in MODULATION_PRESETS.items()
        }
        
        return {
            "success": True,
            "presets": presets,
            "count": len(presets),
        }
    except Exception as e:
        logger.error(f"Error listing modulation presets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voice/synthesis/profiles")
async def list_voice_profiles() -> Dict[str, Any]:
    """List available voice synthesis profiles.

    Returns:
        Dictionary with profile names and their configurations
    """
    metrics.increment("synthesis_profiles_requested", tags={"endpoint": "voice/synthesis/profiles"})
    
    try:
        from music_brain.voice.synth import VOICE_PROFILES
        
        profiles = {
            name: profile.to_dict() 
            for name, profile in VOICE_PROFILES.items()
        }
        
        return {
            "success": True,
            "profiles": profiles,
            "count": len(profiles),
        }
    except Exception as e:
        logger.error(f"Error listing voice profiles: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Compatibility API Class (for backward compatibility with tests)
# =============================================================================

class DAiWAPI:
    """
    Compatibility wrapper class for the older API interface.
    
    This class provides a programmatic interface to Music Brain functionality
    for backward compatibility with existing tests and code.
    """
    
    def __init__(self):
        """Initialize the API with all required components."""
        from music_brain.harmony import HarmonyGenerator
        from music_brain.audio.analyzer import AudioAnalyzer
        
        self.harmony_generator = HarmonyGenerator()
        self.audio_analyzer = AudioAnalyzer()
    
    def generate_basic_progression(
        self,
        key: str,
        mode: str,
        pattern: str,
        output_midi: Optional[str] = None,
        tempo_bpm: int = 120,
    ) -> Dict[str, Any]:
        """Generate a basic chord progression from Roman numeral pattern."""
        # Use HarmonyGenerator's method
        result = self.harmony_generator.generate_basic_progression(
            key=key,
            mode=mode,
            pattern=pattern,
        )
        
        response = {
            "harmony": {
                "key": result.key,
                "mode": result.mode,
                "chords": result.chords,
            },
            "voicings": [
                {
                    "root": v.root,
                    "notes": v.notes,
                    "duration_beats": v.duration_beats,
                    "velocity": v.velocity,
                }
                for v in result.voicings
            ],
        }
        
        # Generate MIDI if requested
        if output_midi:
            from music_brain.harmony import generate_midi_from_harmony
            generate_midi_from_harmony(result, output_midi, tempo_bpm=tempo_bpm)
            response["midi_path"] = output_midi
        
        return response
    
    def generate_harmony_from_intent(
        self,
        intent: Any,  # CompleteSongIntent
        output_midi: Optional[str] = None,
        tempo_bpm: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate harmony from a CompleteSongIntent."""
        # Use HarmonyGenerator's generate_from_intent method
        result = self.harmony_generator.generate_from_intent(intent)
        
        response = {
            "harmony": {
                "key": result.key,
                "mode": result.mode,
                "chords": result.chords,
            },
            "voicings": [
                {
                    "root": v.root,
                    "notes": v.notes,
                    "duration_beats": v.duration_beats,
                }
                for v in result.voicings
            ],
        }
        
        if output_midi:
            from music_brain.harmony import generate_midi_from_harmony
            tempo = tempo_bpm or 120
            generate_midi_from_harmony(result, output_midi, tempo_bpm=tempo)
            response["midi_path"] = output_midi
        
        return response
    
    def humanize_drums(
        self,
        midi_path: str,
        complexity: float = 0.5,
        vulnerability: float = 0.5,
        preset: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Humanize drum MIDI file."""
        import mido
        from music_brain.groove_engine import apply_groove
        
        # Load MIDI file
        midi_file = mido.MidiFile(midi_path)
        ppq = midi_file.ticks_per_beat
        
        # Convert MIDI to events
        events = []
        for track in midi_file.tracks:
            current_tick = 0
            for msg in track:
                current_tick += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    events.append({
                        "start_tick": current_tick,
                        "pitch": msg.note,
                        "velocity": msg.velocity,
                        "duration_ticks": 480,  # Default duration
                    })
        
        # Apply humanization
        if preset:
            # Use preset settings if available
            from music_brain.groove.groove_engine import settings_from_preset
            try:
                settings = settings_from_preset(preset)
                complexity = getattr(settings, "complexity", complexity)
                vulnerability = getattr(settings, "vulnerability", vulnerability)
            except:
                pass
        
        humanized_events = apply_groove(
            events,
            complexity=complexity,
            vulnerability=vulnerability,
            ppq=ppq,
        )
        
        # Save to output path
        if output_path:
            # Create new MIDI file with humanized events
            new_midi = mido.MidiFile(ticks_per_beat=ppq)
            track = mido.MidiTrack()
            new_midi.tracks.append(track)
            
            for event in humanized_events:
                track.append(mido.Message(
                    "note_on",
                    note=event["pitch"],
                    velocity=event["velocity"],
                    time=event["start_tick"],
                ))
            
            new_midi.save(output_path)
        
        result = {
            "complexity": complexity,
            "vulnerability": vulnerability,
            "output_path": output_path or midi_path,
        }
        
        if preset:
            result["preset_used"] = preset
        
        return result
    
    def diagnose_progression(self, progression_string: str) -> Dict[str, Any]:
        """Diagnose a chord progression."""
        from music_brain.structure.progression import diagnose_progression
        return diagnose_progression(progression_string)
    
    def suggest_reharmonizations(
        self,
        progression: str,
        style: str = "jazz",
        count: int = 3,
    ) -> List[Dict[str, Any]]:
        """Suggest reharmonizations for a progression."""
        from music_brain.structure.progression import generate_reharmonizations
        return generate_reharmonizations(progression, style=style, count=count)
    
    def therapy_session(
        self,
        text: str,
        motivation: int = 5,
        chaos_tolerance: float = 0.5,
        output_midi: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a therapy session and generate music."""
        from music_brain.structure.comprehensive_engine import TherapySession
        
        session = TherapySession()
        session.process_core_input(text)
        session.set_scales(motivation, chaos_tolerance)
        plan = session.generate_plan()
        
        response = {
            "affect": {
                "primary": session.state.affect_result.primary if session.state.affect_result else "neutral",
                "mode": session.state.suggested_mode,
            },
            "plan": plan.__dict__,
            "harmony_plan": {
                "root_note": plan.root_note,
                "mode": plan.mode,
                "tempo_bpm": plan.tempo_bpm,
                "chord_symbols": plan.chord_symbols,
            },
        }
        
        if output_midi:
            from music_brain.structure.comprehensive_engine import render_plan_to_midi
            render_plan_to_midi(plan, output_midi)
            response["midi_path"] = output_midi
        
        return response
    
    def list_humanization_presets(self) -> List[str]:
        """List available humanization presets."""
        from music_brain.groove.groove_engine import list_presets
        try:
            return list_presets()
        except:
            # Fallback list
            return [
                "tight_mechanical",
                "loose_human",
                "vulnerable_intimate",
                "confident_powerful",
            ]
    
    def get_humanization_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """Get information about a humanization preset."""
        from music_brain.groove.groove_engine import get_preset
        try:
            preset_data = get_preset(preset_name)
            if preset_data:
                return preset_data
        except:
            pass
        
        # Fallback
        return {
            "name": preset_name,
            "complexity": 0.5,
            "vulnerability": 0.5,
        }
    
    def analyze_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Analyze an audio file."""
        result = self.audio_analyzer.analyze_file(audio_path)
        return result.to_dict()
    
    def analyze_audio_waveform(self, samples: Any, sample_rate: int = 44100) -> Dict[str, Any]:
        """Analyze audio waveform data."""
        result = self.audio_analyzer.analyze_waveform(samples, sample_rate)
        return result.to_dict()
    
    def detect_audio_bpm(self, samples: Any, sample_rate: int = 44100) -> float:
        """Detect BPM from audio waveform samples or file path."""
        if isinstance(samples, str):
            # File path - load it
            import librosa
            audio_samples, sr = librosa.load(samples, sr=None)
            return self.audio_analyzer.detect_bpm(audio_samples, sr)
        else:
            # Waveform samples
            return self.audio_analyzer.detect_bpm(samples, sample_rate)
    
    def detect_audio_key(self, samples: Any, sample_rate: int = 44100) -> tuple:
        """Detect key from audio waveform samples or file path."""
        if isinstance(samples, str):
            # File path - load it first
            import librosa
            audio_samples, sr = librosa.load(samples, sr=None)
            return self.audio_analyzer.detect_key(audio_samples, sr)
        else:
            # Waveform samples
            return self.audio_analyzer.detect_key(samples, sample_rate)
    
    def suggest_rule_breaks(self, emotion: str) -> List[str]:
        """Suggest rule breaks for an emotion."""
        from music_brain.harmony import RuleBreakType
        # Return list of rule break types as strings
        return [rule.value for rule in RuleBreakType]
    
    def list_available_rules(self) -> Dict[str, List[str]]:
        """List available rule-breaking categories."""
        from music_brain.harmony import RuleBreakType
        rules = {
            "harmony": [],
            "rhythm": [],
            "production": [],
        }
        for rule in RuleBreakType:
            if rule.name.startswith("HARMONY_"):
                rules["harmony"].append(rule.value)
            elif rule.name.startswith("RHYTHM_"):
                rules["rhythm"].append(rule.value)
            elif rule.name.startswith("PRODUCTION_"):
                rules["production"].append(rule.value)
        return rules
    
    def validate_song_intent(self, intent: Any) -> List[str]:
        """Validate a CompleteSongIntent and return list of issues."""
        issues = []
        # Basic validation
        if not hasattr(intent, "song_root"):
            issues.append("Missing song_root")
        if not hasattr(intent, "technical_constraints"):
            issues.append("Missing technical_constraints")
        return issues
    
    def process_song_intent(self, intent: Any) -> Dict[str, Any]:
        """Process a CompleteSongIntent and generate music."""
        # Use generate_harmony_from_intent
        return self.generate_harmony_from_intent(intent)


# Convenience instance
api = DAiWAPI()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
