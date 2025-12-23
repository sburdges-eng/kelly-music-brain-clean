"""
Audio Analysis Tools - MCP tools for audio file analysis.

Provides 6 tools:
- detect_bpm
- detect_key
- analyze_audio_feel
- extract_chords
- detect_scale
- analyze_theory
"""

from typing import Any, Dict, List
import json
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import DAiW audio modules
try:
    from music_brain.audio import (
        analyze_feel,
        AudioFeatures,
        AudioAnalyzer,
        ChordDetector,
        TheoryAnalyzer,
        detect_chords_from_audio,
    )
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Try to import audio cataloger for BPM/key detection
try:
    from tools.audio_cataloger.audio_cataloger import detect_bpm_key
    AUDIO_CATALOGER_AVAILABLE = True
except ImportError:
    AUDIO_CATALOGER_AVAILABLE = False


def register_tools(server: Server) -> None:
    """Register all audio analysis tools with the MCP server."""
    if not MCP_AVAILABLE:
        return
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available audio analysis tools."""
        return [
            Tool(
                name="detect_bpm",
                description="Detect tempo (BPM) from an audio file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to audio file (WAV, MP3, AIFF, etc.)"
                        }
                    },
                    "required": ["audio_file"]
                }
            ),
            Tool(
                name="detect_key",
                description="Detect musical key and mode from an audio file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to audio file"
                        }
                    },
                    "required": ["audio_file"]
                }
            ),
            Tool(
                name="analyze_audio_feel",
                description="Analyze groove feel and energy characteristics from audio.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to audio file"
                        }
                    },
                    "required": ["audio_file"]
                }
            ),
            Tool(
                name="extract_chords",
                description="Extract chord progression from audio file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to audio file"
                        },
                        "window_size": {
                            "type": "number",
                            "description": "Chord detection window in seconds",
                            "default": 0.5
                        }
                    },
                    "required": ["audio_file"]
                }
            ),
            Tool(
                name="detect_scale",
                description="Detect scales/modes from audio file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to audio file"
                        }
                    },
                    "required": ["audio_file"]
                }
            ),
            Tool(
                name="analyze_theory",
                description="Complete music theory analysis (scales, modes, harmonic complexity).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_file": {
                            "type": "string",
                            "description": "Path to audio file"
                        }
                    },
                    "required": ["audio_file"]
                }
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        try:
            audio_file = arguments.get("audio_file", "")
            
            if not Path(audio_file).exists():
                return [TextContent(type="text", text=f"Error: Audio file not found: {audio_file}")]
            
            if name == "detect_bpm":
                if AUDIO_AVAILABLE:
                    try:
                        analyzer = AudioAnalyzer()
                        analysis = analyzer.analyze_file(
                            audio_file,
                            detect_key=False,
                            detect_bpm=True,
                            extract_features_flag=False,
                            analyze_segments=False,
                        )
                        bpm_result = analysis.bpm_result
                        result = {
                            "audio_file": audio_file,
                            "bpm": bpm_result.bpm if bpm_result else None,
                            "confidence": bpm_result.confidence if bpm_result else 0.0,
                            "alternatives": bpm_result.tempo_alternatives[:3] if bpm_result else [],
                        }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "BPM detection requires librosa. Install with: pip install librosa",
                        }
                elif AUDIO_CATALOGER_AVAILABLE:
                    try:
                        bpm, _ = detect_bpm_key(audio_file)
                        # Convert string confidence to float for consistency (0.0-1.0)
                        confidence_val = 0.8 if bpm else 0.3
                        result = {
                            "audio_file": audio_file,
                            "bpm": float(bpm) if bpm else None,
                            "confidence": confidence_val,
                        }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "BPM detection requires librosa. Install with: pip install librosa",
                        }
                else:
                    result = {
                        "audio_file": audio_file,
                        "error": "Audio analysis modules not available",
                        "note": "Install librosa and optional audio_cataloger helpers",
                    }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "detect_key":
                if AUDIO_AVAILABLE:
                    try:
                        analyzer = AudioAnalyzer()
                        analysis = analyzer.analyze_file(
                            audio_file,
                            detect_key=True,
                            detect_bpm=False,
                            extract_features_flag=False,
                            analyze_segments=False,
                        )
                        key_result = analysis.key_result
                        result = {
                            "audio_file": audio_file,
                            "key": key_result.key if key_result else None,
                            "mode": key_result.mode.value if key_result else None,
                            "confidence": key_result.confidence if key_result else 0.0,
                        }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "Key detection requires librosa. Install with: pip install librosa",
                        }
                elif AUDIO_CATALOGER_AVAILABLE:
                    try:
                        _, key_info = detect_bpm_key(audio_file)
                        if key_info:
                            key, mode = key_info.split() if " " in key_info else (key_info, "major")
                            # Convert string confidence to float for consistency (0.0-1.0)
                            result = {
                                "audio_file": audio_file,
                                "key": key,
                                "mode": mode,
                                "confidence": 0.8,
                            }
                        else:
                            result = {
                                "audio_file": audio_file,
                                "key": None,
                                "mode": None,
                                "confidence": 0.3,
                            }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "Key detection requires librosa. Install with: pip install librosa",
                        }
                else:
                    result = {
                        "audio_file": audio_file,
                        "error": "Audio analysis modules not available",
                    }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "analyze_audio_feel":
                if AUDIO_AVAILABLE:
                    try:
                        features = analyze_feel(audio_file)
                        result = {
                            "audio_file": audio_file,
                            "tempo_bpm": features.tempo_bpm,
                            "energy_curve": features.energy_curve[:10] if len(features.energy_curve) > 10 else features.energy_curve,
                            "dynamic_range_db": features.dynamic_range_db,
                            "rms_mean": features.rms_mean
                        }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "Audio analysis requires librosa. Install with: pip install librosa"
                        }
                else:
                    result = {
                        "audio_file": audio_file,
                        "error": "Audio analysis module not available",
                        "note": "Install librosa and ensure music_brain.audio is accessible"
                    }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "extract_chords":
                if AUDIO_AVAILABLE:
                    try:
                        window_size = arguments.get("window_size", 0.5)
                        detection = detect_chords_from_audio(audio_file, window_size=window_size)
                        result = {
                            "audio_file": audio_file,
                            "chords": detection.chord_sequence,
                            "unique_chords": detection.unique_chords,
                            "estimated_key": detection.estimated_key,
                            "confidence": detection.confidence,
                            "chord_count": len(detection.chords)
                        }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "Chord detection requires librosa. Install with: pip install librosa"
                        }
                else:
                    result = {
                        "audio_file": audio_file,
                        "error": "Audio analysis module not available",
                        "note": "Install librosa: pip install librosa"
                    }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "detect_scale":
                if AUDIO_AVAILABLE:
                    try:
                        analyzer = TheoryAnalyzer()
                        analysis = analyzer.analyze_audio(audio_file)
                        
                        scales = []
                        for scale in analysis.detected_scales[:5]:
                            scales.append({
                                "scale": scale.full_name,
                                "confidence": scale.confidence,
                                "is_mode": scale.is_mode,
                                "characteristics": scale.characteristics
                            })
                        
                        result = {
                            "audio_file": audio_file,
                            "primary_scale": analysis.primary_scale.full_name if analysis.primary_scale else None,
                            "mode": analysis.mode,
                            "detected_scales": scales
                        }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "Scale detection requires librosa. Install with: pip install librosa"
                        }
                else:
                    result = {
                        "audio_file": audio_file,
                        "error": "Audio analysis module not available"
                    }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "analyze_theory":
                if AUDIO_AVAILABLE:
                    try:
                        analyzer = TheoryAnalyzer()
                        analysis = analyzer.analyze_audio(audio_file)
                        
                        result = {
                            "audio_file": audio_file,
                            "key_center": analysis.key_center,
                            "mode": analysis.mode,
                            "harmonic_complexity": analysis.harmonic_complexity,
                            "primary_scale": analysis.primary_scale.to_dict() if analysis.primary_scale else None,
                            "detected_scales": [s.to_dict() for s in analysis.detected_scales[:3]]
                        }
                    except Exception as e:
                        result = {
                            "audio_file": audio_file,
                            "error": str(e),
                            "note": "Theory analysis requires librosa. Install with: pip install librosa"
                        }
                else:
                    result = {
                        "audio_file": audio_file,
                        "error": "Audio analysis module not available"
                    }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

