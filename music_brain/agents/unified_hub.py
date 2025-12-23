#!/usr/bin/env python3
"""
Unified Hub - Central Orchestration for DAiW Music Brain

LOCAL SYSTEM - No cloud APIs required after initial Ollama setup.

The UnifiedHub coordinates:
- Voice synthesis (formant, MIDI CC control)
- DAW control (Ableton, Logic Pro, Reaper via protocol abstraction)
- AI agents (local Ollama LLM or ONNX HTTP)
- ML pipeline (real-time emotion recognition via penta_core_native)
- Plugin/extension system for custom components
- Session management (save/load)

One-Time Setup:
    1. Install Ollama: https://ollama.ai
    2. Pull models: ollama pull llama3
    3. Start server: ollama serve

Then everything runs locally.

Usage:
    from music_brain.agents import UnifiedHub

    with UnifiedHub() as hub:
        hub.connect_daw()
        hub.speak("Hello world", vowel="O")
        hub.play()
        response = hub.ask_agent("composer", "Write a sad progression")

        # ML-driven automation
        hub.enable_ml_pipeline()
        hub.set_emotion_driven_dynamics(True)

        # Plugin extensions
        hub.register_plugin("my_analyzer", MyCustomAnalyzer())
"""

import os
import json
import time
import threading
import queue
import atexit
import logging
import asyncio
import functools
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Optional, Dict, List, Any, Callable, Type, Union,
    Coroutine, Awaitable, Set,
)
from enum import Enum

logger = logging.getLogger(__name__)

# Type alias for LLM instances
LLMInstance = Any  # Union[LocalLLM, OnnxLLM] - using Any to avoid import-time issues

from .ableton_bridge import (
    AbletonBridge,
    AbletonOSCBridge,
    AbletonMIDIBridge,
    OSCConfig,
    MIDIConfig,
    TransportState,
    VoiceCC,
    VOWEL_FORMANTS,
)
from .crewai_music_agents import (
    MusicCrew,
    MusicAgent,
    LocalLLM,
    LocalLLMConfig,
    OnnxLLM,
    OnnxLLMConfig,
    ToolManager,
    AGENT_ROLES,
    LLMBackend,
)
from .command import (
    Command,
    CommandCategory,
    CommandFactory,
    CommandHistory,
    CommandResult,
    HistoryStats,
    SetBreathinessCommand,
    SetTempoCommand,
    SetVibratoCommand,
    SetVowelCommand,
    NoteOnCommand,
    NoteOffCommand,
)
from .telemetry import (
    ComponentType,
    HealthDashboard,
    HealthReport,
    HealthStatus,
)
from .voice_profiles import (
    VoiceProfileManager,
    VoiceProfile,
    Gender,
    AccentRegion,
    SpeechPattern,
    get_voice_manager,
)


# =============================================================================
# DAW Protocol Abstraction (Multi-DAW Support)
# =============================================================================

class DAWProtocol(ABC):
    """
    Abstract protocol for DAW communication.
    
    Implement this interface to add support for new DAWs:
    - Logic Pro (via AppleScript + MIDI)
    - Reaper (via ReaScript OSC)
    - Bitwig (via OSC/WebSocket)
    - FL Studio (via FL Remote)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """DAW identifier name."""
        ...
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to DAW."""
        ...
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to DAW."""
        ...
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from DAW."""
        ...
    
    # Transport controls
    @abstractmethod
    def play(self) -> None:
        """Start playback."""
        ...
    
    @abstractmethod
    def stop(self) -> None:
        """Stop playback."""
        ...
    
    @abstractmethod
    def record(self) -> None:
        """Start recording."""
        ...
    
    @abstractmethod
    def set_tempo(self, bpm: float) -> None:
        """Set tempo in BPM."""
        ...
    
    @abstractmethod
    def get_tempo(self) -> float:
        """Get current tempo."""
        ...
    
    # MIDI operations
    @abstractmethod
    def send_note(self, note: int, velocity: int, duration_ms: int, channel: int = 0) -> None:
        """Send a MIDI note."""
        ...
    
    @abstractmethod
    def send_chord(self, notes: List[int], velocity: int, duration_ms: int, channel: int = 0) -> None:
        """Send multiple notes as a chord."""
        ...
    
    @abstractmethod
    def send_cc(self, cc: int, value: int, channel: int = 0) -> None:
        """Send MIDI CC."""
        ...
    
    # Voice synthesis (optional - default no-op)
    def set_vowel(self, vowel: str, channel: int = 0) -> None:
        """Set vowel for formant synthesis."""
        pass
    
    def set_breathiness(self, amount: float, channel: int = 0) -> None:
        """Set breathiness (0-1)."""
        pass
    
    def set_vibrato(self, rate: float, depth: float, channel: int = 0) -> None:
        """Set vibrato parameters."""
        pass


class AbletonDAW(DAWProtocol):
    """Ableton Live implementation using existing AbletonBridge."""
    
    def __init__(self, osc_config: Optional[OSCConfig] = None, midi_config: Optional[MIDIConfig] = None):
        self._bridge = AbletonBridge(osc_config, midi_config)
    
    @property
    def name(self) -> str:
        return "ableton"
    
    @property
    def is_connected(self) -> bool:
        return self._bridge.is_connected
    
    @property
    def bridge(self) -> AbletonBridge:
        """Access underlying bridge for advanced operations."""
        return self._bridge
    
    def connect(self) -> bool:
        return self._bridge.connect()
    
    def disconnect(self) -> None:
        self._bridge.disconnect()
    
    def play(self) -> None:
        self._bridge.play()
    
    def stop(self) -> None:
        self._bridge.stop()
    
    def record(self) -> None:
        self._bridge.record()
    
    def set_tempo(self, bpm: float) -> None:
        self._bridge.set_tempo(bpm)
    
    def get_tempo(self) -> float:
        return self._bridge.transport.tempo
    
    def send_note(self, note: int, velocity: int, duration_ms: int, channel: int = 0) -> None:
        self._bridge.send_note(note, velocity, duration_ms, channel)
    
    def send_chord(self, notes: List[int], velocity: int, duration_ms: int, channel: int = 0) -> None:
        self._bridge.send_chord(notes, velocity, duration_ms, channel)
    
    def send_cc(self, cc: int, value: int, channel: int = 0) -> None:
        self._bridge.send_cc(cc, value, channel)
    
    def set_vowel(self, vowel: str, channel: int = 0) -> None:
        self._bridge.set_vowel(vowel, channel)
    
    def set_breathiness(self, amount: float, channel: int = 0) -> None:
        self._bridge.set_breathiness(amount, channel)
    
    def set_vibrato(self, rate: float, depth: float, channel: int = 0) -> None:
        self._bridge.set_vibrato(rate, depth, channel)


class LogicProDAW(DAWProtocol):
    """
    Logic Pro X implementation stub.
    
    Uses AppleScript for transport + Core MIDI for notes.
    Requires macOS and Logic Pro X running.
    """
    
    def __init__(self):
        self._connected = False
        self._tempo = 120.0
        self._midi_output = None
    
    @property
    def name(self) -> str:
        return "logic_pro"
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> bool:
        try:
            import mido
            # Create virtual MIDI port for Logic Pro
            self._midi_output = mido.open_output("DAiW Logic", virtual=True)
            self._connected = True
            logger.info("Connected to Logic Pro via virtual MIDI")
            return True
        except Exception as e:
            logger.error(f"Logic Pro connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        if self._midi_output:
            self._midi_output.close()
            self._midi_output = None
        self._connected = False
    
    def _run_applescript(self, script: str) -> bool:
        """Execute AppleScript command."""
        import subprocess
        try:
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
            return True
        except Exception as e:
            logger.error(f"AppleScript failed: {e}")
            return False
    
    def play(self) -> None:
        self._run_applescript('tell application "Logic Pro X" to play')
    
    def stop(self) -> None:
        self._run_applescript('tell application "Logic Pro X" to stop')
    
    def record(self) -> None:
        self._run_applescript('tell application "Logic Pro X" to set record enabled to true')
        self._run_applescript('tell application "Logic Pro X" to play')
    
    def set_tempo(self, bpm: float) -> None:
        self._tempo = bpm
        self._run_applescript(f'tell application "Logic Pro X" to set tempo to {bpm}')
    
    def get_tempo(self) -> float:
        return self._tempo
    
    def send_note(self, note: int, velocity: int, duration_ms: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido
            self._midi_output.send(mido.Message('note_on', note=note, velocity=velocity, channel=channel))
            time.sleep(duration_ms / 1000.0)
            self._midi_output.send(mido.Message('note_off', note=note, velocity=0, channel=channel))
    
    def send_chord(self, notes: List[int], velocity: int, duration_ms: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido
            for note in notes:
                self._midi_output.send(mido.Message('note_on', note=note, velocity=velocity, channel=channel))
            time.sleep(duration_ms / 1000.0)
            for note in notes:
                self._midi_output.send(mido.Message('note_off', note=note, velocity=0, channel=channel))
    
    def send_cc(self, cc: int, value: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido
            self._midi_output.send(mido.Message('control_change', control=cc, value=value, channel=channel))


class ReaperDAW(DAWProtocol):
    """
    Reaper implementation stub.
    
    Uses ReaScript OSC API for control.
    Requires Reaper with OSC configured.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self._host = host
        self._port = port
        self._connected = False
        self._tempo = 120.0
        self._client = None
        self._midi_output = None
    
    @property
    def name(self) -> str:
        return "reaper"
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> bool:
        try:
            from pythonosc import udp_client
            import mido
            self._client = udp_client.SimpleUDPClient(self._host, self._port)
            self._midi_output = mido.open_output("DAiW Reaper", virtual=True)
            self._connected = True
            logger.info("Connected to Reaper via OSC")
            return True
        except Exception as e:
            logger.error(f"Reaper connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        self._client = None
        if self._midi_output:
            self._midi_output.close()
            self._midi_output = None
        self._connected = False
    
    def _send_osc(self, address: str, *args) -> None:
        if self._client:
            self._client.send_message(address, list(args) if args else [])
    
    def play(self) -> None:
        self._send_osc("/action/40044")  # Reaper action: Transport: Play
    
    def stop(self) -> None:
        self._send_osc("/action/40667")  # Reaper action: Transport: Stop
    
    def record(self) -> None:
        self._send_osc("/action/1013")  # Reaper action: Transport: Record
    
    def set_tempo(self, bpm: float) -> None:
        self._tempo = bpm
        self._send_osc("/tempo/raw", bpm)
    
    def get_tempo(self) -> float:
        return self._tempo
    
    def send_note(self, note: int, velocity: int, duration_ms: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido
            self._midi_output.send(mido.Message('note_on', note=note, velocity=velocity, channel=channel))
            time.sleep(duration_ms / 1000.0)
            self._midi_output.send(mido.Message('note_off', note=note, velocity=0, channel=channel))
    
    def send_chord(self, notes: List[int], velocity: int, duration_ms: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido
            for note in notes:
                self._midi_output.send(mido.Message('note_on', note=note, velocity=velocity, channel=channel))
            time.sleep(duration_ms / 1000.0)
            for note in notes:
                self._midi_output.send(mido.Message('note_off', note=note, velocity=0, channel=channel))
    
    def send_cc(self, cc: int, value: int, channel: int = 0) -> None:
        if self._midi_output:
            import mido
            self._midi_output.send(mido.Message('control_change', control=cc, value=value, channel=channel))


# DAW Factory
DAW_REGISTRY: Dict[str, Type[DAWProtocol]] = {
    "ableton": AbletonDAW,
    "logic_pro": LogicProDAW,
    "reaper": ReaperDAW,
}


def create_daw(daw_type: str, **kwargs) -> DAWProtocol:
    """Factory function to create DAW instances."""
    if daw_type not in DAW_REGISTRY:
        raise ValueError(f"Unknown DAW type: {daw_type}. Available: {list(DAW_REGISTRY.keys())}")
    return DAW_REGISTRY[daw_type](**kwargs)


def register_daw(name: str, daw_class: Type[DAWProtocol]) -> None:
    """Register a custom DAW implementation."""
    DAW_REGISTRY[name] = daw_class


# =============================================================================
# Plugin/Extension System
# =============================================================================

class PluginType(Enum):
    """Types of plugins that can be registered."""
    ANALYZER = "analyzer"       # Audio/MIDI analysis
    GENERATOR = "generator"     # Content generation (melodies, chords)
    PROCESSOR = "processor"     # Signal/MIDI processing
    VOICE_ENGINE = "voice"      # Voice synthesis engine
    ML_MODEL = "ml_model"       # ML inference model
    CUSTOM = "custom"           # User-defined


@dataclass
class PluginInfo:
    """Metadata about a registered plugin."""
    name: str
    plugin_type: PluginType
    version: str
    description: str
    instance: Any
    enabled: bool = True


class PluginInterface(ABC):
    """Base interface for hub plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin identifier."""
        ...
    
    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """Type of plugin."""
        ...
    
    @property
    def version(self) -> str:
        """Plugin version."""
        return "1.0.0"
    
    @property
    def description(self) -> str:
        """Plugin description."""
        return ""
    
    def on_register(self, hub: "UnifiedHub") -> None:
        """Called when plugin is registered with hub."""
        pass
    
    def on_unregister(self) -> None:
        """Called when plugin is unregistered."""
        pass
    
    def on_session_start(self) -> None:
        """Called when a new session starts."""
        pass
    
    def on_session_end(self) -> None:
        """Called when session ends."""
        pass


class PluginRegistry:
    """Manages plugin registration and lifecycle."""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._hooks: Dict[str, List[Callable]] = {}
    
    def register(self, plugin: PluginInterface, hub: "UnifiedHub") -> bool:
        """Register a plugin."""
        if plugin.name in self._plugins:
            logger.warning(f"Plugin '{plugin.name}' already registered, replacing")
        
        info = PluginInfo(
            name=plugin.name,
            plugin_type=plugin.plugin_type,
            version=plugin.version,
            description=plugin.description,
            instance=plugin,
            enabled=True,
        )
        self._plugins[plugin.name] = info
        
        try:
            plugin.on_register(hub)
            logger.info(f"Registered plugin: {plugin.name} ({plugin.plugin_type.value})")
            return True
        except Exception as e:
            logger.error(f"Plugin registration failed for {plugin.name}: {e}")
            del self._plugins[plugin.name]
            return False
    
    def unregister(self, name: str) -> bool:
        """Unregister a plugin."""
        if name not in self._plugins:
            return False
        
        info = self._plugins[name]
        try:
            info.instance.on_unregister()
        except Exception as e:
            logger.error(f"Plugin cleanup failed for {name}: {e}")
        
        del self._plugins[name]
        logger.info(f"Unregistered plugin: {name}")
        return True
    
    def get(self, name: str) -> Optional[PluginInterface]:
        """Get a plugin by name."""
        info = self._plugins.get(name)
        return info.instance if info and info.enabled else None
    
    def get_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all plugins of a specific type."""
        return [
            info.instance for info in self._plugins.values()
            if info.plugin_type == plugin_type and info.enabled
        ]
    
    def list_plugins(self) -> List[PluginInfo]:
        """List all registered plugins."""
        return list(self._plugins.values())
    
    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        if name in self._plugins:
            self._plugins[name].enabled = True
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """Disable a plugin without unregistering."""
        if name in self._plugins:
            self._plugins[name].enabled = False
            return True
        return False
    
    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a hook callback for an event."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
    
    def trigger_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """Trigger all hooks for an event."""
        results = []
        for callback in self._hooks.get(event, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook error for {event}: {e}")
        return results
    
    def broadcast_session_start(self) -> None:
        """Notify all plugins of session start."""
        for info in self._plugins.values():
            if info.enabled:
                try:
                    info.instance.on_session_start()
                except Exception as e:
                    logger.error(f"Session start hook failed for {info.name}: {e}")
    
    def broadcast_session_end(self) -> None:
        """Notify all plugins of session end."""
        for info in self._plugins.values():
            if info.enabled:
                try:
                    info.instance.on_session_end()
                except Exception as e:
                    logger.error(f"Session end hook failed for {info.name}: {e}")


# =============================================================================
# ML Pipeline Integration
# =============================================================================

@dataclass
class MLPipelineConfig:
    """Configuration for ML pipeline integration."""
    enabled: bool = False
    models_dir: str = "models"
    registry_path: str = "models/registry.json"
    poll_interval_ms: float = 10.0
    emotion_smoothing: float = 0.3  # EMA alpha for emotion smoothing


@dataclass
class EmotionState:
    """Current detected emotional state from ML."""
    valence: float = 0.0       # -1 (negative) to 1 (positive)
    arousal: float = 0.0       # -1 (calm) to 1 (excited)
    dominance: float = 0.0     # -1 (submissive) to 1 (dominant)
    confidence: float = 0.0    # 0-1 confidence score
    label: str = "neutral"     # Discrete emotion label
    timestamp: float = 0.0


class MLPipeline:
    """
    Integration layer for penta_core_native ML inference.
    
    Provides real-time emotion recognition and parameter mapping
    for emotion-driven automation.
    """
    
    # Emotion label mapping from ML output
    EMOTION_LABELS = [
        "neutral", "happy", "sad", "angry", "fearful",
        "surprised", "disgusted", "calm", "anxious", "nostalgic",
    ]
    
    def __init__(self, config: MLPipelineConfig):
        self.config = config
        self._ml_interface = None
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._emotion_state = EmotionState()
        self._emotion_lock = threading.Lock()
        self._callbacks: List[Callable[[EmotionState], None]] = []
        self._request_counter = 0
    
    def start(self) -> bool:
        """Initialize and start the ML pipeline."""
        if self._running:
            return True
        
        try:
            # Try to import penta_core_native
            import penta_core_native as pcn
            
            ml = pcn.ml
            ml_config = ml.MLConfig()
            ml_config.model_directory = self.config.models_dir
            
            self._ml_interface = ml.MLInterface(ml_config)
            
            # Load model registry
            registry_path = Path(self.config.registry_path)
            if registry_path.exists() and self._ml_interface is not None:
                if not self._ml_interface.load_registry(str(registry_path)):
                    logger.warning("Failed to load some models from registry")
            else:
                logger.warning(f"ML registry not found: {registry_path}")
            
            # Start inference thread
            if self._ml_interface is None or not self._ml_interface.start():
                logger.error("Failed to start ML inference thread")
                return False
            
            self._running = True
            
            # Start polling thread
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()
            
            logger.info("ML pipeline started")
            return True
            
        except ImportError:
            logger.warning(
                "penta_core_native not available. ML pipeline disabled. "
                "Build with: cmake --build build --target penta_core_native"
            )
            return False
        except Exception as e:
            logger.error(f"ML pipeline initialization failed: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the ML pipeline."""
        self._running = False
        
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=1.0)
        
        if self._ml_interface:
            self._ml_interface.stop()
            self._ml_interface = None
        
        logger.info("ML pipeline stopped")
    
    def submit_audio_features(self, features: List[float]) -> Optional[int]:
        """
        Submit audio features for emotion recognition.
        
        Args:
            features: 128-dimensional audio feature vector
                     (RMS, ZCR, spectral features, MFCCs, etc.)
        
        Returns:
            Request ID if submitted, None if failed
        """
        if not self._running or not self._ml_interface:
            return None
        
        try:
            import penta_core_native as pcn
            from array import array
            
            # Prepare feature array (pad to 128 if needed)
            feat_array = array("f", features[:128])
            while len(feat_array) < 128:
                feat_array.append(0.0)
            
            request_id = self._request_counter
            self._request_counter += 1
            
            queued, _ = self._ml_interface.submit_features(
                pcn.ml.ModelType.EmotionRecognizer,
                feat_array,
                timestamp=int(time.time() * 1000)
            )
            
            return request_id if queued else None
            
        except Exception as e:
            logger.error(f"Feature submission failed: {e}")
            return None
    
    def get_emotion_state(self) -> EmotionState:
        """Get current emotion state (thread-safe)."""
        with self._emotion_lock:
            return EmotionState(**asdict(self._emotion_state))
    
    def on_emotion_change(self, callback: Callable[[EmotionState], None]) -> None:
        """Register callback for emotion state changes."""
        self._callbacks.append(callback)
    
    def _poll_loop(self) -> None:
        """Background loop polling for inference results."""
        while self._running and self._ml_interface is not None:
            try:
                result = self._ml_interface.poll_result()
                if result is not None:
                    self._process_result(result)
            except Exception as e:
                logger.error(f"ML poll error: {e}")
            
            time.sleep(self.config.poll_interval_ms / 1000.0)
    
    def _process_result(self, result: Dict[str, Any]) -> None:
        """Process inference result and update emotion state."""
        if not result.get("success"):
            return
        
        output = result.get("output_data", [])
        output_size = result.get("output_size", 0)
        
        if output_size < 3:
            return
        
        # Extract VAD (valence-arousal-dominance) from output
        alpha = self.config.emotion_smoothing
        
        with self._emotion_lock:
            # Smooth emotion values using EMA
            self._emotion_state.valence = (
                alpha * output[0] + (1 - alpha) * self._emotion_state.valence
            )
            self._emotion_state.arousal = (
                alpha * output[1] + (1 - alpha) * self._emotion_state.arousal
            )
            self._emotion_state.dominance = (
                alpha * output[2] + (1 - alpha) * self._emotion_state.dominance
            )
            self._emotion_state.confidence = result.get("confidence", 0.5)
            self._emotion_state.timestamp = time.time()
            
            # Determine discrete label from VAD
            self._emotion_state.label = self._vad_to_label(
                self._emotion_state.valence,
                self._emotion_state.arousal,
            )
            
            new_state = EmotionState(**asdict(self._emotion_state))
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.error(f"Emotion callback error: {e}")
    
    def _vad_to_label(self, valence: float, arousal: float) -> str:
        """Convert VAD coordinates to discrete emotion label."""
        # Simple quadrant mapping
        if arousal > 0.3:
            if valence > 0.3:
                return "happy"
            elif valence < -0.3:
                return "angry"
            else:
                return "surprised"
        elif arousal < -0.3:
            if valence > 0.3:
                return "calm"
            elif valence < -0.3:
                return "sad"
            else:
                return "neutral"
        else:
            if valence > 0.3:
                return "nostalgic"
            elif valence < -0.3:
                return "anxious"
            else:
                return "neutral"
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ML pipeline statistics."""
        if not self._ml_interface:
            return {"available": False}
        
        try:
            stats = self._ml_interface.get_stats()
            return {
                "available": True,
                "total_requests": stats.total_requests,
                "completed_requests": stats.completed_requests,
                "failed_requests": stats.failed_requests,
                "avg_latency_ms": stats.avg_latency_ms,
                "max_latency_ms": stats.max_latency_ms,
            }
        except Exception:
            return {"available": False}


# =============================================================================
# Async Event-Driven Architecture
# =============================================================================

class EventType(Enum):
    """Event types for the async event bus."""
    # DAW Events
    DAW_CONNECTED = "daw.connected"
    DAW_DISCONNECTED = "daw.disconnected"
    DAW_TRANSPORT_CHANGE = "daw.transport"
    DAW_TEMPO_CHANGE = "daw.tempo"
    
    # Voice Events
    VOICE_NOTE_ON = "voice.note_on"
    VOICE_NOTE_OFF = "voice.note_off"
    VOICE_SPEAK_START = "voice.speak_start"
    VOICE_SPEAK_END = "voice.speak_end"
    
    # ML Events
    ML_EMOTION_CHANGE = "ml.emotion"
    ML_INFERENCE_COMPLETE = "ml.inference"
    
    # LLM Events
    LLM_RESPONSE = "llm.response"
    LLM_STREAM_CHUNK = "llm.stream_chunk"
    
    # Session Events
    SESSION_START = "session.start"
    SESSION_SAVE = "session.save"
    SESSION_LOAD = "session.load"
    
    # Plugin Events
    PLUGIN_REGISTERED = "plugin.registered"
    PLUGIN_UNREGISTERED = "plugin.unregistered"
    
    # Custom
    CUSTOM = "custom"


@dataclass
class Event:
    """Event object for the async event bus."""
    type: EventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    source: str = "hub"


class AsyncEventBus:
    """
    Async event bus for non-blocking pub/sub communication.
    
    Enables decoupled, event-driven architecture where components
    can subscribe to events without blocking the main thread.
    
    Usage:
        bus = AsyncEventBus()
        
        # Subscribe to events
        async def on_emotion(event):
            print(f"Emotion: {event.data}")
        
        bus.subscribe(EventType.ML_EMOTION_CHANGE, on_emotion)
        
        # Publish events (async)
        await bus.emit(Event(EventType.ML_EMOTION_CHANGE, emotion_state))
        
        # Or sync emit (runs in background)
        bus.emit_sync(Event(EventType.DAW_CONNECTED, True))
    """
    
    def __init__(self, max_workers: int = 4):
        self._subscribers: Dict[EventType, Set[Callable]] = {}
        self._async_subscribers: Dict[EventType, Set[Callable]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._event_queue: asyncio.Queue = None  # Initialized lazily
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        
        # Event history for replay
        self._history: List[Event] = []
        self._history_limit = 100
    
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop
    
    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], Any],
        is_async: bool = False
    ) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event fires
            is_async: True if callback is async (coroutine)
        """
        if is_async:
            if event_type not in self._async_subscribers:
                self._async_subscribers[event_type] = set()
            self._async_subscribers[event_type].add(callback)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = set()
            self._subscribers[event_type].add(callback)
    
    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], Any]
    ) -> bool:
        """Unsubscribe from an event type."""
        removed = False
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)
            removed = True
        if event_type in self._async_subscribers:
            self._async_subscribers[event_type].discard(callback)
            removed = True
        return removed
    
    async def emit(self, event: Event) -> List[Any]:
        """
        Emit an event asynchronously.
        
        All async subscribers are awaited concurrently.
        Sync subscribers are run in thread pool.
        
        Returns:
            List of results from subscribers
        """
        results = []
        
        # Store in history
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history.pop(0)
        
        # Run sync subscribers in executor
        sync_subs = self._subscribers.get(event.type, set())
        for callback in sync_subs:
            try:
                loop = self._ensure_loop()
                result = await loop.run_in_executor(
                    self._executor, 
                    functools.partial(callback, event)
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Event subscriber error ({event.type}): {e}")
        
        # Run async subscribers concurrently
        async_subs = self._async_subscribers.get(event.type, set())
        if async_subs:
            coros = [cb(event) for cb in async_subs]
            async_results = await asyncio.gather(*coros, return_exceptions=True)
            for result in async_results:
                if isinstance(result, Exception):
                    logger.error(f"Async subscriber error ({event.type}): {result}")
                else:
                    results.append(result)
        
        return results
    
    def emit_sync(self, event: Event) -> None:
        """
        Emit an event from synchronous code.
        
        Schedules the event for async processing without blocking.
        """
        # Store in history
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history.pop(0)
        
        # Run sync subscribers immediately in thread pool
        sync_subs = self._subscribers.get(event.type, set())
        for callback in sync_subs:
            self._executor.submit(self._safe_call, callback, event)
        
        # Schedule async subscribers if loop is running
        try:
            loop = asyncio.get_running_loop()
            async_subs = self._async_subscribers.get(event.type, set())
            for callback in async_subs:
                loop.create_task(self._safe_async_call(callback, event))
        except RuntimeError:
            # No running loop - skip async subscribers
            pass
    
    def _safe_call(self, callback: Callable, event: Event) -> None:
        """Safely call a sync callback."""
        try:
            callback(event)
        except Exception as e:
            logger.error(f"Event callback error: {e}")
    
    async def _safe_async_call(self, callback: Callable, event: Event) -> None:
        """Safely call an async callback."""
        try:
            await callback(event)
        except Exception as e:
            logger.error(f"Async event callback error: {e}")
    
    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 10
    ) -> List[Event]:
        """Get recent event history."""
        if event_type:
            filtered = [e for e in self._history if e.type == event_type]
            return filtered[-limit:]
        return self._history[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()
    
    def shutdown(self) -> None:
        """Shutdown the event bus."""
        self._running = False
        self._executor.shutdown(wait=False)
        self._subscribers.clear()
        self._async_subscribers.clear()


class AsyncHubMixin:
    """
    Mixin class providing async methods for UnifiedHub.
    
    Add async capabilities to the hub without breaking existing sync API.
    """
    
    _event_bus: AsyncEventBus
    _async_executor: ThreadPoolExecutor
    
    def _init_async(self) -> None:
        """Initialize async components."""
        self._event_bus = AsyncEventBus()
        self._async_executor = ThreadPoolExecutor(max_workers=4)
    
    @property
    def events(self) -> AsyncEventBus:
        """Access the event bus."""
        return self._event_bus
    
    # =========================================================================
    # Async DAW Operations
    # =========================================================================
    
    async def connect_daw_async(self) -> bool:
        """Async version of connect_daw."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._async_executor,
            self.connect_daw
        )
        await self._event_bus.emit(Event(
            EventType.DAW_CONNECTED,
            data=result,
            source="hub"
        ))
        return result
    
    async def play_async(self) -> None:
        """Async version of play."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._async_executor, self.play)
        await self._event_bus.emit(Event(
            EventType.DAW_TRANSPORT_CHANGE,
            data={"playing": True},
            source="hub"
        ))
    
    async def stop_async(self) -> None:
        """Async version of stop_playback."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._async_executor, self.stop_playback)
        await self._event_bus.emit(Event(
            EventType.DAW_TRANSPORT_CHANGE,
            data={"playing": False},
            source="hub"
        ))
    
    async def set_tempo_async(self, bpm: float) -> None:
        """Async version of set_tempo."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._async_executor,
            functools.partial(self.set_tempo, bpm)
        )
        await self._event_bus.emit(Event(
            EventType.DAW_TEMPO_CHANGE,
            data=bpm,
            source="hub"
        ))
    
    # =========================================================================
    # Async LLM Operations
    # =========================================================================
    
    async def ask_agent_async(
        self,
        role_id: str,
        task: str,
        stream: bool = False
    ) -> Union[str, Coroutine]:
        """
        Async version of ask_agent.
        
        Args:
            role_id: Agent role to ask
            task: Task or question
            stream: If True, yields chunks as they arrive
        
        Returns:
            Agent response (or async generator if streaming)
        """
        loop = asyncio.get_running_loop()
        
        if stream:
            # Return async generator for streaming
            return self._stream_agent_response(role_id, task)
        else:
            # Run synchronous ask in executor
            result = await loop.run_in_executor(
                self._async_executor,
                functools.partial(self.ask_agent, role_id, task)
            )
            await self._event_bus.emit(Event(
                EventType.LLM_RESPONSE,
                data={"role": role_id, "response": result},
                source="hub"
            ))
            return result
    
    async def _stream_agent_response(
        self,
        role_id: str,
        task: str
    ):
        """Stream agent response chunks."""
        # This would integrate with streaming LLM API
        # For now, simulate with full response split into chunks
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            self._async_executor,
            functools.partial(self.ask_agent, role_id, task)
        )
        
        # Yield word by word for demo
        words = response.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            await self._event_bus.emit(Event(
                EventType.LLM_STREAM_CHUNK,
                data={"role": role_id, "chunk": chunk},
                source="hub"
            ))
            yield chunk
            await asyncio.sleep(0.02)  # Small delay between chunks
    
    async def produce_async(self, brief: str) -> Dict[str, str]:
        """Async version of produce."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._async_executor,
            functools.partial(self.produce, brief)
        )
    
    # =========================================================================
    # Async Voice Operations
    # =========================================================================
    
    async def speak_async(
        self,
        text: str,
        vowel: Optional[str] = None,
        rate: int = 175
    ) -> None:
        """Async version of speak."""
        await self._event_bus.emit(Event(
            EventType.VOICE_SPEAK_START,
            data={"text": text},
            source="hub"
        ))
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._async_executor,
            functools.partial(self.speak, text, vowel, rate)
        )
        
        await self._event_bus.emit(Event(
            EventType.VOICE_SPEAK_END,
            data={"text": text},
            source="hub"
        ))
    
    async def note_on_async(
        self,
        pitch: int,
        velocity: int = 100,
        channel: Optional[int] = None
    ) -> None:
        """Async version of note_on."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._async_executor,
            functools.partial(self.note_on, pitch, velocity, channel)
        )
        await self._event_bus.emit(Event(
            EventType.VOICE_NOTE_ON,
            data={"pitch": pitch, "velocity": velocity},
            source="hub"
        ))
    
    # =========================================================================
    # Async Session Operations
    # =========================================================================
    
    async def save_session_async(self, name: Optional[str] = None) -> str:
        """Async version of save_session."""
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(
            self._async_executor,
            functools.partial(self.save_session, name)
        )
        await self._event_bus.emit(Event(
            EventType.SESSION_SAVE,
            data={"path": path},
            source="hub"
        ))
        return path
    
    async def load_session_async(self, filepath: str) -> bool:
        """Async version of load_session."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._async_executor,
            functools.partial(self.load_session, filepath)
        )
        await self._event_bus.emit(Event(
            EventType.SESSION_LOAD,
            data={"path": filepath, "success": result},
            source="hub"
        ))
        return result
    
    # =========================================================================
    # Async Context Manager
    # =========================================================================
    
    async def __aenter__(self):
        """Async context manager entry."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._async_executor, self.start)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._async_executor, self.stop)
        self._event_bus.shutdown()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HubConfig:
    """Configuration for the UnifiedHub."""
    # Paths
    session_dir: str = "~/.daiw/sessions"
    config_dir: str = "~/.daiw/config"

    # DAW (Multi-DAW support)
    daw_type: str = "ableton"            # "ableton", "logic_pro", "reaper"
    osc_host: str = "127.0.0.1"
    osc_send_port: int = 9000
    osc_receive_port: int = 9001
    midi_port: str = "DAiW Voice"

    # LLM
    llm_model: str = "llama3"
    llm_url: str = "http://localhost:11434"
    llm_backend: str = "ollama"          # "ollama" or "onnx_http"
    llm_onnx_url: str = "http://localhost:8008"

    # Voice
    default_voice_channel: int = 0
    default_voice_profile: str = "default"

    # ML Pipeline
    ml_enabled: bool = False
    ml_models_dir: str = "models"
    ml_registry_path: str = "models/registry.json"
    ml_emotion_smoothing: float = 0.3
    emotion_driven_dynamics: bool = False  # Auto-adjust dynamics from ML emotion

    def __post_init__(self):
        self.session_dir = os.path.expanduser(self.session_dir)
        self.config_dir = os.path.expanduser(self.config_dir)


@dataclass
class SessionConfig:
    """Session-specific configuration."""
    name: str = "untitled"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tempo: float = 120.0
    key: str = "C"
    mode: str = "major"
    emotion: str = "neutral"
    voice_profile: str = "default"
    notes: List[str] = field(default_factory=list)


@dataclass
class VoiceState:
    """Current state of voice synthesis."""
    vowel: str = "A"
    formant_shift: float = 0.0
    breathiness: float = 0.0
    vibrato_rate: float = 0.0
    vibrato_depth: float = 0.0
    pitch: int = 60  # MIDI note
    velocity: int = 100
    active: bool = False


@dataclass
class DAWState:
    """Current state of DAW connection."""
    connected: bool = False
    playing: bool = False
    recording: bool = False
    tempo: float = 120.0
    position: float = 0.0


# =============================================================================
# Voice Synthesizer (Local)
# =============================================================================

class LocalVoiceSynth:
    """
    Local voice synthesis using system TTS and MIDI CC for formant control.

    NO CLOUD APIs - Uses macOS 'say' or espeak on Linux.

    Supports:
    - Voice profiles with pitch, accent, speech patterns
    - Learning custom pronunciations
    - Speech impediment simulation
    """

    def __init__(self, midi_bridge: Optional[AbletonMIDIBridge] = None):
        self.midi = midi_bridge
        self._speaking = False
        self._current_note = None
        self._platform = self._detect_platform()
        self._profile_manager = get_voice_manager()
        self._active_profile: Optional[str] = None

        # Initialize preset profiles
        self._profile_manager.create_preset_profiles()

    def _detect_platform(self) -> str:
        import platform
        system = platform.system()
        if system == "Darwin":
            return "macos"
        elif system == "Linux":
            return "linux"
        elif system == "Windows":
            return "windows"
        return "unknown"

    def speak(
        self,
        text: str,
        vowel: Optional[str] = None,
        rate: Optional[int] = None,
        pitch: Optional[int] = None,
        profile: Optional[str] = None
    ) -> bool:
        """
        Speak text using local TTS with voice profile support.

        Args:
            text: Text to speak
            vowel: Optional vowel hint for formant control
            rate: Speech rate (words per minute) - auto from profile if None
            pitch: Base pitch (0-100) - auto from profile if None
            profile: Voice profile name to use

        Returns:
            True if speaking started
        """
        # Apply voice profile if specified or active
        use_profile = profile or self._active_profile
        voice_params = {}

        if use_profile:
            text, voice_params = self._profile_manager.apply_profile(text, use_profile)

        # Get rate and pitch from profile or defaults
        if rate is None:
            rate = int(175 * voice_params.get("rate", 1.0))
        if pitch is None:
            # Map Hz to 0-100 scale (roughly 80-400 Hz)
            base_hz = voice_params.get("pitch", 170)
            pitch = int((base_hz - 80) / 320 * 100)
            pitch = max(0, min(100, pitch))

        if vowel and self.midi:
            self.set_vowel(vowel)

        # Apply voice quality to MIDI if available
        if self.midi and voice_params:
            if "breathiness" in voice_params:
                self.set_breathiness(voice_params["breathiness"])
            if "formant_shift" in voice_params:
                self.set_formant_shift(voice_params["formant_shift"])

        self._speaking = True

        try:
            if self._platform == "macos":
                import subprocess
                # macOS say supports voice selection
                cmd = ["say", "-r", str(rate)]
                subprocess.Popen(
                    cmd + [text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return True
            elif self._platform == "linux":
                import subprocess
                subprocess.Popen(
                    ["espeak", "-s", str(rate), "-p", str(pitch), text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return True
            elif self._platform == "windows":
                import subprocess
                # Use PowerShell with System.Speech for TTS
                # Rate: -10 (slowest) to 10 (fastest), default 0
                # Map rate (words per minute, ~175 default) to -10 to 10 scale
                ps_rate = int((rate - 175) / 25)  # Roughly maps 100-250 wpm to -3 to 3
                ps_rate = max(-10, min(10, ps_rate))

                # Escape text for PowerShell (single quotes, escape existing quotes)
                escaped_text = text.replace("'", "''")

                # PowerShell command using System.Speech.Synthesis
                ps_command = (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$synth.Rate = {ps_rate}; "
                    f"$synth.Speak('{escaped_text}')"
                )

                subprocess.Popen(
                    ["powershell", "-Command", ps_command],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                return True
            else:
                print(f"TTS not implemented for {self._platform}")
                return False
        except Exception as e:
            print(f"TTS error: {e}")
            return False
        finally:
            self._speaking = False

    def set_profile(self, profile_name: str):
        """Set the active voice profile."""
        self._active_profile = profile_name

    def get_profile(self) -> Optional[str]:
        """Get the active voice profile name."""
        return self._active_profile

    def create_profile(
        self,
        name: str,
        gender: str = "neutral",
        base_pitch: Optional[float] = None,
        accent: str = "american_general",
        speech_patterns: Optional[List[str]] = None,
        **kwargs
    ) -> VoiceProfile:
        """
        Create a new voice profile.

        Args:
            name: Profile name
            gender: male/female/neutral/child
            base_pitch: Base pitch in Hz
            accent: Accent region (see list_accents())
            speech_patterns: List of speech patterns (see list_speech_patterns())

        Returns:
            Created VoiceProfile
        """
        gender_enum = Gender(gender) if isinstance(gender, str) else gender
        accent_enum = AccentRegion(accent) if isinstance(accent, str) else accent
        patterns = [
            SpeechPattern(p) if isinstance(p, str) else p
            for p in (speech_patterns or [])
        ]

        return self._profile_manager.create_profile(
            name=name,
            gender=gender_enum,
            base_pitch=base_pitch,
            accent=accent_enum,
            speech_patterns=patterns,
            **kwargs
        )

    def learn_pronunciation(self, word: str, pronunciation: str):
        """Learn a custom pronunciation for the active profile."""
        if self._active_profile:
            self._profile_manager.learn_pronunciation(
                self._active_profile, word, pronunciation
            )

    def learn_phrase(self, phrase: str, replacement: str):
        """Learn a phrase replacement for the active profile."""
        if self._active_profile:
            self._profile_manager.learn_phrase(
                self._active_profile, phrase, replacement
            )

    def list_profiles(self) -> List[str]:
        """List available voice profiles."""
        return self._profile_manager.list_profiles()

    def list_accents(self) -> List[str]:
        """List available accents."""
        return [a.value for a in AccentRegion]

    def list_speech_patterns(self) -> List[str]:
        """List available speech patterns."""
        return [p.value for p in SpeechPattern]

    def set_formant_shift(self, shift: float, channel: int = 0):
        """Set formant shift (-1 to 1)."""
        if self.midi:
            value = int((shift + 1) * 63.5)
            self.midi.send_cc(VoiceCC.FORMANT_SHIFT.value, value, channel)

    def note_on(self, pitch: int, velocity: int = 100, channel: int = 0):
        """Start a note (for vocoder/synth voice)."""
        if self.midi:
            self.midi.send_note_on(pitch, velocity, channel)
            self._current_note = (pitch, channel)

    def note_off(self, pitch: Optional[int] = None, channel: int = 0):
        """Stop a note."""
        if self.midi:
            if pitch is None and self._current_note:
                pitch, channel = self._current_note
            if pitch is not None:
                self.midi.send_note_off(pitch, channel)
            self._current_note = None

    def set_vowel(self, vowel: str, channel: int = 0):
        """Set vowel for formant synthesis."""
        if self.midi:
            self.midi.set_vowel(vowel, channel)

    def set_formants(self, f1: int, f2: int, channel: int = 0):
        """Set formant frequencies directly (via CC)."""
        if self.midi:
            # Map F1 (200-1000 Hz) to CC value
            f1_cc = int((f1 - 200) / 800 * 127)
            # Map F2 (500-3000 Hz) to CC value
            f2_cc = int((f2 - 500) / 2500 * 127)
            self.midi.send_cc(VoiceCC.FORMANT_SHIFT.value, f1_cc, channel)
            # Could add F2 CC if available

    def set_breathiness(self, amount: float, channel: int = 0):
        """Set breathiness (0-1)."""
        if self.midi:
            self.midi.set_breathiness(amount, channel)

    def set_vibrato(self, rate: float, depth: float, channel: int = 0):
        """Set vibrato (0-1 each)."""
        if self.midi:
            self.midi.set_vibrato(rate, depth, channel)

    @property
    def is_speaking(self) -> bool:
        return self._speaking


# =============================================================================
# Unified Hub
# =============================================================================

class UnifiedHub(AsyncHubMixin):
    """
    Central orchestration hub for DAiW Music Brain.

    LOCAL SYSTEM - All processing runs locally:
    - LLM: Ollama or ONNX HTTP (local)
    - Voice: System TTS + MIDI CC
    - DAW: Ableton, Logic Pro, or Reaper via protocol abstraction
    - ML: Real-time emotion recognition via penta_core_native
    - Plugins: Extensible component system
    - Async: Event-driven architecture with non-blocking operations

    Sync Usage:
        with UnifiedHub() as hub:
            hub.connect_daw()
            hub.play()
            response = hub.ask_agent("composer", "Write a grief progression")

    Async Usage:
        async with UnifiedHub() as hub:
            await hub.connect_daw_async()
            await hub.play_async()
            response = await hub.ask_agent_async("composer", "Write a progression")
            
            # Event-driven
            hub.events.subscribe(EventType.ML_EMOTION_CHANGE, on_emotion, is_async=True)
    """

    def __init__(self, config: Optional[HubConfig] = None):
        self.config = config or HubConfig()

        # Initialize async components (from AsyncHubMixin)
        self._init_async()

        # Components
        self._daw: Optional[DAWProtocol] = None
        self._bridge: Optional[AbletonBridge] = None  # Legacy compatibility
        self._voice: Optional[LocalVoiceSynth] = None
        self._crew: Optional[MusicCrew] = None
        self._llm: LLMInstance = None

        # Plugin system
        self._plugins = PluginRegistry()

        # ML Pipeline
        self._ml_pipeline: Optional[MLPipeline] = None
        self._emotion_driven_dynamics = self.config.emotion_driven_dynamics

        # Command History (Undo/Redo)
        self._command_history = CommandHistory(max_size=100)
        self._command_factory: Optional[CommandFactory] = None

        # Health Dashboard & Telemetry
        self._health_dashboard: Optional[HealthDashboard] = None

        # State
        self._session = SessionConfig()
        self._voice_state = VoiceState()
        self._daw_state = DAWState()
        self._running = False

        # Callbacks (legacy - prefer events)
        self._callbacks: Dict[str, List[Callable]] = {}

        # Ensure directories exist
        os.makedirs(self.config.session_dir, exist_ok=True)
        os.makedirs(self.config.config_dir, exist_ok=True)

        # Register cleanup
        atexit.register(self._shutdown)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> "UnifiedHub":
        """Start the hub and all components."""
        self._running = True

        # Initialize LLM (Ollama by default, ONNX HTTP optional)
        backend = LLMBackend.ONNX_HTTP if self.config.llm_backend.lower() in ["onnx", "onnx_http"] else LLMBackend.OLLAMA

        if backend == LLMBackend.ONNX_HTTP:
            self._llm = OnnxLLM(OnnxLLMConfig(base_url=self.config.llm_onnx_url))
        else:
            self._llm = LocalLLM(LocalLLMConfig(
                model=self.config.llm_model,
                base_url=self.config.llm_url
            ))

        # Initialize DAW via protocol abstraction
        if self.config.daw_type == "ableton":
            self._daw = AbletonDAW(
                osc_config=OSCConfig(
                    host=self.config.osc_host,
                    send_port=self.config.osc_send_port,
                    receive_port=self.config.osc_receive_port
                ),
                midi_config=MIDIConfig(
                    output_port=self.config.midi_port,
                    virtual=True
                )
            )
            # Legacy compatibility - expose underlying bridge
            self._bridge = self._daw.bridge
        else:
            # Use factory for other DAW types
            self._daw = create_daw(self.config.daw_type)
            self._bridge = None

        # Initialize voice (requires MIDI bridge for Ableton)
        if isinstance(self._daw, AbletonDAW):
            self._voice = LocalVoiceSynth(self._daw.bridge.midi)
        else:
            self._voice = LocalVoiceSynth(None)

        # Initialize crew
        if backend == LLMBackend.ONNX_HTTP:
            self._crew = MusicCrew(
                llm_backend=backend,
                onnx_config=OnnxLLMConfig(base_url=self.config.llm_onnx_url),
            )
        else:
            self._crew = MusicCrew(LocalLLMConfig(
                model=self.config.llm_model,
                base_url=self.config.llm_url
            ))
        self._crew.setup(self._bridge)

        # Initialize ML pipeline if enabled
        if self.config.ml_enabled:
            self.enable_ml_pipeline()

        # Initialize command factory
        self._command_factory = CommandFactory(self)

        # Initialize health dashboard
        self._health_dashboard = HealthDashboard(self)
        self._health_dashboard.start_monitoring(interval=30.0)

        # Notify plugins of session start
        self._plugins.broadcast_session_start()

        return self

    # =========================================================================
    # Plugin System
    # =========================================================================

    def register_plugin(self, plugin: PluginInterface) -> bool:
        """
        Register a plugin with the hub.

        Args:
            plugin: Plugin instance implementing PluginInterface

        Returns:
            True if registered successfully
        """
        return self._plugins.register(plugin, self)

    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin by name."""
        return self._plugins.unregister(name)

    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[PluginInfo]:
        """List all registered plugins."""
        return self._plugins.list_plugins()

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all plugins of a specific type."""
        return self._plugins.get_by_type(plugin_type)

    @property
    def plugins(self) -> PluginRegistry:
        """Access the plugin registry directly."""
        return self._plugins

    # =========================================================================
    # Command History (Undo/Redo)
    # =========================================================================

    def undo(self) -> Optional[CommandResult]:
        """
        Undo the last command.

        Returns:
            CommandResult if undone, None if nothing to undo
        """
        result = self._command_history.undo()
        if result:
            self._trigger_callback("undo", result)
        return result

    def redo(self) -> Optional[CommandResult]:
        """
        Redo the last undone command.

        Returns:
            CommandResult if redone, None if nothing to redo
        """
        result = self._command_history.redo()
        if result:
            self._trigger_callback("redo", result)
        return result

    @property
    def can_undo(self) -> bool:
        """Check if there are commands to undo."""
        return self._command_history.can_undo

    @property
    def can_redo(self) -> bool:
        """Check if there are commands to redo."""
        return self._command_history.can_redo

    @property
    def command_history(self) -> CommandHistory:
        """Access the command history directly."""
        return self._command_history

    def get_command_stats(self) -> HistoryStats:
        """Get command history statistics."""
        return self._command_history.get_stats()

    def clear_history(self) -> None:
        """Clear all command history."""
        self._command_history.clear()

    # =========================================================================
    # Health Dashboard & Telemetry
    # =========================================================================

    @property
    def health_dashboard(self) -> Optional[HealthDashboard]:
        """Access the health dashboard."""
        return self._health_dashboard

    def get_health_report(self) -> Optional[HealthReport]:
        """
        Get a full health report for all components.

        Returns:
            HealthReport with status of all components
        """
        if self._health_dashboard:
            return self._health_dashboard.get_report()
        return None

    def get_component_health(self, component: str) -> Optional[Dict[str, Any]]:
        """
        Get health status of a specific component.

        Args:
            component: One of "daw", "llm", "ml_pipeline", "voice",
                      "audio", "websocket", "plugin"

        Returns:
            Health details dict or None
        """
        if self._health_dashboard:
            health = self._health_dashboard.get_component_health(component)
            if health:
                return {
                    "status": health.status.value,
                    "message": health.message,
                    "details": health.details,
                }
        return None

    @property
    def is_healthy(self) -> bool:
        """Quick check if the overall system is healthy."""
        if self._health_dashboard:
            return self._health_dashboard.is_healthy
        return self._running

    def record_latency(self, component: str, ms: float) -> None:
        """Record latency for a component (for external tracking)."""
        if self._health_dashboard:
            try:
                ctype = ComponentType(component)
                self._health_dashboard.record_latency(ctype, ms)
            except ValueError:
                pass

    def record_operation(
        self, component: str, success: bool, error: str = ""
    ) -> None:
        """Record operation result for a component."""
        if self._health_dashboard:
            try:
                ctype = ComponentType(component)
                if success:
                    self._health_dashboard.record_success(ctype)
                else:
                    self._health_dashboard.record_failure(ctype, error)
            except ValueError:
                pass

    # =========================================================================
    # ML Pipeline
    # =========================================================================

    def enable_ml_pipeline(self) -> bool:
        """
        Enable the ML pipeline for real-time emotion recognition.

        Returns:
            True if ML pipeline started successfully
        """
        if self._ml_pipeline and self._ml_pipeline.is_running:
            return True

        ml_config = MLPipelineConfig(
            enabled=True,
            models_dir=self.config.ml_models_dir,
            registry_path=self.config.ml_registry_path,
            emotion_smoothing=self.config.ml_emotion_smoothing,
        )
        self._ml_pipeline = MLPipeline(ml_config)

        if self._ml_pipeline.start():
            # Register emotion change handler
            self._ml_pipeline.on_emotion_change(self._on_emotion_change)
            logger.info("ML pipeline enabled")
            return True
        else:
            self._ml_pipeline = None
            return False

    def disable_ml_pipeline(self) -> None:
        """Disable the ML pipeline."""
        if self._ml_pipeline:
            self._ml_pipeline.stop()
            self._ml_pipeline = None
            logger.info("ML pipeline disabled")

    def submit_audio_features(self, features: List[float]) -> Optional[int]:
        """
        Submit audio features for ML emotion analysis.

        Args:
            features: Audio feature vector (128 dimensions recommended)

        Returns:
            Request ID if submitted, None if ML not available
        """
        if self._ml_pipeline:
            return self._ml_pipeline.submit_audio_features(features)
        return None

    def get_emotion_state(self) -> Optional[EmotionState]:
        """Get current detected emotion state."""
        if self._ml_pipeline:
            return self._ml_pipeline.get_emotion_state()
        return None

    def set_emotion_driven_dynamics(self, enabled: bool) -> None:
        """
        Enable/disable emotion-driven dynamics automation.

        When enabled, detected emotions automatically adjust:
        - Voice breathiness (higher arousal = more breathiness)
        - Vibrato depth (higher emotion intensity = more vibrato)
        - CC parameters mapped to emotional dimensions
        """
        self._emotion_driven_dynamics = enabled
        logger.info(f"Emotion-driven dynamics: {'enabled' if enabled else 'disabled'}")

    def _on_emotion_change(self, emotion: EmotionState) -> None:
        """Handle emotion state changes from ML pipeline."""
        self._trigger_callback("emotion", emotion)

        # Apply emotion-driven dynamics if enabled
        if self._emotion_driven_dynamics and self._daw:
            # Map arousal to breathiness (0-1)
            breathiness = (emotion.arousal + 1) / 2 * 0.5  # 0-0.5 range
            self.set_breathiness(breathiness)

            # Map emotion intensity to vibrato depth
            intensity = abs(emotion.valence) + abs(emotion.arousal)
            vibrato_depth = min(intensity / 2, 0.6)  # Cap at 0.6
            vibrato_rate = 0.3 + (emotion.arousal + 1) / 2 * 0.4  # 0.3-0.7
            self.set_vibrato(vibrato_rate, vibrato_depth)

        # Trigger plugin hooks
        self._plugins.trigger_hook("emotion_change", emotion)

    @property
    def ml_available(self) -> bool:
        """Check if ML pipeline is running."""
        return self._ml_pipeline is not None and self._ml_pipeline.is_running

    def get_ml_stats(self) -> Dict[str, Any]:
        """Get ML pipeline statistics."""
        if self._ml_pipeline:
            return self._ml_pipeline.get_stats()
        return {"available": False}

    # =========================================================================
    # DAW Protocol (Multi-DAW)
    # =========================================================================

    @property
    def daw(self) -> Optional[DAWProtocol]:
        """Access the DAW protocol interface."""
        return self._daw

    @property
    def daw_type(self) -> str:
        """Get current DAW type."""
        return self._daw.name if self._daw else "none"

    def switch_daw(self, daw_type: str, **kwargs) -> bool:
        """
        Switch to a different DAW.

        Args:
            daw_type: "ableton", "logic_pro", or "reaper"
            **kwargs: DAW-specific configuration

        Returns:
            True if switched successfully
        """
        # Disconnect current DAW
        if self._daw:
            self._daw.disconnect()

        try:
            self._daw = create_daw(daw_type, **kwargs)

            # Update bridge for legacy compatibility
            if isinstance(self._daw, AbletonDAW):
                self._bridge = self._daw.bridge
            else:
                self._bridge = None

            self.config.daw_type = daw_type
            logger.info(f"Switched to DAW: {daw_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch DAW: {e}")
            return False

    def check_llm_health(self) -> Dict[str, Any]:
        """
        Return current LLM backend, availability, and endpoint URL.
        """
        if not self._crew:
            return {"backend": None, "available": False, "url": None}

        llm = self._crew.llm
        cfg = getattr(llm, "config", None)
        url = getattr(cfg, "base_url", None) if cfg else None

        return {
            "backend": self._crew.llm_backend.value,
            "available": bool(getattr(llm, "is_available", False)),
            "url": url,
        }

    def stop(self):
        """Stop the hub gracefully."""
        self._running = False

        # Notify plugins of session end
        self._plugins.broadcast_session_end()

        # Emit session end event
        self._event_bus.emit_sync(Event(EventType.SESSION_SAVE, source="hub"))

        # Stop health monitoring
        if self._health_dashboard:
            self._health_dashboard.stop_monitoring()
            self._health_dashboard = None

        # Stop any active notes (skip recording to avoid command history changes)
        if self._voice_state.active:
            self.note_off(record=False)

        # Stop ML pipeline
        if self._ml_pipeline:
            self._ml_pipeline.stop()
            self._ml_pipeline = None

        # Shutdown components in order
        if self._crew:
            self._crew.shutdown()
            self._crew = None

        if self._daw:
            self._daw.disconnect()
            self._daw = None
            self._bridge = None

        # Shutdown async components
        self._event_bus.shutdown()
        self._async_executor.shutdown(wait=False)

        self._voice = None
        self._command_factory = None
        self._callbacks.clear()

    def force_stop(self):
        """Force immediate stop."""
        self._running = False

        # Kill all notes
        if isinstance(self._daw, AbletonDAW) and self._daw.bridge.midi:
            self._daw.bridge.midi.all_notes_off()

        self.stop()

    def _shutdown(self):
        """Atexit shutdown handler."""
        if self._running:
            self.force_stop()

    @property
    def is_running(self) -> bool:
        return self._running

    # =========================================================================
    # DAW Control (via Protocol Abstraction)
    # =========================================================================

    def connect_daw(self) -> bool:
        """Connect to configured DAW."""
        if self._daw:
            success = self._daw.connect()
            self._daw_state.connected = success
            self._trigger_callback("daw_connected", success)
            logger.info(f"Connected to {self._daw.name}: {success}")
            return success
        return False

    def disconnect_daw(self):
        """Disconnect from DAW."""
        if self._daw:
            self._daw.disconnect()
            self._daw_state.connected = False

    def play(self):
        """Start DAW playback."""
        if self._daw:
            self._daw.play()
            self._daw_state.playing = True

    def stop_playback(self):
        """Stop DAW playback."""
        if self._daw:
            self._daw.stop()
            self._daw_state.playing = False

    def record(self):
        """Start DAW recording."""
        if self._daw:
            self._daw.record()
            self._daw_state.recording = True

    def set_tempo(self, bpm: float, record: bool = True):
        """Set DAW tempo."""
        if record and self._command_factory:
            cmd = SetTempoCommand(self, bpm)
            self._command_history.execute(cmd)
        elif self._daw:
            self._daw.set_tempo(bpm)
            self._daw_state.tempo = bpm
            self._session.tempo = bpm

    def send_note(self, note: int, velocity: int = 100, duration_ms: int = 500):
        """Send a MIDI note to DAW."""
        if self._daw:
            self._daw.send_note(note, velocity, duration_ms)

    def send_chord(self, notes: List[int], velocity: int = 100, duration_ms: int = 500):
        """Send a chord to DAW."""
        if self._daw:
            self._daw.send_chord(notes, velocity, duration_ms)

    # =========================================================================
    # Voice Control
    # =========================================================================

    def speak(self, text: str, vowel: Optional[str] = None, rate: int = 175):
        """Speak text using local TTS."""
        if self._voice:
            self._voice.speak(text, vowel, rate)

    def note_on(
        self,
        pitch: int,
        velocity: int = 100,
        channel: Optional[int] = None,
        record: bool = True,
    ):
        """Start a voice note."""
        ch = channel if channel is not None else self.config.default_voice_channel
        if record and self._command_factory:
            cmd = NoteOnCommand(self, pitch, velocity, ch)
            self._command_history.execute(cmd)
        elif self._voice:
            self._voice.note_on(pitch, velocity, ch)
            self._voice_state.pitch = pitch
            self._voice_state.velocity = velocity
            self._voice_state.active = True

    def note_off(
        self,
        pitch: Optional[int] = None,
        channel: Optional[int] = None,
        record: bool = True,
    ):
        """Stop a voice note."""
        ch = channel if channel is not None else self.config.default_voice_channel
        if record and self._command_factory:
            cmd = NoteOffCommand(self, pitch, ch)
            self._command_history.execute(cmd)
        elif self._voice:
            self._voice.note_off(pitch, ch)
            self._voice_state.active = False

    def set_vowel(
        self,
        vowel: str,
        channel: Optional[int] = None,
        record: bool = True,
    ):
        """Set voice vowel (A, E, I, O, U)."""
        ch = channel if channel is not None else self.config.default_voice_channel
        if record and self._command_factory:
            cmd = SetVowelCommand(self, vowel, ch)
            self._command_history.execute(cmd)
        elif self._voice:
            self._voice.set_vowel(vowel, ch)
            self._voice_state.vowel = vowel.upper()

    def set_breathiness(
        self,
        amount: float,
        channel: Optional[int] = None,
        record: bool = True,
    ):
        """Set voice breathiness (0-1)."""
        ch = channel if channel is not None else self.config.default_voice_channel
        if record and self._command_factory:
            cmd = SetBreathinessCommand(self, amount, ch)
            self._command_history.execute(cmd)
        elif self._voice:
            self._voice.set_breathiness(amount, ch)
            self._voice_state.breathiness = amount

    def set_vibrato(
        self,
        rate: float,
        depth: float,
        channel: Optional[int] = None,
        record: bool = True,
    ):
        """Set voice vibrato."""
        ch = channel if channel is not None else self.config.default_voice_channel
        if record and self._command_factory:
            cmd = SetVibratoCommand(self, rate, depth, ch)
            self._command_history.execute(cmd)
        elif self._voice:
            self._voice.set_vibrato(rate, depth, ch)
            self._voice_state.vibrato_rate = rate
            self._voice_state.vibrato_depth = depth

    def sing_vowel_sequence(
        self,
        vowels: List[str],
        pitch: int = 60,
        duration_ms: int = 300,
        channel: Optional[int] = None
    ):
        """Sing a sequence of vowels on a single pitch."""
        ch = channel if channel is not None else self.config.default_voice_channel

        def sequence():
            self.note_on(pitch, 100, ch)
            for vowel in vowels:
                self.set_vowel(vowel, ch)
                time.sleep(duration_ms / 1000)
            self.note_off(pitch, ch)

        threading.Thread(target=sequence, daemon=True).start()

    # =========================================================================
    # AI Agents
    # =========================================================================

    def ask_agent(self, role_id: str, task: str) -> str:
        """
        Ask a specific AI agent about a task.

        Args:
            role_id: One of: voice_director, composer, mix_engineer,
                     daw_controller, producer, lyricist
            task: The question or task

        Returns:
            Agent's response
        """
        if self._crew:
            return self._crew.ask(role_id, task)
        return "AI agents not initialized"

    def produce(self, brief: str) -> Dict[str, str]:
        """
        Have the Producer coordinate a production task.

        Args:
            brief: Creative brief

        Returns:
            Dict with responses from relevant agents
        """
        if self._crew:
            return self._crew.produce(brief)
        return {"error": "AI agents not initialized"}

    def analyze_lyrics(self, lyrics: str) -> Dict[str, str]:
        """
        Analyze lyrics for vocal production.

        Returns vowel guide, break points, delivery notes.
        """
        results = {}

        if self._crew:
            # Lyricist analysis
            results["syllables"] = self._crew.ask(
                "lyricist",
                f"Analyze syllable stress and vowel sounds:\n{lyrics}"
            )

            # Voice Director guidance
            results["vocal_guidance"] = self._crew.ask(
                "voice_director",
                f"Provide vowel modification and break point guidance:\n{lyrics}"
            )

        return results

    def suggest_progression(self, emotion: str, key: str = "C") -> str:
        """
        Suggest a chord progression for an emotion.

        Args:
            emotion: Target emotion (grief, anxiety, joy, etc.)
            key: Musical key

        Returns:
            Suggested progression with explanation
        """
        if self._crew:
            return self._crew.ask(
                "composer",
                f"Suggest a chord progression in {key} for the emotion: {emotion}\n"
                f"Include modal interchange if appropriate and explain the emotional effect."
            )
        return "AI agents not initialized"

    @property
    def llm_available(self) -> bool:
        """Check if local LLM is available."""
        return self._llm is not None and self._llm.is_available

    # =========================================================================
    # Session Management
    # =========================================================================

    def new_session(self, name: str = "untitled"):
        """Create a new session."""
        self._session = SessionConfig(name=name)
        self._voice_state = VoiceState()
        self._trigger_callback("session_new", name)

    def save_session(self, name: Optional[str] = None) -> str:
        """
        Save current session to file.

        Returns:
            Path to saved session file
        """
        if name:
            self._session.name = name

        self._session.updated_at = datetime.now().isoformat()

        # Build session data
        data = {
            "session": asdict(self._session),
            "voice_state": asdict(self._voice_state),
            "daw_state": asdict(self._daw_state),
        }

        # Save to file
        filename = f"{self._session.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.config.session_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self._trigger_callback("session_saved", filepath)
        return filepath

    def load_session(self, filepath: str) -> bool:
        """
        Load a session from file.

        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Restore session
            self._session = SessionConfig(**data.get("session", {}))

            # Restore voice state
            vs_data = data.get("voice_state", {})
            self._voice_state = VoiceState(**vs_data)

            # Apply voice state
            if self._voice:
                self.set_vowel(self._voice_state.vowel)
                self.set_breathiness(self._voice_state.breathiness)
                self.set_vibrato(
                    self._voice_state.vibrato_rate,
                    self._voice_state.vibrato_depth
                )

            # Apply DAW state
            if self._bridge and self._daw_state.connected:
                self.set_tempo(self._session.tempo)

            self._trigger_callback("session_loaded", filepath)
            return True

        except Exception as e:
            print(f"Error loading session: {e}")
            return False

    def list_sessions(self) -> List[str]:
        """List available session files."""
        session_dir = Path(self.config.session_dir)
        return [f.name for f in session_dir.glob("*.json")]

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on(self, event: str, callback: Callable):
        """Register a callback for an event."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def off(self, event: str, callback: Optional[Callable] = None):
        """Remove callback(s)."""
        if callback:
            self._callbacks.get(event, []).remove(callback)
        else:
            self._callbacks.pop(event, None)

    def _trigger_callback(self, event: str, data: Any = None):
        """Trigger callbacks for an event."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(data)
            except Exception as e:
                print(f"Callback error for {event}: {e}")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def session(self) -> SessionConfig:
        return self._session

    @property
    def voice_state(self) -> VoiceState:
        return self._voice_state

    @property
    def daw_state(self) -> DAWState:
        return self._daw_state

    @property
    def daw_connected(self) -> bool:
        return self._daw_state.connected

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __del__(self):
        self._shutdown()


# =============================================================================
# Global Instance Management
# =============================================================================

_default_hub: Optional[UnifiedHub] = None


def get_hub() -> UnifiedHub:
    """Get or create the default hub."""
    global _default_hub
    if _default_hub is None:
        _default_hub = UnifiedHub()
        _default_hub.start()
    return _default_hub


def start_hub() -> UnifiedHub:
    """Start and return the default hub."""
    return get_hub()


def stop_hub():
    """Stop the default hub gracefully."""
    global _default_hub
    if _default_hub:
        _default_hub.stop()
        _default_hub = None


def force_stop_hub():
    """Force stop the default hub immediately."""
    global _default_hub
    if _default_hub:
        _default_hub.force_stop()
        _default_hub = None


def shutdown_all():
    """Complete system shutdown."""
    force_stop_hub()


# =============================================================================
# MCP Tools (for AI access)
# =============================================================================

def get_hub_mcp_tools() -> List[Dict[str, Any]]:
    """Return MCP tool definitions for the hub."""
    return [
        {
            "name": "hub_connect_daw",
            "description": "Connect to Ableton Live",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "hub_play",
            "description": "Start DAW playback",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "hub_stop",
            "description": "Stop DAW playback",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "hub_speak",
            "description": "Speak text using local TTS",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to speak"},
                    "vowel": {"type": "string", "description": "Vowel hint (A/E/I/O/U)"}
                },
                "required": ["text"]
            }
        },
        {
            "name": "hub_ask_agent",
            "description": "Ask an AI agent about a music production task",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["voice_director", "composer", "mix_engineer",
                                 "daw_controller", "producer", "lyricist"],
                        "description": "Agent role to ask"
                    },
                    "task": {"type": "string", "description": "Task or question"}
                },
                "required": ["role", "task"]
            }
        },
        {
            "name": "hub_analyze_lyrics",
            "description": "Analyze lyrics for vocal production",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "lyrics": {"type": "string", "description": "Lyrics to analyze"}
                },
                "required": ["lyrics"]
            }
        },
        {
            "name": "hub_suggest_progression",
            "description": "Suggest chord progression for an emotion",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "emotion": {"type": "string", "description": "Target emotion"},
                    "key": {"type": "string", "description": "Musical key (default: C)"}
                },
                "required": ["emotion"]
            }
        },
    ]


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Testing UnifiedHub (LOCAL - no cloud APIs)")
    print("=" * 60)

    # Configure with optional ML
    config = HubConfig(
        daw_type="ableton",  # or "logic_pro", "reaper"
        ml_enabled=False,    # Set True if penta_core_native built
    )

    with UnifiedHub(config) as hub:
        print(f"\nHub started: {hub.is_running}")
        print(f"LLM available: {hub.llm_available}")
        print(f"ML available: {hub.ml_available}")
        print(f"DAW type: {hub.daw_type}")
        print(f"Registered plugins: {len(hub.list_plugins())}")

        if not hub.llm_available:
            print("\nTo enable AI agents:")
            print("  1. Install Ollama: https://ollama.ai")
            print("  2. Pull model: ollama pull llama3")
            print("  3. Start server: ollama serve")

        # Test DAW connection
        print(f"\nConnecting to {hub.daw_type}...")
        if hub.connect_daw():
            print("DAW connected!")

            # Test voice
            print("\nTesting voice...")
            hub.set_vowel("A")
            hub.speak("Hello, this is the DAiW Music Brain")

            time.sleep(2)

            # Test note
            print("Testing note...")
            hub.note_on(60, 100)
            time.sleep(0.5)
            hub.note_off(60)
        else:
            print(f"DAW not connected ({hub.daw_type} may not be running)")

        # Test AI if available
        if hub.llm_available:
            print("\nTesting AI agent...")
            response = hub.ask_agent(
                "composer",
                "Suggest a 4-chord progression for grief"
            )
            print(f"Composer says:\n{response}")

        # Test ML pipeline if available
        if hub.ml_available:
            print("\nML Pipeline Stats:")
            print(hub.get_ml_stats())

            # Enable emotion-driven dynamics
            hub.set_emotion_driven_dynamics(True)
            print("Emotion-driven dynamics enabled")

        # Save session
        print("\nSaving session...")
        path = hub.save_session("test_session")
        print(f"Saved to: {path}")

    print("\nHub stopped cleanly.")

    # ==========================================================================
    # Async Example
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Testing Async Event-Driven Architecture")
    print("=" * 60)

    async def async_demo():
        """Demonstrate async hub capabilities."""
        
        # Event handler
        events_received = []
        
        async def on_transport_change(event: Event):
            events_received.append(event)
            print(f"  [EVENT] Transport changed: {event.data}")
        
        async def on_emotion_change(event: Event):
            print(f"  [EVENT] Emotion detected: {event.data}")
        
        config = HubConfig(daw_type="ableton", ml_enabled=False)
        
        async with UnifiedHub(config) as hub:
            print(f"\nAsync hub started: {hub.is_running}")
            
            # Subscribe to events
            hub.events.subscribe(
                EventType.DAW_TRANSPORT_CHANGE,
                on_transport_change,
                is_async=True
            )
            hub.events.subscribe(
                EventType.ML_EMOTION_CHANGE,
                on_emotion_change,
                is_async=True
            )
            
            # Connect (async)
            connected = await hub.connect_daw_async()
            print(f"DAW connected (async): {connected}")
            
            if connected:
                # Transport control (async)
                await hub.play_async()
                await asyncio.sleep(0.5)
                await hub.stop_async()
                
                # Tempo change (async)
                await hub.set_tempo_async(100.0)
            
            # Check event history
            print(f"\nEvents received: {len(events_received)}")
            for evt in hub.events.get_history(limit=5):
                print(f"  - {evt.type.value}: {evt.data}")
        
        print("\nAsync hub stopped cleanly.")

    # Run async demo
    try:
        asyncio.run(async_demo())
    except Exception as e:
        print(f"Async demo error: {e}")
