"""
DAiW Music Brain - Agent System

LOCAL SYSTEM - No cloud APIs required after initial Ollama setup.

Components:
- DAWProtocol: Abstract interface for multi-DAW support
- DAW Bridges: Ableton, Logic Pro, Reaper, Bitwig
- MusicCrew: AI agents for music production (local Ollama LLM)
- UnifiedHub: Central orchestration (sync)
- AsyncUnifiedHub: Async-first hub with reactive state + WebSocket API

Architecture (v2):
- Multi-DAW: Abstract DAWProtocol with auto-detection
- Reactive State: Observable containers that auto-notify on changes
- Event Bus: Async pub/sub for decoupled communication
- WebSocket API: Real-time bidirectional control from React/external tools

One-Time Setup:
    # Install Ollama (https://ollama.ai)
    curl -fsSL https://ollama.ai/install.sh | sh

    # Pull a model (one-time download)
    ollama pull llama3

    # Start the server (runs locally)
    ollama serve

Then the system runs 100% locally with no internet required.

Usage (Sync - Original):
    from music_brain.agents import UnifiedHub

    with UnifiedHub() as hub:
        hub.connect_daw()
        hub.speak("Hello world")
        hub.play()

Usage (Async - New):
    from music_brain.agents import AsyncUnifiedHub

    async def main():
        async with AsyncUnifiedHub() as hub:
            await hub.connect_daw()
            await hub.speak("Hello world")

            # Subscribe to state changes
            hub.state["voice"].subscribe(lambda o, n: print(f"Voice: {n}"))

            # Events
            @hub.events.on("daw.play")
            async def on_play(event):
                print("DAW started!")

            # WebSocket available at ws://localhost:8765

    asyncio.run(main())

Shutdown Methods:
    All classes support multiple shutdown patterns:

    1. Context manager (recommended):
       with UnifiedHub() as hub:
           hub.play()
       # Automatically stopped

    2. Explicit shutdown:
       hub = UnifiedHub()
       hub.start()
       hub.stop()

    3. Force stop (immediate):
       force_stop_hub()

    4. Global shutdown:
       shutdown_all()

    5. Automatic cleanup:
       - All classes register with atexit
       - MIDI sends CC 123 (all notes off) before closing
       - OSC server threads are joined with timeout
"""

# =============================================================================
# Ableton Bridge - DAW Communication
# =============================================================================

from .ableton_bridge import (
    # Main classes
    AbletonBridge,
    AbletonOSCBridge,
    AbletonMIDIBridge,

    # Configuration
    OSCConfig,
    MIDIConfig,

    # State classes
    TransportState,
    TrackInfo,

    # Voice control
    VoiceCC,
    VOWEL_FORMANTS,

    # Convenience functions
    get_bridge,
    connect_daw,
    disconnect_daw,

    # MCP tools
    get_mcp_tools as get_bridge_mcp_tools,
)

# =============================================================================
# CrewAI Music Agents - Local LLM Agents
# =============================================================================

from .crewai_music_agents import (
    # LLM
    LocalLLM,
    LocalLLMConfig,
    OnnxLLM,
    OnnxLLMConfig,
    LLMBackend,

    # Tools
    Tool,
    ToolManager,

    # Agents
    AgentRole,
    MusicAgent,
    AGENT_ROLES,

    # Crew
    MusicCrew,

    # Pre-defined tasks
    voice_production_task,
    song_production_task,

    # Convenience functions
    get_crew,
    get_llm_status,
    shutdown_crew,
)

# =============================================================================
# Voice Profiles - Customizable Voice Characteristics
# =============================================================================

from .voice_profiles import (
    # Main classes
    VoiceProfileManager,
    VoiceProfile,

    # Enums
    Gender,
    AccentRegion,
    SpeechPattern,

    # Convenience functions
    get_voice_manager,
    apply_voice_profile,
    learn_word,
    list_accents,
    list_speech_patterns,
)

# =============================================================================
# Unified Hub - Central Orchestration
# =============================================================================

from .unified_hub import (
    # Main class
    UnifiedHub,

    # Configuration
    HubConfig,
    SessionConfig,

    # State classes
    VoiceState,
    DAWState,

    # Voice synthesis
    LocalVoiceSynth,

    # Global functions
    get_hub,
    start_hub,
    stop_hub,
    force_stop_hub,
    shutdown_all,

    # MCP tools
    get_hub_mcp_tools,
)

from .ml_pipeline import (
    MLPipeline,
    EmotionFeatures,
    MLInferenceResult,
    DynamicsResult,
    GrooveResult,
    EmotionEmbedding,
    HarmonyResult,
    ModelType as MLModelType,
)

# =============================================================================
# Command History (Undo/Redo)
# =============================================================================

from .command import (
    Command,
    CommandCategory,
    CommandFactory,
    CommandHistory,
    CommandResult,
    HistoryStats,
    CompoundCommand,
)

# =============================================================================
# Health Dashboard & Telemetry
# =============================================================================

from .telemetry import (
    ComponentType,
    HealthChecker,
    HealthDashboard,
    HealthReport,
    HealthStatus,
    LatencyStats,
    ThroughputStats,
)

# =============================================================================
# Async Hub - Event-Driven Architecture (v2)
# =============================================================================

from .async_hub import (
    AsyncUnifiedHub,
    get_async_hub,
    stop_async_hub,
)

# =============================================================================
# Event Bus - Async Pub/Sub
# =============================================================================

from .events import (
    Event,
    EventResult,
    EventHandler,
    EventPriority,
    EventBus,
    EventChannel,
    EventQueue,
)

# =============================================================================
# DAW Protocol - Multi-DAW Abstraction
# =============================================================================

from .daw_protocol import (
    # Types
    DAWType,
    DAWCapabilities,
    # Protocol
    DAWProtocol,
    BaseDAWBridge,
    # Registry & Factory
    DAWRegistry,
    get_daw_bridge,
)

from .daw_bridges import (
    # Bridges
    AbletonDAWBridge,
    LogicProBridge,
    ReaperBridge,
    BitwigBridge,
    # Configs
    AbletonConfig,
    LogicProConfig,
    ReaperConfig,
    BitwigConfig,
)

# =============================================================================
# Reactive State Management
# =============================================================================

from .reactive import (
    # Core classes
    Observable,
    ReactiveState,
    StateAggregator,
    ComputedState,
    BatchContext,

    # Types
    StateCallback,
    AsyncStateCallback,

    # Decorators
    reactive_dataclass,
    observe,
)

# =============================================================================
# WebSocket Real-time API
# =============================================================================

from .websocket_api import (
    HubWebSocketServer,
    WSClient,
    WSMessage,
    MessageType,
    create_websocket_server,
    HAS_WEBSOCKETS,
)

# =============================================================================
# Convenience Aliases
# =============================================================================

# Shutdown aliases
shutdown_tools = shutdown_crew

# Keep lambda-style convenience to preserve truthiness expectations
get_tool_manager = (
    lambda: (get_crew().tools if get_crew() else None)
)  # noqa: E731

# =============================================================================
# Module Info
# =============================================================================

__all__ = [
    # Ableton Bridge
    "AbletonBridge",
    "AbletonOSCBridge",
    "AbletonMIDIBridge",
    "OSCConfig",
    "MIDIConfig",
    "TransportState",
    "TrackInfo",
    "VoiceCC",
    "VOWEL_FORMANTS",
    "get_bridge",
    "connect_daw",
    "disconnect_daw",
    "get_bridge_mcp_tools",

    # Local LLM
    "LocalLLM",
    "LocalLLMConfig",
    "OnnxLLM",
    "OnnxLLMConfig",
    "LLMBackend",

    # Tools
    "Tool",
    "ToolManager",

    # Agents
    "AgentRole",
    "MusicAgent",
    "AGENT_ROLES",
    "MusicCrew",
    "voice_production_task",
    "song_production_task",
    "get_crew",
    "get_llm_status",
    "shutdown_crew",

    # Voice Profiles
    "VoiceProfileManager",
    "VoiceProfile",
    "Gender",
    "AccentRegion",
    "SpeechPattern",
    "get_voice_manager",
    "apply_voice_profile",
    "learn_word",
    "list_accents",
    "list_speech_patterns",

    # Unified Hub (Sync)
    "UnifiedHub",
    "HubConfig",
    "SessionConfig",
    "VoiceState",
    "DAWState",
    "LocalVoiceSynth",
    "get_hub",
    "start_hub",

    # ML Pipeline
    "MLPipeline",
    "EmotionFeatures",
    "MLInferenceResult",
    "DynamicsResult",
    "GrooveResult",
    "EmotionEmbedding",
    "HarmonyResult",
    "MLModelType",
    "stop_hub",
    "force_stop_hub",
    "shutdown_all",
    "get_hub_mcp_tools",

    # Command History (Undo/Redo)
    "Command",
    "CommandCategory",
    "CommandFactory",
    "CommandHistory",
    "CommandResult",
    "HistoryStats",
    "CompoundCommand",

    # Health Dashboard & Telemetry
    "ComponentType",
    "HealthChecker",
    "HealthDashboard",
    "HealthReport",
    "HealthStatus",
    "LatencyStats",
    "ThroughputStats",

    # Async Hub (Event-Driven)
    "AsyncUnifiedHub",
    "get_async_hub",
    "stop_async_hub",

    # Reactive State
    "Observable",
    "BatchContext",
    "ReactiveState",
    "StateAggregator",
    "ComputedState",
    "StateCallback",
    "AsyncStateCallback",
    "reactive_dataclass",
    "observe",

    # Event Bus
    "Event",
    "EventResult",
    "EventHandler",
    "EventPriority",
    "EventBus",
    "EventChannel",
    "EventQueue",

    # WebSocket API
    "HubWebSocketServer",
    "WSMessage",
    "WSClient",
    "MessageType",
    "create_websocket_server",
    "HAS_WEBSOCKETS",

    # DAW Protocol (Multi-DAW)
    "DAWType",
    "DAWCapabilities",
    "DAWProtocol",
    "BaseDAWBridge",
    "DAWRegistry",
    "get_daw_bridge",
    # DAW Bridges
    "AbletonDAWBridge",
    "LogicProBridge",
    "ReaperBridge",
    "BitwigBridge",
    "AbletonConfig",
    "LogicProConfig",
    "ReaperConfig",
    "BitwigConfig",

    # Aliases
    "shutdown_tools",
    "get_tool_manager",
]

__version__ = "1.0.0"
__author__ = "DAiW"


# =============================================================================
# Quick Test
# =============================================================================

def _test() -> None:
    """Quick test of the agent system."""
    print("DAiW Agent System - Quick Test")
    print("=" * 50)
    print("LOCAL SYSTEM - No cloud APIs")
    print()

    # Check LLM
    llm = LocalLLM()
    print(f"Ollama available: {llm.is_available}")
    print(f"WebSocket support: {HAS_WEBSOCKETS}")

    if not llm.is_available:
        print()
        print("To enable AI agents, run:")
        print("  ollama serve")
        print("  ollama pull llama3")

    print()
    print("Available components:")
    print("  - AbletonBridge (OSC/MIDI)")
    print("  - MusicCrew (6 AI agents)")
    print("  - UnifiedHub (sync orchestration)")
    print("  - AsyncUnifiedHub (async + reactive + WebSocket)")
    print()
    print("Sync Usage:")
    print("  from music_brain.agents import start_hub")
    print("  hub = start_hub()")
    print("  hub.connect_daw()")
    print()
    print("Async Usage:")
    print("  from music_brain.agents import AsyncUnifiedHub")
    print("  async with AsyncUnifiedHub() as hub:")
    print("      await hub.connect_daw()")
    print("      # WebSocket at ws://localhost:8765")


if __name__ == "__main__":
    _test()
