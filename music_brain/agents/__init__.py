"""
DAiW Music Brain - Agent System

LOCAL SYSTEM - No cloud APIs required after initial Ollama setup.

Components:
- AbletonBridge: OSC/MIDI communication with Ableton Live
- MusicCrew: AI agents for music production (local Ollama LLM)
- UnifiedHub: Central orchestration

One-Time Setup:
    # Install Ollama (https://ollama.ai)
    curl -fsSL https://ollama.ai/install.sh | sh

    # Pull a model (one-time download)
    ollama pull llama3

    # Start the server (runs locally)
    ollama serve

Then the system runs 100% locally with no internet required.

Usage:
    # Quick start
    from music_brain.agents import start_hub, stop_hub

    hub = start_hub()
    hub.connect_daw()
    hub.speak("Hello world")
    hub.play()
    response = hub.ask_agent("composer", "Write a sad progression")
    stop_hub()

    # Or with context manager
    from music_brain.agents import UnifiedHub

    with UnifiedHub() as hub:
        hub.connect_daw()
        hub.send_chord([60, 64, 67], velocity=100, duration_ms=1000)

Shutdown Methods:
    All classes support multiple shutdown patterns:

    1. Context manager (recommended):
       with UnifiedHub() as hub:
           hub.play()
       # Automatically stopped

    2. Explicit shutdown:
       hub = UnifiedHub()
       hub.start()
       # ... use hub ...
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
    VOWEL_FORMANTS,
    # Main classes
    AbletonBridge,
    AbletonMIDIBridge,
    AbletonOSCBridge,
    MIDIConfig,
    # Configuration
    OSCConfig,
    TrackInfo,
    # State classes
    TransportState,
    # Voice control
    VoiceCC,
    connect_daw,
    disconnect_daw,
    # Convenience functions
    get_bridge,
)
from .ableton_bridge import (
    # MCP tools
    get_mcp_tools as get_bridge_mcp_tools,
)

# =============================================================================
# CrewAI Music Agents - Local LLM Agents
# =============================================================================
from .crewai_music_agents import (
    AGENT_ROLES,
    # Agents
    AgentRole,
    LLMBackend,
    # LLM
    LocalLLM,
    LocalLLMConfig,
    MusicAgent,
    # Crew
    MusicCrew,
    # Tools
    Tool,
    ToolManager,
    # Convenience functions
    get_crew,
    shutdown_crew,
    song_production_task,
    # Pre-defined tasks
    voice_production_task,
)

# =============================================================================
# Unified Hub - Central Orchestration
# =============================================================================
from .unified_hub import (
    DAWState,
    # Configuration
    HubConfig,
    # Voice synthesis
    LocalVoiceSynth,
    SessionConfig,
    # Main class
    UnifiedHub,
    # State classes
    VoiceState,
    force_stop_hub,
    # Global functions
    get_hub,
    # MCP tools
    get_hub_mcp_tools,
    shutdown_all,
    start_hub,
    stop_hub,
)

# =============================================================================
# Voice Profiles - Customizable Voice Characteristics
# =============================================================================
from .voice_profiles import (
    AccentRegion,
    # Enums
    Gender,
    SpeechPattern,
    VoiceProfile,
    # Main classes
    VoiceProfileManager,
    apply_voice_profile,
    # Convenience functions
    get_voice_manager,
    learn_word,
    list_accents,
    list_speech_patterns,
)

# =============================================================================
# Convenience Aliases
# =============================================================================

# Shutdown aliases
shutdown_tools = shutdown_crew


def get_tool_manager():
    """Get the tool manager from the crew."""
    crew = get_crew()
    return crew.tools if crew else None

# =============================================================================
# Module Info
# =============================================================================

__all__ = [
    "AGENT_ROLES",
    "VOWEL_FORMANTS",
    # Ableton Bridge
    "AbletonBridge",
    "AbletonMIDIBridge",
    "AbletonOSCBridge",
    "AccentRegion",
    # Agents
    "AgentRole",
    "DAWState",
    "Gender",
    "HubConfig",
    "LLMBackend",
    # Local LLM
    "LocalLLM",
    "LocalLLMConfig",
    "LocalVoiceSynth",
    "MIDIConfig",
    "MusicAgent",
    "MusicCrew",
    "OSCConfig",
    "SessionConfig",
    "SpeechPattern",
    # Tools
    "Tool",
    "ToolManager",
    "TrackInfo",
    "TransportState",
    # Unified Hub
    "UnifiedHub",
    "VoiceCC",
    "VoiceProfile",
    # Voice Profiles
    "VoiceProfileManager",
    "VoiceState",
    "apply_voice_profile",
    "connect_daw",
    "disconnect_daw",
    "force_stop_hub",
    "get_bridge",
    "get_bridge_mcp_tools",
    "get_crew",
    "get_hub",
    "get_hub_mcp_tools",
    "get_tool_manager",
    "get_voice_manager",
    "learn_word",
    "list_accents",
    "list_speech_patterns",
    "shutdown_all",
    "shutdown_crew",
    # Aliases
    "shutdown_tools",
    "song_production_task",
    "start_hub",
    "stop_hub",
    "voice_production_task",
]

__version__ = "1.0.0"
__author__ = "DAiW"


# =============================================================================
# Quick Test
# =============================================================================

def _test():
    """Quick test of the agent system."""
    print("DAiW Agent System - Quick Test")
    print("=" * 50)
    print("LOCAL SYSTEM - No cloud APIs")
    print()

    # Check LLM
    llm = LocalLLM()
    print(f"Ollama available: {llm.is_available}")

    if not llm.is_available:
        print()
        print("To enable AI agents, run:")
        print("  ollama serve")
        print("  ollama pull llama3")

    print()
    print("Available components:")
    print("  - AbletonBridge (OSC/MIDI)")
    print("  - MusicCrew (6 AI agents)")
    print("  - UnifiedHub (orchestration)")
    print()
    print("Usage:")
    print("  from music_brain.agents import start_hub, stop_hub")
    print("  hub = start_hub()")
    print("  hub.connect_daw()")
    print("  stop_hub()")


if __name__ == "__main__":
    _test()
