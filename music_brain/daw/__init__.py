"""
DAW Integration - Multi-DAW integration utilities.

Provides bridges for working with different DAW file formats,
MIDI export/import, DAW-specific features, and timeline markers.

Supported DAWs:
- Logic Pro (AU, MIDI export)
- FL Studio (VST3, pattern-based workflow)
- Pro Tools (AAX, high-resolution timing)
- Reaper (OSC, ReaScript, multi-format)
- Ableton Live (via MIDI/OSC)
- Cubase (via MIDI)
"""

# Logic Pro
from music_brain.daw.logic import (
    LogicProject,
    export_to_logic,
    import_from_logic,
    create_logic_template,
    LOGIC_PPQ,
)

# Mixer Parameters (Emotion -> DAW automation)
from music_brain.daw.mixer_params import (
    MixerParameters,
    EmotionMapper,
    export_to_logic_automation,
    export_mixer_settings,
    describe_mixer_params,
    MIXER_PRESETS,
    SaturationType,
    FilterType,
    ReverbType,
)

# FL Studio
from music_brain.daw.fl_studio import (
    FLProject,
    FLPattern,
    FLPatternType,
    export_to_fl_studio,
    import_from_fl_studio,
    create_fl_template,
    get_fl_vst3_info,
    FL_STUDIO_PPQ,
    FL_STUDIO_PPQ_HD,
)

# Pro Tools
from music_brain.daw.pro_tools import (
    PTSession,
    PTTrack,
    PTTrackType,
    export_to_pro_tools,
    import_from_pro_tools,
    create_pt_template,
    get_aax_plugin_info,
    create_aax_manifest,
    AAXPluginConfig,
    PRO_TOOLS_PPQ,
)

# Reaper
from music_brain.daw.reaper import (
    ReaperProject,
    ReaperTrack,
    ReaperTrackType,
    ReaperOSC,
    ReaperAction,
    export_to_reaper,
    import_from_reaper,
    create_reaper_template,
    generate_reascript_lua,
    get_reaper_plugin_info,
    REAPER_PPQ,
)

# Markers (shared across DAWs)
from music_brain.daw.markers import (
    MarkerEvent,
    EmotionalSection,
    export_markers_midi,
    export_sections_midi,
    merge_markers_with_midi,
    get_standard_structure,
    get_emotional_structure,
)

__all__ = [
    # Logic Pro
    "LogicProject",
    "export_to_logic",
    "import_from_logic",
    "create_logic_template",
    "LOGIC_PPQ",
    # Mixer Parameters
    "MixerParameters",
    "EmotionMapper",
    "export_to_logic_automation",
    "export_mixer_settings",
    "describe_mixer_params",
    "MIXER_PRESETS",
    "SaturationType",
    "FilterType",
    "ReverbType",
    # FL Studio
    "FLProject",
    "FLPattern",
    "FLPatternType",
    "export_to_fl_studio",
    "import_from_fl_studio",
    "create_fl_template",
    "get_fl_vst3_info",
    "FL_STUDIO_PPQ",
    "FL_STUDIO_PPQ_HD",
    # Pro Tools
    "PTSession",
    "PTTrack",
    "PTTrackType",
    "export_to_pro_tools",
    "import_from_pro_tools",
    "create_pt_template",
    "get_aax_plugin_info",
    "create_aax_manifest",
    "AAXPluginConfig",
    "PRO_TOOLS_PPQ",
    # Reaper
    "ReaperProject",
    "ReaperTrack",
    "ReaperTrackType",
    "ReaperOSC",
    "ReaperAction",
    "export_to_reaper",
    "import_from_reaper",
    "create_reaper_template",
    "generate_reascript_lua",
    "get_reaper_plugin_info",
    "REAPER_PPQ",
    # Markers
    "MarkerEvent",
    "EmotionalSection",
    "export_markers_midi",
    "export_sections_midi",
    "merge_markers_with_midi",
    "get_standard_structure",
    "get_emotional_structure",
]


def get_daw_ppq(daw_name: str) -> int:
    """
    Get the default PPQ for a DAW.

    Args:
        daw_name: DAW name (logic, fl_studio, pro_tools, reaper, ableton, cubase)

    Returns:
        PPQ value
    """
    ppq_map = {
        "logic": LOGIC_PPQ,
        "logic_pro": LOGIC_PPQ,
        "fl_studio": FL_STUDIO_PPQ,
        "fl": FL_STUDIO_PPQ,
        "pro_tools": PRO_TOOLS_PPQ,
        "protools": PRO_TOOLS_PPQ,
        "reaper": REAPER_PPQ,
        "ableton": 96,
        "ableton_live": 96,
        "cubase": 480,
    }
    return ppq_map.get(daw_name.lower().replace(" ", "_"), 480)


def export_for_daw(
    midi_path: str,
    daw: str,
    output_path: str = None,
) -> str:
    """
    Export a MIDI file optimized for a specific DAW.

    Args:
        midi_path: Input MIDI file
        daw: Target DAW name
        output_path: Output path (auto-generated if None)

    Returns:
        Path to exported file
    """
    daw_lower = daw.lower().replace(" ", "_")

    if daw_lower in ["logic", "logic_pro"]:
        return export_to_logic(midi_path, output_path)
    elif daw_lower in ["fl", "fl_studio"]:
        return export_to_fl_studio(midi_path, output_path)
    elif daw_lower in ["pro_tools", "protools"]:
        return export_to_pro_tools(midi_path, output_path)
    elif daw_lower == "reaper":
        return export_to_reaper(midi_path, output_path)
    else:
        raise ValueError(f"Unsupported DAW: {daw}")
