# Research Prompt: JUCE Plugin Architecture

RESEARCH DOMAIN: JUCE VST3/AU plugin best practices

CONTEXT:
@iDAW_Core/PluginProcessor.cpp
@src/plugin/plugin_processor.cpp
@src/osc/OSCHub.cpp

MY CURRENT STATE:
- Basic VST3 host/plugin communication
- OSC bridge for DAW control
- Wavetable synth implementation

WHAT I NEED FROM REPOS:
1. How to structure a complex multi-timbral plugin
2. Thread-safe parameter automation
3. Preset management systems
4. MIDI message routing
5. Audio buffer management (no allocations in processBlock)

EXAMPLES TO FIND:
- Professional synth plugins (Surge, Vital, Dexed)
- Effect processors with complex routing
- Multi-output plugins

ANALYZE: 
- Code organization (folder structure)
- Parameter handling patterns
- State save/load mechanisms
- GUI ↔ Audio thread communication

OUTPUT: 
- Architecture diagrams (mermaid)
- Code snippets for improvements
- Refactoring roadmap
