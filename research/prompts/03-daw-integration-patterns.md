# Research Prompt: DAW Integration Patterns

RESEARCH DOMAIN: Python/C++ DAW control via OSC/MIDI

CONTEXT:
@music_brain/logic_pro.py
@src/osc/OSCClient.cpp
@src/osc/OSCServer.cpp
@src/midi/MidiIO.cpp

CURRENT IMPLEMENTATION:
- OSC messages to Logic Pro
- MIDI CC for parameter control
- Basic transport control

RESEARCH FOCUS:
Find repos that implement:
1. Bidirectional DAW communication
2. Project state synchronization
3. Plugin parameter automation
4. Timeline/arrangement control
5. Sample/loop triggering

PRIORITY DAWS:
- Logic Pro (primary)
- Ableton Live (secondary)
- Reaper (for scripting examples)

LOOK FOR:
- OSC message libraries (pyliblo, python-osc)
- MIDI routing frameworks
- DAW-specific Python APIs
- C++ OSC implementations (oscpack)

DELIVERABLE:
- Comparison of OSC vs MIDI vs native APIs
- Code examples for each DAW
- Integration architecture recommendations
