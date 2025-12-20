#!/usr/bin/env python3
"""
Penta Core Integration Demo
==========================
Demonstrates orchestration of C++ engine, rulebooks, and teacher modules.
"""

import numpy as np
from penta_core import PentaCore
from penta_core.rules import VoiceLeadingRules, HarmonyRules, CounterpointRules, RhythmRules, get_techniques_for_emotion
from penta_core.teachers import RuleBreakingTeacher

# Initialize core engine
core = PentaCore(sample_rate=48000.0)

# Simulate audio and MIDI input
audio = np.random.randn(48000)  # 1 second mono audio
midi_notes = [(60, 100), (64, 100), (67, 100)]  # C major triad

# Process inputs
core.process(audio, midi_notes)
state = core.get_state()
print("Musical State:", state)

# OSC communication
core.start_osc()
core.osc.send_message('/penta/harmony/chord', 60, 'major')
msg = core.osc.receive_message()
print("OSC Message Received:", msg)
core.stop_osc()

# Rulebooks and teacher
teacher = RuleBreakingTeacher()
lesson = teacher.teach_rule("parallel_fifths")
print("Rule-Breaking Lesson:", lesson)

# Emotion techniques
grief_techniques = get_techniques_for_emotion("grief")
print("Techniques for 'grief':", grief_techniques)

# Rhythm genre pocket
dilla_pocket = RhythmRules.get_genre_pocket("dilla")
print("Dilla pocket:", dilla_pocket)

print("Integration demo complete.")
