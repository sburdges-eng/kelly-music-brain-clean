"""
ML OSC Learning Module

This module provides machine learning capabilities for learning and generating
OSC (Open Sound Control) message patterns. It enables the system to:
- Record OSC message sequences from users or external controllers
- Learn patterns and relationships in OSC control data
- Generate contextually appropriate OSC sequences
- Predict next likely control messages based on context

The OSC learning system integrates with Kelly's existing Brain API and
audio engine to provide intelligent control mapping and automation.
"""

from music_brain.ml_osc.recorder import OSCRecorder, OSCSequence
from music_brain.ml_osc.dataset import OSCDataset, OSCMessageEncoder
from music_brain.ml_osc.model import OSCPatternLearner, OSCPredictor
from music_brain.ml_osc.generator import OSCSequenceGenerator

__all__ = [
    "OSCRecorder",
    "OSCSequence",
    "OSCDataset",
    "OSCMessageEncoder",
    "OSCPatternLearner",
    "OSCPredictor",
    "OSCSequenceGenerator",
]

__version__ = "0.1.0"
