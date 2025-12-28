"""
OSC Message Recording and Storage

Records OSC messages with timing information for training ML models.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class OSCMessage:
    """Represents a single OSC message with metadata."""
    
    address: str
    args: List[Any]
    timestamp: float  # Relative time in seconds from session start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "address": self.address,
            "args": list(self.args),
            "timestamp": self.timestamp,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OSCMessage":
        """Create from dictionary."""
        return OSCMessage(
            address=data["address"],
            args=data["args"],
            timestamp=data["timestamp"],
        )


@dataclass
class OSCSequence:
    """A recorded sequence of OSC messages."""
    
    messages: List[OSCMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    
    def add_message(self, address: str, args: List[Any], timestamp: float):
        """Add a message to the sequence."""
        msg = OSCMessage(address=address, args=args, timestamp=timestamp)
        self.messages.append(msg)
        self.duration = max(self.duration, timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "duration": self.duration,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OSCSequence":
        """Create from dictionary."""
        seq = OSCSequence()
        seq.messages = [OSCMessage.from_dict(m) for m in data.get("messages", [])]
        seq.metadata = data.get("metadata", {})
        seq.duration = data.get("duration", 0.0)
        return seq
    
    def save(self, path: Path):
        """Save sequence to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def load(path: Path) -> "OSCSequence":
        """Load sequence from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return OSCSequence.from_dict(data)


class OSCRecorder:
    """
    Records OSC messages from a server for ML training.
    
    Example usage:
        recorder = OSCRecorder(port=8000)
        recorder.start_recording()
        # ... OSC messages arrive ...
        time.sleep(10)
        sequence = recorder.stop_recording()
        sequence.save("my_recording.json")
    """
    
    def __init__(self, port: int = 8000, ip: str = "127.0.0.1"):
        """
        Initialize OSC recorder.
        
        Args:
            port: UDP port to listen on
            ip: IP address to bind to
        """
        self.port = port
        self.ip = ip
        self.recording = False
        self.current_sequence: Optional[OSCSequence] = None
        self.start_time: float = 0.0
        self.messages_buffer: List[tuple] = []
    
    def _handle_message(self, address: str, *args):
        """Internal handler for incoming OSC messages."""
        if self.recording and self.current_sequence is not None:
            timestamp = time.time() - self.start_time
            self.current_sequence.add_message(address, list(args), timestamp)
    
    def start_recording(self, metadata: Optional[Dict[str, Any]] = None) -> OSCSequence:
        """
        Start recording OSC messages.
        
        Args:
            metadata: Optional metadata to attach to the recording
            
        Returns:
            The OSCSequence being recorded
        """
        if self.recording:
            raise RuntimeError("Already recording")
        
        self.current_sequence = OSCSequence(metadata=metadata or {})
        self.start_time = time.time()
        self.recording = True
        
        return self.current_sequence
    
    def record_message(self, address: str, *args):
        """
        Manually record a message (alternative to server-based recording).
        
        Args:
            address: OSC address pattern
            *args: OSC message arguments
        """
        if self.recording and self.current_sequence is not None:
            timestamp = time.time() - self.start_time
            self.current_sequence.add_message(address, list(args), timestamp)
    
    def stop_recording(self) -> OSCSequence:
        """
        Stop recording and return the recorded sequence.
        
        Returns:
            The recorded OSCSequence
        """
        if not self.recording:
            raise RuntimeError("Not currently recording")
        
        self.recording = False
        sequence = self.current_sequence
        self.current_sequence = None
        return sequence
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording
