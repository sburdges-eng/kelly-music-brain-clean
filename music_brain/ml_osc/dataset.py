"""
OSC Dataset and Encoding for ML Training

Provides dataset loaders and encoding schemes for OSC message sequences.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass

from music_brain.ml_osc.recorder import OSCSequence, OSCMessage


@dataclass
class OSCMessageEncoder:
    """
    Encodes OSC messages into numerical vectors for ML models.
    
    Handles different argument types (int, float, string) and converts
    them into a fixed-size feature vector.
    """
    
    address_vocab: Dict[str, int] = None
    max_args: int = 8  # Maximum number of arguments per message
    
    def __post_init__(self):
        if self.address_vocab is None:
            self.address_vocab = {}
    
    def fit(self, sequences: List[OSCSequence]):
        """
        Build vocabulary from sequences.
        
        Args:
            sequences: List of OSCSequence objects
        """
        addresses = set()
        for seq in sequences:
            for msg in seq.messages:
                addresses.add(msg.address)
        
        # Create address to index mapping
        self.address_vocab = {addr: idx for idx, addr in enumerate(sorted(addresses))}
    
    def encode_message(self, msg: OSCMessage) -> np.ndarray:
        """
        Encode a single OSC message to a feature vector.
        
        Format: [address_id, timestamp, arg1, arg2, ..., argN, padding...]
        
        Args:
            msg: OSCMessage to encode
            
        Returns:
            Feature vector of shape (max_args + 2,)
        """
        # Address one-hot encoding (use index from vocab)
        addr_id = self.address_vocab.get(msg.address, -1)
        
        # Convert args to floats, handle different types
        args_encoded = []
        for arg in msg.args[:self.max_args]:
            if isinstance(arg, (int, float)):
                args_encoded.append(float(arg))
            elif isinstance(arg, str):
                # Simple hash encoding for strings
                args_encoded.append(float(hash(arg) % 1000) / 1000.0)
            elif isinstance(arg, bool):
                args_encoded.append(float(arg))
            else:
                args_encoded.append(0.0)
        
        # Pad to max_args
        while len(args_encoded) < self.max_args:
            args_encoded.append(0.0)
        
        # Combine: [address_id, timestamp, args...]
        feature = np.array([addr_id, msg.timestamp] + args_encoded, dtype=np.float32)
        return feature
    
    def decode_message(self, feature: np.ndarray) -> OSCMessage:
        """
        Decode feature vector back to OSCMessage.
        
        Args:
            feature: Feature vector
            
        Returns:
            Decoded OSCMessage
        """
        addr_id = int(feature[0])
        timestamp = feature[1]
        args = feature[2:].tolist()
        
        # Remove padding zeros
        args = [a for a in args if a != 0.0]
        
        # Reverse lookup address
        address = None
        for addr, idx in self.address_vocab.items():
            if idx == addr_id:
                address = addr
                break
        
        if address is None:
            address = f"/unknown/{addr_id}"
        
        return OSCMessage(address=address, args=args, timestamp=timestamp)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.address_vocab)
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimensionality."""
        return self.max_args + 2  # address_id + timestamp + args


class OSCDataset:
    """
    Dataset for training ML models on OSC sequences.
    
    Loads sequences from JSON files and provides batched access.
    """
    
    def __init__(self, data_dir: Path, encoder: Optional[OSCMessageEncoder] = None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing .json sequence files
            encoder: Optional pre-fitted encoder (will create new one if None)
        """
        self.data_dir = Path(data_dir)
        self.sequences: List[OSCSequence] = []
        self.encoder = encoder
        
        # Load all sequences
        self._load_sequences()
        
        # Fit encoder if not provided
        if self.encoder is None:
            self.encoder = OSCMessageEncoder()
            self.encoder.fit(self.sequences)
    
    def _load_sequences(self):
        """Load all JSON sequences from data directory."""
        if not self.data_dir.exists():
            return
        
        for json_file in self.data_dir.glob("*.json"):
            try:
                seq = OSCSequence.load(json_file)
                self.sequences.append(seq)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
    
    def get_sequence_matrix(self, seq: OSCSequence) -> np.ndarray:
        """
        Convert a sequence to a matrix of features.
        
        Args:
            seq: OSCSequence to convert
            
        Returns:
            Matrix of shape (num_messages, feature_dim)
        """
        features = [self.encoder.encode_message(msg) for msg in seq.messages]
        return np.array(features)
    
    def get_training_pairs(self, seq: OSCSequence, context_length: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate training pairs (context, target) from a sequence.
        
        Args:
            seq: OSCSequence to process
            context_length: Number of previous messages to use as context
            
        Returns:
            List of (context, target) pairs where:
                - context: shape (context_length, feature_dim)
                - target: shape (feature_dim,)
        """
        if len(seq.messages) < context_length + 1:
            return []
        
        features = self.get_sequence_matrix(seq)
        pairs = []
        
        for i in range(context_length, len(features)):
            context = features[i - context_length:i]
            target = features[i]
            pairs.append((context, target))
        
        return pairs
    
    def get_all_training_pairs(self, context_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all training pairs from all sequences.
        
        Args:
            context_length: Number of previous messages to use as context
            
        Returns:
            (X, y) where:
                - X: shape (num_samples, context_length, feature_dim)
                - y: shape (num_samples, feature_dim)
        """
        all_contexts = []
        all_targets = []
        
        for seq in self.sequences:
            pairs = self.get_training_pairs(seq, context_length)
            for context, target in pairs:
                all_contexts.append(context)
                all_targets.append(target)
        
        return np.array(all_contexts), np.array(all_targets)
    
    def __len__(self) -> int:
        """Get number of sequences in dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> OSCSequence:
        """Get sequence by index."""
        return self.sequences[idx]
