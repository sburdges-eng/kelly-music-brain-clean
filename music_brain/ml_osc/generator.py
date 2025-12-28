"""
OSC Sequence Generator

Generates new OSC sequences using trained models.
"""

import numpy as np
from typing import List, Optional
from pathlib import Path

from music_brain.ml_osc.recorder import OSCSequence, OSCMessage
from music_brain.ml_osc.dataset import OSCMessageEncoder
from music_brain.ml_osc.model import OSCPredictor


class OSCSequenceGenerator:
    """
    Generates OSC sequences using a trained model.
    
    Example usage:
        generator = OSCSequenceGenerator(predictor, encoder)
        sequence = generator.generate(
            seed_sequence=seed,
            length=100,
            temperature=0.8
        )
    """
    
    def __init__(self, predictor: OSCPredictor, encoder: OSCMessageEncoder):
        """
        Initialize generator.
        
        Args:
            predictor: Trained OSCPredictor model
            encoder: OSCMessageEncoder for encoding/decoding messages
        """
        self.predictor = predictor
        self.encoder = encoder
    
    def generate(
        self,
        seed_sequence: OSCSequence,
        length: int = 50,
        temperature: float = 1.0,
        context_length: int = 5
    ) -> OSCSequence:
        """
        Generate a new OSC sequence.
        
        Args:
            seed_sequence: Initial sequence to start from (must have at least context_length messages)
            length: Number of messages to generate
            temperature: Sampling temperature (higher = more random, lower = more deterministic)
            context_length: Number of previous messages to use as context
            
        Returns:
            Generated OSCSequence
        """
        if len(seed_sequence.messages) < context_length:
            raise ValueError(f"Seed sequence must have at least {context_length} messages")
        
        # Start with seed sequence
        generated = OSCSequence(metadata={"generated": True, "seed_length": len(seed_sequence.messages)})
        
        # Add seed messages
        for msg in seed_sequence.messages:
            generated.add_message(msg.address, msg.args, msg.timestamp)
        
        # Generate new messages
        for _ in range(length):
            # Get context (last context_length messages)
            context_messages = generated.messages[-context_length:]
            context_features = np.array([
                self.encoder.encode_message(msg) for msg in context_messages
            ])
            
            # Predict next message
            predicted_features = self.predictor.predict(context_features)
            
            # Apply temperature to make it more/less random
            if temperature != 1.0:
                predicted_features = predicted_features / temperature
            
            # Add some randomness for variety
            if temperature > 0:
                noise = np.random.normal(0, 0.1 * temperature, predicted_features.shape)
                predicted_features = predicted_features + noise
            
            # Decode to message
            predicted_msg = self.encoder.decode_message(predicted_features)
            
            # Adjust timestamp (add small increment from last message)
            if generated.messages:
                last_time = generated.messages[-1].timestamp
                predicted_msg.timestamp = last_time + 0.1  # 100ms default spacing
            
            # Add to generated sequence
            generated.add_message(
                predicted_msg.address,
                predicted_msg.args,
                predicted_msg.timestamp
            )
        
        return generated
    
    def interpolate(
        self,
        sequence_a: OSCSequence,
        sequence_b: OSCSequence,
        steps: int = 10,
        context_length: int = 5
    ) -> List[OSCSequence]:
        """
        Generate interpolated sequences between two sequences.
        
        Args:
            sequence_a: First sequence
            sequence_b: Second sequence
            steps: Number of interpolation steps
            context_length: Context length for generation
            
        Returns:
            List of interpolated sequences
        """
        # Encode both sequences
        features_a = np.array([self.encoder.encode_message(msg) for msg in sequence_a.messages])
        features_b = np.array([self.encoder.encode_message(msg) for msg in sequence_b.messages])
        
        # Use minimum length
        min_len = min(len(features_a), len(features_b))
        features_a = features_a[:min_len]
        features_b = features_b[:min_len]
        
        # Generate interpolated sequences
        interpolated = []
        for alpha in np.linspace(0, 1, steps):
            # Interpolate features
            interpolated_features = (1 - alpha) * features_a + alpha * features_b
            
            # Decode to sequence
            interp_seq = OSCSequence(metadata={
                "interpolated": True,
                "alpha": float(alpha)
            })
            
            for features in interpolated_features:
                msg = self.encoder.decode_message(features)
                interp_seq.add_message(msg.address, msg.args, msg.timestamp)
            
            interpolated.append(interp_seq)
        
        return interpolated
    
    def vary(
        self,
        sequence: OSCSequence,
        variation_strength: float = 0.2,
        num_variations: int = 5
    ) -> List[OSCSequence]:
        """
        Create variations of a sequence.
        
        Args:
            sequence: Input sequence
            variation_strength: How much to vary (0.0 = no change, 1.0 = very different)
            num_variations: Number of variations to create
            
        Returns:
            List of varied sequences
        """
        variations = []
        
        for _ in range(num_variations):
            varied = OSCSequence(metadata={
                "variation": True,
                "strength": variation_strength
            })
            
            for msg in sequence.messages:
                # Encode and add noise
                features = self.encoder.encode_message(msg)
                noise = np.random.normal(0, variation_strength, features.shape)
                varied_features = features + noise
                
                # Decode
                varied_msg = self.encoder.decode_message(varied_features)
                varied.add_message(
                    varied_msg.address,
                    varied_msg.args,
                    varied_msg.timestamp
                )
            
            variations.append(varied)
        
        return variations
