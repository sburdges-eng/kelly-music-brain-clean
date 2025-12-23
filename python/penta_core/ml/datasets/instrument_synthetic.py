"""
Synthetic Instrument Data Generation for Kelly ML Training.

Generates labeled training data for instrument recognition:
- Instrument family and specific instrument
- Playing technique and articulation
- Expression style and energy level
- Musical role and sentiment

Uses music theory rules to create varied, labeled examples.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Instrument Knowledge Base
# =============================================================================


# Instrument families and their characteristics
INSTRUMENT_FAMILIES = {
    'piano': {
        'specific': ['acoustic_grand', 'acoustic_upright', 'electric_piano', 'honky_tonk', 'clavinet'],
        'techniques': ['sustained', 'staccato', 'legato', 'hammered', 'tremolo'],
        'articulations': ['normal', 'accented', 'tenuto', 'marcato'],
        'registers': [0, 1, 2],  # All registers
        'harmonic_ratio': (0.7, 0.95),
        'attack_range_ms': (5, 50),
        'decay_range_ms': (100, 2000),
        'typical_emotions': ['happy', 'sad', 'contemplative', 'passionate', 'playful'],
    },
    'guitar': {
        'specific': ['acoustic_steel', 'acoustic_nylon', 'electric_clean', 'electric_distorted', 'bass'],
        'techniques': ['strummed', 'picked', 'fingerstyle', 'palm_muted', 'harmonics', 'slapped'],
        'articulations': ['normal', 'accented', 'staccato', 'legato'],
        'registers': [0, 1, 2],
        'harmonic_ratio': (0.6, 0.9),
        'attack_range_ms': (10, 100),
        'decay_range_ms': (200, 3000),
        'typical_emotions': ['happy', 'melancholic', 'aggressive', 'gentle', 'playful'],
    },
    'strings_bowed': {
        'specific': ['violin', 'viola', 'cello', 'double_bass', 'string_section'],
        'techniques': ['bowed', 'legato', 'staccato', 'tremolo', 'pizzicato', 'harmonics'],
        'articulations': ['normal', 'accented', 'con_sordino', 'pizzicato', 'marcato'],
        'registers': [0, 1, 2],
        'harmonic_ratio': (0.75, 0.95),
        'attack_range_ms': (30, 200),
        'decay_range_ms': (500, 5000),
        'typical_emotions': ['sad', 'melancholic', 'passionate', 'triumphant', 'serene'],
    },
    'brass': {
        'specific': ['trumpet', 'trombone', 'french_horn', 'tuba', 'brass_section'],
        'techniques': ['sustained', 'staccato', 'legato', 'flutter_tongue', 'glissando'],
        'articulations': ['normal', 'accented', 'marcato', 'sforzando', 'con_sordino'],
        'registers': [0, 1, 2],
        'harmonic_ratio': (0.7, 0.9),
        'attack_range_ms': (20, 100),
        'decay_range_ms': (200, 2000),
        'typical_emotions': ['triumphant', 'aggressive', 'solemn', 'joyful', 'mysterious'],
    },
    'woodwind': {
        'specific': ['flute', 'clarinet', 'oboe', 'saxophone', 'bassoon'],
        'techniques': ['sustained', 'staccato', 'legato', 'trill', 'flutter_tongue'],
        'articulations': ['normal', 'accented', 'staccato', 'legato'],
        'registers': [1, 2],  # Mostly mid and high
        'harmonic_ratio': (0.65, 0.9),
        'attack_range_ms': (20, 80),
        'decay_range_ms': (100, 1000),
        'typical_emotions': ['playful', 'melancholic', 'gentle', 'mysterious', 'joyful'],
    },
    'synth_lead': {
        'specific': ['saw_lead', 'square_lead', 'supersaw', 'fm_lead', 'wavetable_lead'],
        'techniques': ['sustained', 'staccato', 'legato', 'glissando', 'vibrato'],
        'articulations': ['normal', 'accented', 'staccato'],
        'registers': [1, 2],
        'harmonic_ratio': (0.5, 0.85),
        'attack_range_ms': (1, 50),
        'decay_range_ms': (50, 500),
        'typical_emotions': ['energetic', 'aggressive', 'triumphant', 'anxious', 'playful'],
    },
    'synth_pad': {
        'specific': ['warm_pad', 'bright_pad', 'evolving_pad', 'vocal_pad', 'string_pad'],
        'techniques': ['sustained', 'legato'],
        'articulations': ['normal'],
        'registers': [0, 1, 2],
        'harmonic_ratio': (0.4, 0.8),
        'attack_range_ms': (100, 2000),
        'decay_range_ms': (1000, 10000),
        'typical_emotions': ['serene', 'mysterious', 'contemplative', 'sad', 'peaceful'],
    },
    'drums': {
        'specific': ['kick', 'snare', 'hihat', 'tom', 'cymbal', 'full_kit'],
        'techniques': ['hammered', 'sustained', 'staccato'],
        'articulations': ['normal', 'accented', 'ghost'],
        'registers': [0, 1, 2],
        'harmonic_ratio': (0.1, 0.5),  # Low harmonic content
        'attack_range_ms': (1, 30),
        'decay_range_ms': (30, 500),
        'typical_emotions': ['energetic', 'aggressive', 'playful', 'triumphant'],
    },
    'voice': {
        'specific': ['soprano', 'alto', 'tenor', 'bass', 'choir'],
        'techniques': ['sustained', 'legato', 'vibrato', 'staccato'],
        'articulations': ['normal', 'accented'],
        'registers': [0, 1, 2],
        'harmonic_ratio': (0.7, 0.95),
        'attack_range_ms': (50, 200),
        'decay_range_ms': (200, 2000),
        'typical_emotions': ['passionate', 'melancholic', 'joyful', 'serene', 'triumphant'],
    },
}

# Expression characteristics for each emotion
EXPRESSION_PROFILES = {
    'aggressive': {
        'energy_range': (0.7, 1.0),
        'attack_multiplier': (0.5, 0.8),  # Faster attacks
        'vibrato_depth': (0, 30),
        'dynamics_variance': (0.3, 0.5),
        'brightness': (0.6, 0.9),
        'tension': (0.6, 0.9),
        'valence': (-0.3, 0.1),  # Slightly negative
        'arousal': (0.7, 1.0),
    },
    'gentle': {
        'energy_range': (0.2, 0.5),
        'attack_multiplier': (1.2, 2.0),  # Slower attacks
        'vibrato_depth': (10, 50),
        'dynamics_variance': (0.1, 0.2),
        'brightness': (0.3, 0.6),
        'tension': (0.1, 0.3),
        'valence': (0.2, 0.6),
        'arousal': (0.1, 0.4),
    },
    'melancholic': {
        'energy_range': (0.2, 0.5),
        'attack_multiplier': (1.0, 1.5),
        'vibrato_depth': (20, 60),
        'dynamics_variance': (0.2, 0.3),
        'brightness': (0.2, 0.4),
        'tension': (0.4, 0.6),
        'valence': (-0.8, -0.3),
        'arousal': (0.2, 0.5),
    },
    'joyful': {
        'energy_range': (0.5, 0.8),
        'attack_multiplier': (0.7, 1.0),
        'vibrato_depth': (10, 40),
        'dynamics_variance': (0.2, 0.4),
        'brightness': (0.6, 0.9),
        'tension': (0.2, 0.4),
        'valence': (0.5, 0.9),
        'arousal': (0.5, 0.8),
    },
    'mysterious': {
        'energy_range': (0.3, 0.6),
        'attack_multiplier': (1.0, 1.5),
        'vibrato_depth': (5, 30),
        'dynamics_variance': (0.2, 0.4),
        'brightness': (0.3, 0.5),
        'tension': (0.5, 0.7),
        'valence': (-0.2, 0.2),
        'arousal': (0.3, 0.6),
    },
    'triumphant': {
        'energy_range': (0.7, 1.0),
        'attack_multiplier': (0.6, 0.9),
        'vibrato_depth': (20, 50),
        'dynamics_variance': (0.3, 0.5),
        'brightness': (0.7, 1.0),
        'tension': (0.3, 0.5),
        'valence': (0.6, 1.0),
        'arousal': (0.7, 1.0),
    },
    'playful': {
        'energy_range': (0.5, 0.7),
        'attack_multiplier': (0.7, 1.0),
        'vibrato_depth': (5, 25),
        'dynamics_variance': (0.3, 0.5),
        'brightness': (0.5, 0.8),
        'tension': (0.2, 0.4),
        'valence': (0.4, 0.8),
        'arousal': (0.5, 0.7),
    },
    'solemn': {
        'energy_range': (0.3, 0.5),
        'attack_multiplier': (1.2, 1.8),
        'vibrato_depth': (10, 40),
        'dynamics_variance': (0.1, 0.2),
        'brightness': (0.3, 0.5),
        'tension': (0.4, 0.6),
        'valence': (-0.4, 0.0),
        'arousal': (0.2, 0.4),
    },
    'passionate': {
        'energy_range': (0.6, 0.9),
        'attack_multiplier': (0.8, 1.1),
        'vibrato_depth': (30, 70),
        'dynamics_variance': (0.4, 0.6),
        'brightness': (0.5, 0.7),
        'tension': (0.5, 0.7),
        'valence': (0.1, 0.5),
        'arousal': (0.6, 0.9),
    },
    'contemplative': {
        'energy_range': (0.2, 0.4),
        'attack_multiplier': (1.3, 2.0),
        'vibrato_depth': (10, 30),
        'dynamics_variance': (0.1, 0.2),
        'brightness': (0.4, 0.6),
        'tension': (0.3, 0.5),
        'valence': (-0.1, 0.3),
        'arousal': (0.1, 0.3),
    },
    'anxious': {
        'energy_range': (0.5, 0.8),
        'attack_multiplier': (0.6, 0.9),
        'vibrato_depth': (30, 60),
        'dynamics_variance': (0.4, 0.6),
        'brightness': (0.5, 0.7),
        'tension': (0.7, 0.9),
        'valence': (-0.6, -0.2),
        'arousal': (0.6, 0.9),
    },
    'serene': {
        'energy_range': (0.1, 0.3),
        'attack_multiplier': (1.5, 2.5),
        'vibrato_depth': (5, 20),
        'dynamics_variance': (0.05, 0.15),
        'brightness': (0.4, 0.6),
        'tension': (0.1, 0.2),
        'valence': (0.3, 0.7),
        'arousal': (0.0, 0.2),
    },
}

# Musical roles and their characteristics
MUSICAL_ROLES = {
    'lead_melody': {
        'register_preference': [1, 2],  # Mid to high
        'energy_modifier': 1.1,
        'articulation_variance': 0.3,
    },
    'counter_melody': {
        'register_preference': [1],
        'energy_modifier': 0.8,
        'articulation_variance': 0.2,
    },
    'harmony': {
        'register_preference': [1],
        'energy_modifier': 0.7,
        'articulation_variance': 0.1,
    },
    'bass': {
        'register_preference': [0],
        'energy_modifier': 0.9,
        'articulation_variance': 0.2,
    },
    'rhythm': {
        'register_preference': [0, 1],
        'energy_modifier': 0.8,
        'articulation_variance': 0.4,
    },
    'pad': {
        'register_preference': [0, 1, 2],
        'energy_modifier': 0.5,
        'articulation_variance': 0.05,
    },
    'accent': {
        'register_preference': [1, 2],
        'energy_modifier': 1.2,
        'articulation_variance': 0.5,
    },
    'ambient': {
        'register_preference': [0, 1, 2],
        'energy_modifier': 0.3,
        'articulation_variance': 0.02,
    },
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class InstrumentGeneratorConfig:
    """Configuration for synthetic instrument data generation."""
    
    samples_per_family: int = 500
    samples_per_emotion: int = 200
    
    # Feature ranges
    pitch_range: Tuple[int, int] = (36, 96)  # MIDI notes
    duration_range: Tuple[float, float] = (0.5, 5.0)  # seconds
    
    # Randomization
    random_seed: Optional[int] = None
    add_noise: bool = True
    noise_level: float = 0.05


# =============================================================================
# Synthetic Instrument Generator
# =============================================================================


class InstrumentSyntheticGenerator:
    """
    Generates synthetic labeled data for instrument recognition training.
    
    Creates samples with both technical labels (instrument, technique) and
    emotional labels (expression, energy, sentiment).
    
    Usage:
        generator = InstrumentSyntheticGenerator()
        
        # Generate full dataset
        samples = generator.generate_full_dataset(1000)
        
        # Generate for specific instrument
        piano_samples = generator.generate_for_family('piano', 100)
        
        # Generate for specific emotion
        sad_samples = generator.generate_for_emotion('melancholic', 100)
    """
    
    def __init__(self, config: Optional[InstrumentGeneratorConfig] = None):
        self.config = config or InstrumentGeneratorConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
    
    def generate_full_dataset(
        self,
        total_samples: int,
        output_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a complete balanced dataset."""
        samples = []
        
        # Distribute samples across families and emotions
        families = list(INSTRUMENT_FAMILIES.keys())
        emotions = list(EXPRESSION_PROFILES.keys())
        
        samples_per_combo = total_samples // (len(families) * 2)  # Rough distribution
        
        for family in families:
            family_samples = self.generate_for_family(
                family, 
                samples_per_combo * 2,
            )
            samples.extend(family_samples)
        
        # Ensure emotional balance
        for emotion in emotions:
            emotion_samples = self.generate_for_emotion(
                emotion,
                samples_per_combo,
            )
            samples.extend(emotion_samples)
        
        # Shuffle
        random.shuffle(samples)
        
        # Save if output_dir provided
        if output_dir:
            self._save_samples(samples, output_dir)
        
        logger.info(f"Generated {len(samples)} synthetic instrument samples")
        return samples
    
    def generate_for_family(
        self,
        family: str,
        num_samples: int,
    ) -> List[Dict[str, Any]]:
        """Generate samples for a specific instrument family."""
        if family not in INSTRUMENT_FAMILIES:
            logger.warning(f"Unknown instrument family: {family}")
            return []
        
        family_info = INSTRUMENT_FAMILIES[family]
        samples = []
        
        for i in range(num_samples):
            # Choose specific instrument and characteristics
            specific = random.choice(family_info['specific'])
            technique = random.choice(family_info['techniques'])
            articulation = random.choice(family_info['articulations'])
            register = random.choice(family_info['registers'])
            
            # Choose emotion (biased toward typical emotions for this instrument)
            if random.random() < 0.7:
                emotion = random.choice(family_info['typical_emotions'])
            else:
                emotion = random.choice(list(EXPRESSION_PROFILES.keys()))
            
            # Choose musical role
            role = random.choice(list(MUSICAL_ROLES.keys()))
            
            # Generate the sample
            sample = self._generate_sample(
                family=family,
                specific=specific,
                technique=technique,
                articulation=articulation,
                register=register,
                emotion=emotion,
                role=role,
                family_info=family_info,
                index=i,
            )
            samples.append(sample)
        
        return samples
    
    def generate_for_emotion(
        self,
        emotion: str,
        num_samples: int,
    ) -> List[Dict[str, Any]]:
        """Generate samples for a specific emotion across instruments."""
        if emotion not in EXPRESSION_PROFILES:
            logger.warning(f"Unknown emotion: {emotion}")
            return []
        
        samples = []
        families = list(INSTRUMENT_FAMILIES.keys())
        
        for i in range(num_samples):
            # Choose random family
            family = random.choice(families)
            family_info = INSTRUMENT_FAMILIES[family]
            
            specific = random.choice(family_info['specific'])
            technique = random.choice(family_info['techniques'])
            articulation = random.choice(family_info['articulations'])
            register = random.choice(family_info['registers'])
            role = random.choice(list(MUSICAL_ROLES.keys()))
            
            sample = self._generate_sample(
                family=family,
                specific=specific,
                technique=technique,
                articulation=articulation,
                register=register,
                emotion=emotion,
                role=role,
                family_info=family_info,
                index=i,
            )
            samples.append(sample)
        
        return samples
    
    def _generate_sample(
        self,
        family: str,
        specific: str,
        technique: str,
        articulation: str,
        register: int,
        emotion: str,
        role: str,
        family_info: Dict,
        index: int,
    ) -> Dict[str, Any]:
        """Generate a single synthetic sample with all labels."""
        expr_profile = EXPRESSION_PROFILES[emotion]
        role_info = MUSICAL_ROLES[role]
        
        # Generate technical features
        harmonic_ratio = random.uniform(*family_info['harmonic_ratio'])
        base_attack = random.uniform(*family_info['attack_range_ms'])
        base_decay = random.uniform(*family_info['decay_range_ms'])
        
        # Modify by expression
        attack_mult = random.uniform(*expr_profile['attack_multiplier'])
        attack_ms = base_attack * attack_mult
        
        # Generate pitch based on register
        pitch_ranges = [
            (36, 55),   # Bass
            (55, 75),   # Mid
            (75, 96),   # Treble
        ]
        pitch_range = pitch_ranges[register]
        pitch_mean = random.uniform(*pitch_range)
        pitch_std = random.uniform(2, 12)
        
        # Generate vibrato
        vibrato_depth = random.uniform(*expr_profile['vibrato_depth'])
        vibrato_rate = random.uniform(4, 8) if vibrato_depth > 10 else 0
        
        # Generate dynamics
        energy = random.uniform(*expr_profile['energy_range'])
        energy *= role_info['energy_modifier']
        dynamics_variance = random.uniform(*expr_profile['dynamics_variance'])
        dynamics_variance *= (1 + role_info['articulation_variance'])
        
        # Generate sentiment
        brightness = random.uniform(*expr_profile['brightness'])
        tension = random.uniform(*expr_profile['tension'])
        valence = random.uniform(*expr_profile['valence'])
        arousal = random.uniform(*expr_profile['arousal'])
        
        # Determine legato/staccato tendency
        if technique in ['legato', 'sustained', 'bowed']:
            legato_score = random.uniform(0.6, 1.0)
        elif technique in ['staccato', 'picked', 'hammered']:
            legato_score = random.uniform(0.0, 0.4)
        else:
            legato_score = random.uniform(0.3, 0.7)
        staccato_score = 1.0 - legato_score
        
        # Humanization
        timing_human = random.uniform(5, 30) if emotion != 'aggressive' else random.uniform(2, 15)
        velocity_human = dynamics_variance * random.uniform(0.5, 1.5)
        
        # Build technical feature vector (80 dims, normalized)
        technical_vector = self._build_technical_vector(
            harmonic_ratio=harmonic_ratio,
            attack_ms=attack_ms,
            decay_ms=base_decay,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            register=register,
            vibrato_rate=vibrato_rate,
            vibrato_depth=vibrato_depth,
            legato_score=legato_score,
            staccato_score=staccato_score,
        )
        
        # Build emotional feature vector (80 dims, normalized)
        emotional_vector = self._build_emotional_vector(
            energy=energy,
            dynamics_variance=dynamics_variance,
            brightness=brightness,
            tension=tension,
            valence=valence,
            arousal=arousal,
            timing_human=timing_human,
            velocity_human=velocity_human,
        )
        
        # Create sample dict
        sample = {
            'id': f"synth_inst_{family}_{index:05d}",
            'is_synthetic': True,
            
            # Technical labels
            'instrument_family': family,
            'instrument_specific': specific,
            'technique': technique,
            'articulation': articulation,
            'register': register,
            
            # Emotional labels
            'expression_style': emotion,
            'energy_level': int(energy * 8),  # 0-8 scale
            'musical_role': role,
            'sentiment_valence': valence,
            'sentiment_arousal': arousal,
            
            # Derived features
            'harmonic_ratio': harmonic_ratio,
            'attack_time_ms': attack_ms,
            'decay_time_ms': base_decay,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'vibrato_rate': vibrato_rate,
            'vibrato_depth': vibrato_depth,
            'legato_score': legato_score,
            'staccato_score': staccato_score,
            'brightness': brightness,
            'tension': tension,
            'timing_humanization': timing_human,
            'velocity_humanization': velocity_human,
            
            # Feature vectors
            'technical_vector': technical_vector,
            'emotional_vector': emotional_vector,
            'combined_vector': technical_vector + emotional_vector,
        }
        
        return sample
    
    def _build_technical_vector(self, **kwargs) -> List[float]:
        """Build 80-dim technical feature vector."""
        vec = []
        
        # Spectral features (40 dims) - simulated
        vec.append(kwargs.get('harmonic_ratio', 0.5))
        vec.extend([random.gauss(0.5, 0.1) for _ in range(7)])  # Spectral contrast
        vec.extend([random.gauss(0, 0.2) for _ in range(20)])   # MFCCs
        vec.extend([random.gauss(0, 0.1) for _ in range(12)])   # More spectral
        
        # Harmonic features (12 dims)
        vec.append(kwargs.get('harmonic_ratio', 0.5))
        vec.append(kwargs.get('pitch_mean', 60) / 127)
        vec.extend([random.gauss(0.5, 0.1) for _ in range(10)])
        
        # Attack/decay (12 dims)
        vec.append(kwargs.get('attack_ms', 50) / 500)
        vec.append(kwargs.get('decay_ms', 500) / 5000)
        vec.extend([random.gauss(0.5, 0.1) for _ in range(10)])
        
        # Technique (8 dims)
        vec.append(kwargs.get('vibrato_rate', 0) / 10)
        vec.append(kwargs.get('vibrato_depth', 0) / 100)
        vec.append(kwargs.get('legato_score', 0.5))
        vec.append(kwargs.get('staccato_score', 0.5))
        vec.extend([random.gauss(0.5, 0.1) for _ in range(4)])
        
        # Register (8 dims)
        vec.append(kwargs.get('pitch_mean', 60) / 127)
        vec.append(kwargs.get('pitch_std', 10) / 24)
        vec.append(kwargs.get('register', 1) / 2)
        vec.extend([random.gauss(0.5, 0.1) for _ in range(5)])
        
        # Pad/truncate to 80
        while len(vec) < 80:
            vec.append(random.gauss(0.5, 0.1))
        return vec[:80]
    
    def _build_emotional_vector(self, **kwargs) -> List[float]:
        """Build 80-dim emotional feature vector."""
        vec = []
        
        # Expression (20 dims)
        vec.append(kwargs.get('energy', 0.5))
        vec.append(kwargs.get('dynamics_variance', 0.2))
        vec.extend([random.gauss(kwargs.get('energy', 0.5), 0.1) for _ in range(8)])  # Dynamic curve
        vec.extend([random.gauss(0.5, 0.1) for _ in range(10)])
        
        # Energy (16 dims)
        vec.append(kwargs.get('energy', 0.5))
        vec.extend([random.gauss(kwargs.get('energy', 0.5), 0.1) for _ in range(8)])  # Energy curve
        vec.extend([random.gauss(0.5, 0.1) for _ in range(7)])
        
        # Sentiment (20 dims)
        vec.append(kwargs.get('brightness', 0.5))
        vec.append(kwargs.get('tension', 0.5))
        vec.append((kwargs.get('valence', 0) + 1) / 2)  # Normalize to 0-1
        vec.append(kwargs.get('arousal', 0.5))
        vec.extend([random.gauss(0.5, 0.1) for _ in range(16)])
        
        # Human feel (16 dims)
        vec.append(kwargs.get('timing_human', 15) / 50)
        vec.append(kwargs.get('velocity_human', 0.2))
        vec.extend([random.gauss(0.5, 0.1) for _ in range(14)])
        
        # Padding (8 dims)
        vec.extend([random.gauss(0.5, 0.1) for _ in range(8)])
        
        # Pad/truncate to 80
        while len(vec) < 80:
            vec.append(random.gauss(0.5, 0.1))
        return vec[:80]
    
    def _save_samples(self, samples: List[Dict], output_dir: Path):
        """Save samples to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        manifest = {
            'count': len(samples),
            'families': list(set(s['instrument_family'] for s in samples)),
            'emotions': list(set(s['expression_style'] for s in samples)),
            'generated_at': str(np.datetime64('now')),
        }
        
        with open(output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save samples
        for sample in samples:
            family_dir = output_dir / sample['instrument_family']
            family_dir.mkdir(exist_ok=True)
            
            with open(family_dir / f"{sample['id']}.json", 'w') as f:
                json.dump(sample, f, indent=2)
        
        logger.info(f"Saved {len(samples)} samples to {output_dir}")


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_instrument_samples(
    num_samples: int = 1000,
    output_dir: Optional[Path] = None,
    config: Optional[InstrumentGeneratorConfig] = None,
) -> List[Dict[str, Any]]:
    """Generate synthetic instrument samples."""
    generator = InstrumentSyntheticGenerator(config)
    return generator.generate_full_dataset(num_samples, output_dir)


def generate_for_instrument(
    instrument_family: str,
    num_samples: int = 100,
) -> List[Dict[str, Any]]:
    """Generate samples for a specific instrument family."""
    generator = InstrumentSyntheticGenerator()
    return generator.generate_for_family(instrument_family, num_samples)


def generate_for_expression(
    expression_style: str,
    num_samples: int = 100,
) -> List[Dict[str, Any]]:
    """Generate samples for a specific expression style."""
    generator = InstrumentSyntheticGenerator()
    return generator.generate_for_emotion(expression_style, num_samples)

