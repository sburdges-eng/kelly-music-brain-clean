"""
Intent-Based Dataset for ML Training.

Uses CompleteSongIntent schema as training targets for emotion-driven music generation.
This module demonstrates how to structure intent data for ML model training and validation.

Key Features:
- Convert intent schema to model targets (embeddings, labels, continuous values)
- Validate intent consistency for training
- Batch intent data for model input
- Map intent fields to specific model architectures

Usage:
    from python.penta_core.ml.datasets.intent_dataset import IntentDataset, IntentEncoder
    
    # Load intent files
    dataset = IntentDataset(intent_dir="path/to/intents")
    
    # Get encoded targets
    batch = dataset.get_batch(indices=[0, 1, 2])
    # batch contains: emotion_labels, tension_values, rule_break_ids, etc.
    
    # For PyTorch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try to import music_brain intent schema
try:
    from music_brain.session.intent_schema import (
        CompleteSongIntent,
        SongRoot,
        SongIntent,
        TechnicalConstraints,
        validate_intent,
        RULE_BREAKING_EFFECTS,
    )
    INTENT_SCHEMA_AVAILABLE = True
except ImportError:
    INTENT_SCHEMA_AVAILABLE = False
    logger.warning("music_brain.session.intent_schema not available")


# =============================================================================
# Intent Encoding for ML Models
# =============================================================================


@dataclass
class IntentEncodingConfig:
    """Configuration for encoding intent schema to model targets."""
    
    # Emotion encoding
    emotion_labels: List[str] = field(default_factory=lambda: [
        "grief", "joy", "anger", "anxiety", "nostalgia", "defiance", 
        "longing", "calm", "melancholy", "hope"
    ])
    
    # Vulnerability encoding
    vulnerability_map: Dict[str, int] = field(default_factory=lambda: {
        "Low": 0, "Medium": 1, "High": 2
    })
    
    # Narrative arc encoding
    narrative_arc_labels: List[str] = field(default_factory=lambda: [
        "Climb-to-Climax", "Slow Reveal", "Repetitive Despair",
        "Static Reflection", "Sudden Shift", "Descent", 
        "Rise and Fall", "Spiral"
    ])
    
    # Rule breaking categories
    rule_break_categories: List[str] = field(default_factory=lambda: [
        "HARMONY", "RHYTHM", "ARRANGEMENT", "PRODUCTION", "NONE"
    ])
    
    # Continuous value ranges
    tension_range: Tuple[float, float] = (0.0, 1.0)
    tempo_range: Tuple[int, int] = (40, 200)
    
    # Feature dimensions
    embed_dim_emotion: int = 64
    embed_dim_rule_break: int = 32


class IntentEncoder:
    """
    Encodes CompleteSongIntent to ML model targets.
    
    Converts intent schema fields to:
    - Categorical labels (emotion, narrative arc, rule breaks)
    - Continuous values (tension, tempo)
    - One-hot encodings
    - Embeddings (for neural networks)
    """
    
    def __init__(self, config: Optional[IntentEncodingConfig] = None):
        self.config = config or IntentEncodingConfig()
        
        # Build label mappings
        self.emotion_to_id = {e: i for i, e in enumerate(self.config.emotion_labels)}
        self.arc_to_id = {a: i for i, a in enumerate(self.config.narrative_arc_labels)}
        self.rule_category_to_id = {r: i for i, r in enumerate(self.config.rule_break_categories)}
        
        # Collect all specific rules
        self.all_rules = self._collect_all_rules()
        self.rule_to_id = {rule: i for i, rule in enumerate(self.all_rules)}
    
    def _collect_all_rules(self) -> List[str]:
        """Collect all possible rule-breaking values."""
        if not INTENT_SCHEMA_AVAILABLE:
            return []
        
        rules = list(RULE_BREAKING_EFFECTS.keys())
        rules.append("NONE")  # For no rule breaking
        return sorted(rules)
    
    def encode_intent(self, intent: CompleteSongIntent) -> Dict[str, Any]:
        """
        Encode complete song intent to model targets.
        
        Args:
            intent: CompleteSongIntent instance
        
        Returns:
            Dictionary with encoded targets:
            - emotion_label: int (0-N)
            - emotion_onehot: np.ndarray
            - tension: float (0.0-1.0)
            - vulnerability: int (0-2)
            - narrative_arc_label: int
            - rule_break_id: int
            - rule_break_category: int
            - tempo_bpm: float
            - has_justification: bool
            - intent_valid: bool
        """
        if not INTENT_SCHEMA_AVAILABLE:
            raise ImportError("music_brain.session.intent_schema required")
        
        # Phase 1: Emotional Intent
        # Note: Using .lower() for defensive programming - allows "Grief", "grief", "GRIEF"
        # in user-created JSON files while maintaining consistent encoding
        emotion = intent.song_intent.mood_primary.lower()
        emotion_id = self.emotion_to_id.get(emotion, 0)
        emotion_onehot = self._to_onehot(emotion_id, len(self.config.emotion_labels))
        
        tension = np.clip(
            intent.song_intent.mood_secondary_tension,
            self.config.tension_range[0],
            self.config.tension_range[1]
        )
        
        vulnerability = self.config.vulnerability_map.get(
            intent.song_intent.vulnerability_scale, 1
        )
        
        narrative_arc = intent.song_intent.narrative_arc
        arc_id = self.arc_to_id.get(narrative_arc, 0)
        
        # Phase 2: Technical Constraints
        rule_break = intent.technical_constraints.technical_rule_to_break or "NONE"
        rule_break_id = self.rule_to_id.get(rule_break, len(self.rule_to_id) - 1)
        
        # Get rule category (HARMONY, RHYTHM, etc.)
        rule_category = "NONE"
        if rule_break != "NONE" and "_" in rule_break:
            rule_category = rule_break.split("_")[0]
        rule_category_id = self.rule_category_to_id.get(rule_category, len(self.rule_category_to_id) - 1)
        
        # Tempo (handle range)
        tempo_range = intent.technical_constraints.technical_tempo_range
        tempo_avg = (tempo_range[0] + tempo_range[1]) / 2.0
        tempo_normalized = (tempo_avg - self.config.tempo_range[0]) / (
            self.config.tempo_range[1] - self.config.tempo_range[0]
        )
        
        # Metadata
        has_justification = bool(intent.technical_constraints.rule_breaking_justification)
        
        # Validation
        issues = validate_intent(intent)
        intent_valid = len(issues) == 0
        
        return {
            # Emotion targets
            "emotion_label": emotion_id,
            "emotion_onehot": emotion_onehot,
            "emotion_str": emotion,
            
            # Continuous values
            "tension": float(tension),
            "vulnerability": vulnerability,
            "tempo_normalized": float(tempo_normalized),
            "tempo_bpm": float(tempo_avg),
            
            # Structural
            "narrative_arc_label": arc_id,
            "narrative_arc_str": narrative_arc,
            
            # Rule breaking
            "rule_break_id": rule_break_id,
            "rule_break_str": rule_break,
            "rule_break_category": rule_category_id,
            "has_justification": has_justification,
            
            # Validation
            "intent_valid": intent_valid,
            "validation_issues": len(issues),
        }
    
    def _to_onehot(self, index: int, num_classes: int) -> np.ndarray:
        """Convert index to one-hot encoding."""
        onehot = np.zeros(num_classes, dtype=np.float32)
        if 0 <= index < num_classes:
            onehot[index] = 1.0
        return onehot
    
    def decode_emotion(self, emotion_id: int) -> str:
        """Decode emotion ID back to string."""
        if 0 <= emotion_id < len(self.config.emotion_labels):
            return self.config.emotion_labels[emotion_id]
        return "unknown"
    
    def decode_rule_break(self, rule_id: int) -> str:
        """Decode rule break ID back to string."""
        if 0 <= rule_id < len(self.all_rules):
            return self.all_rules[rule_id]
        return "unknown"


# =============================================================================
# Intent Dataset
# =============================================================================


class IntentDataset:
    """
    Dataset of CompleteSongIntent instances for ML training.
    
    Loads intent JSON files and provides encoded targets for training.
    Compatible with PyTorch DataLoader when __getitem__ is used.
    
    Usage:
        dataset = IntentDataset(intent_dir="path/to/intents")
        print(f"Loaded {len(dataset)} intents")
        
        # Get single sample
        sample = dataset[0]
        # sample["intent"] = CompleteSongIntent
        # sample["targets"] = encoded targets dict
        
        # Get batch
        batch = dataset.get_batch([0, 1, 2])
    """
    
    def __init__(
        self,
        intent_dir: Union[str, Path],
        encoding_config: Optional[IntentEncodingConfig] = None,
        validate_on_load: bool = True,
    ):
        """
        Initialize intent dataset.
        
        Args:
            intent_dir: Directory containing intent JSON files
            encoding_config: Configuration for encoding
            validate_on_load: If True, validate all intents on load
        """
        if not INTENT_SCHEMA_AVAILABLE:
            raise ImportError("music_brain.session.intent_schema required")
        
        self.intent_dir = Path(intent_dir)
        self.encoder = IntentEncoder(encoding_config)
        self.validate_on_load = validate_on_load
        
        # Load all intents
        self.intents: List[CompleteSongIntent] = []
        self.intent_paths: List[Path] = []
        self.encoded_targets: List[Dict[str, Any]] = []
        
        self._load_intents()
    
    def _load_intents(self):
        """Load all intent JSON files from directory."""
        if not self.intent_dir.exists():
            raise FileNotFoundError(f"Intent directory not found: {self.intent_dir}")
        
        json_files = sorted(self.intent_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.intent_dir}")
            return
        
        for json_file in json_files:
            try:
                intent = CompleteSongIntent.load(str(json_file))
                
                # Validate if requested
                if self.validate_on_load:
                    issues = validate_intent(intent)
                    if issues:
                        logger.warning(f"Validation issues in {json_file.name}: {issues}")
                
                # Encode targets
                targets = self.encoder.encode_intent(intent)
                
                self.intents.append(intent)
                self.intent_paths.append(json_file)
                self.encoded_targets.append(targets)
                
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(self.intents)} intents from {self.intent_dir}")
    
    def __len__(self) -> int:
        """Return number of intents in dataset."""
        return len(self.intents)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get single sample (PyTorch compatible).
        
        Returns:
            Dictionary with:
            - intent: CompleteSongIntent instance
            - targets: Encoded targets dict
            - path: Path to intent JSON file
        """
        return {
            "intent": self.intents[index],
            "targets": self.encoded_targets[index],
            "path": str(self.intent_paths[index]),
        }
    
    def get_batch(self, indices: List[int]) -> Dict[str, np.ndarray]:
        """
        Get batch of samples with targets stacked as arrays.
        
        Args:
            indices: List of sample indices
        
        Returns:
            Dictionary with batched arrays:
            - emotion_labels: (batch_size,)
            - tensions: (batch_size,)
            - etc.
        """
        batch = {
            "emotion_labels": [],
            "emotion_onehots": [],
            "tensions": [],
            "vulnerabilities": [],
            "narrative_arc_labels": [],
            "rule_break_ids": [],
            "rule_break_categories": [],
            "tempo_normalized": [],
            "tempo_bpm": [],
            "has_justification": [],
            "intent_valid": [],
        }
        
        for idx in indices:
            targets = self.encoded_targets[idx]
            batch["emotion_labels"].append(targets["emotion_label"])
            batch["emotion_onehots"].append(targets["emotion_onehot"])
            batch["tensions"].append(targets["tension"])
            batch["vulnerabilities"].append(targets["vulnerability"])
            batch["narrative_arc_labels"].append(targets["narrative_arc_label"])
            batch["rule_break_ids"].append(targets["rule_break_id"])
            batch["rule_break_categories"].append(targets["rule_break_category"])
            batch["tempo_normalized"].append(targets["tempo_normalized"])
            batch["tempo_bpm"].append(targets["tempo_bpm"])
            batch["has_justification"].append(targets["has_justification"])
            batch["intent_valid"].append(targets["intent_valid"])
        
        # Convert to numpy arrays
        return {
            "emotion_labels": np.array(batch["emotion_labels"], dtype=np.int64),
            "emotion_onehots": np.array(batch["emotion_onehots"], dtype=np.float32),
            "tensions": np.array(batch["tensions"], dtype=np.float32),
            "vulnerabilities": np.array(batch["vulnerabilities"], dtype=np.int64),
            "narrative_arc_labels": np.array(batch["narrative_arc_labels"], dtype=np.int64),
            "rule_break_ids": np.array(batch["rule_break_ids"], dtype=np.int64),
            "rule_break_categories": np.array(batch["rule_break_categories"], dtype=np.int64),
            "tempo_normalized": np.array(batch["tempo_normalized"], dtype=np.float32),
            "tempo_bpm": np.array(batch["tempo_bpm"], dtype=np.float32),
            "has_justification": np.array(batch["has_justification"], dtype=np.bool_),
            "intent_valid": np.array(batch["intent_valid"], dtype=np.bool_),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with counts and distributions
        """
        from collections import Counter
        
        emotions = [t["emotion_str"] for t in self.encoded_targets]
        arcs = [t["narrative_arc_str"] for t in self.encoded_targets]
        rules = [t["rule_break_str"] for t in self.encoded_targets]
        tensions = [t["tension"] for t in self.encoded_targets]
        tempos = [t["tempo_bpm"] for t in self.encoded_targets]
        
        return {
            "total_samples": len(self),
            "emotion_distribution": dict(Counter(emotions)),
            "narrative_arc_distribution": dict(Counter(arcs)),
            "rule_break_distribution": dict(Counter(rules)),
            "tension_stats": {
                "mean": float(np.mean(tensions)),
                "std": float(np.std(tensions)),
                "min": float(np.min(tensions)),
                "max": float(np.max(tensions)),
            },
            "tempo_stats": {
                "mean": float(np.mean(tempos)),
                "std": float(np.std(tempos)),
                "min": float(np.min(tempos)),
                "max": float(np.max(tempos)),
            },
        }


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_dataset_for_training(
    dataset: IntentDataset,
    min_samples_per_emotion: int = 5,
) -> Tuple[bool, List[str]]:
    """
    Validate dataset is suitable for training.
    
    Args:
        dataset: IntentDataset instance
        min_samples_per_emotion: Minimum samples required per emotion
    
    Returns:
        (is_valid, issues_list)
    """
    issues = []
    
    # Check minimum samples
    if len(dataset) == 0:
        issues.append("Dataset is empty")
        return False, issues
    
    # Get statistics
    stats = dataset.get_statistics()
    
    # Check emotion balance
    emotion_dist = stats["emotion_distribution"]
    for emotion, count in emotion_dist.items():
        if count < min_samples_per_emotion:
            issues.append(
                f"Emotion '{emotion}' has only {count} samples "
                f"(minimum: {min_samples_per_emotion})"
            )
    
    # Check for invalid intents
    invalid_count = sum(
        1 for t in dataset.encoded_targets if not t["intent_valid"]
    )
    if invalid_count > 0:
        issues.append(
            f"{invalid_count}/{len(dataset)} intents have validation issues"
        )
    
    # Check tension range
    tension_stats = stats["tension_stats"]
    if tension_stats["std"] < 0.1:
        issues.append(
            f"Tension values have low variance (std={tension_stats['std']:.3f})"
        )
    
    is_valid = len(issues) == 0
    return is_valid, issues
