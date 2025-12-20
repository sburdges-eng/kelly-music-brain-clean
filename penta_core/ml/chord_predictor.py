"""
Chord Predictor - ML-based chord prediction.

Provides neural network-based chord prediction using:
- LSTM/GRU sequence models
- Transformer attention models
- Markov chain fallback
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

from python.penta_core.ml.model_registry import (
    ModelInfo,
    ModelBackend,
    ModelTask,
    get_model,
    register_model,
)
from python.penta_core.ml.inference import create_engine, InferenceEngine


# Chord vocabulary
CHORD_VOCAB = [
    # Major chords
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
    # Minor chords
    "Cm", "Dbm", "Dm", "Ebm", "Em", "Fm", "Gbm", "Gm", "Abm", "Am", "Bbm", "Bm",
    # Seventh chords
    "Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bm7b5",
    "Dbmaj7", "Ebm7", "Fm7", "Gbmaj7", "Ab7", "Bbm7", "Cm7b5",
    # Diminished
    "Cdim", "Ddim", "Edim", "Fdim", "Gdim", "Adim", "Bdim",
    # Augmented
    "Caug", "Daug", "Eaug", "Faug", "Gaug", "Aaug", "Baug",
    # Sus chords
    "Csus2", "Csus4", "Dsus2", "Dsus4", "Esus4", "Gsus4", "Asus2", "Asus4",
    # Other
    "N.C.",  # No chord
]

CHORD_TO_IDX = {chord: i for i, chord in enumerate(CHORD_VOCAB)}
IDX_TO_CHORD = {i: chord for i, chord in enumerate(CHORD_VOCAB)}


@dataclass
class ChordPrediction:
    """A chord prediction with confidence."""
    chord: str
    confidence: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def root(self) -> str:
        """Get chord root note."""
        for i, char in enumerate(self.chord):
            if char in "mb#d7s+au":
                return self.chord[:i] if i > 0 else self.chord[0]
        return self.chord

    @property
    def quality(self) -> str:
        """Get chord quality (major, minor, etc.)."""
        if "dim" in self.chord or "b5" in self.chord:
            return "diminished"
        elif "aug" in self.chord or "+" in self.chord:
            return "augmented"
        elif "m" in self.chord.lower() and "maj" not in self.chord:
            return "minor"
        elif "sus2" in self.chord:
            return "sus2"
        elif "sus4" in self.chord:
            return "sus4"
        return "major"


class ChordPredictor:
    """
    ML-based chord prediction.

    Supports:
    - Neural network models (LSTM, Transformer)
    - Markov chain fallback
    - Style-specific prediction
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_fallback: bool = True,
    ):
        """
        Initialize predictor.

        Args:
            model_name: Name of registered model (None for fallback only)
            use_fallback: Use Markov chain when model unavailable
        """
        self._engine: Optional[InferenceEngine] = None
        self._use_fallback = use_fallback
        self._markov_model: Optional[Dict] = None
        self._sequence_length = 4

        if model_name:
            model_info = get_model(model_name)
            if model_info:
                self._engine = create_engine(model_info)
                self._engine.load()

        # Initialize fallback
        if use_fallback:
            self._init_markov_model()

    def _init_markov_model(self) -> None:
        """Initialize Markov chain model with common progressions."""
        # Transition probabilities based on common chord progressions
        self._markov_model = defaultdict(lambda: defaultdict(float))

        # I-IV-V-I progressions
        progressions = [
            # Pop progressions
            ["C", "G", "Am", "F"],  # I-V-vi-IV
            ["C", "F", "G", "C"],  # I-IV-V-I
            ["Am", "F", "C", "G"],  # vi-IV-I-V
            ["C", "Am", "F", "G"],  # I-vi-IV-V
            # Jazz progressions
            ["Dm7", "G7", "Cmaj7", "Cmaj7"],  # ii-V-I
            ["Am7", "D7", "Gmaj7", "Cmaj7"],  # iii-VI-II-V
            ["Cmaj7", "Am7", "Dm7", "G7"],  # I-vi-ii-V
            # Blues
            ["C", "C", "C", "C", "F", "F", "C", "C", "G7", "F", "C", "G7"],
        ]

        # Build transition probabilities
        for prog in progressions:
            for i in range(len(prog) - 1):
                current = prog[i]
                next_chord = prog[i + 1]
                self._markov_model[current][next_chord] += 1

        # Normalize probabilities
        for current in self._markov_model:
            total = sum(self._markov_model[current].values())
            for next_chord in self._markov_model[current]:
                self._markov_model[current][next_chord] /= total

    def predict(
        self,
        context: List[str],
        num_predictions: int = 5,
        temperature: float = 1.0,
    ) -> ChordPrediction:
        """
        Predict the next chord given context.

        Args:
            context: Previous chords in the progression
            num_predictions: Number of alternative predictions
            temperature: Sampling temperature (higher = more random)

        Returns:
            ChordPrediction with top prediction and alternatives
        """
        # Try neural network first
        if self._engine and self._engine.is_loaded():
            return self._predict_neural(context, num_predictions, temperature)

        # Fall back to Markov model
        if self._use_fallback and self._markov_model:
            return self._predict_markov(context, num_predictions, temperature)

        # Default prediction
        return ChordPrediction(
            chord="C",
            confidence=0.5,
            alternatives=[("G", 0.3), ("Am", 0.2)],
        )

    def _predict_neural(
        self,
        context: List[str],
        num_predictions: int,
        temperature: float,
    ) -> ChordPrediction:
        """Predict using neural network."""
        # Encode context
        indices = [CHORD_TO_IDX.get(c, 0) for c in context[-self._sequence_length:]]

        # Pad if necessary
        while len(indices) < self._sequence_length:
            indices = [0] + indices

        # Create input array
        input_arr = np.array([indices], dtype=np.int64)

        # Run inference
        result = self._engine.infer({"input": input_arr})
        probs = result.get_output()

        # Apply temperature
        if temperature != 1.0:
            probs = np.log(probs + 1e-10) / temperature
            probs = np.exp(probs) / np.sum(np.exp(probs))

        # Get top predictions
        top_indices = np.argsort(probs.flatten())[-num_predictions:][::-1]
        top_probs = probs.flatten()[top_indices]

        # Convert to chords
        top_chord = IDX_TO_CHORD.get(top_indices[0], "C")
        alternatives = [
            (IDX_TO_CHORD.get(idx, "C"), float(prob))
            for idx, prob in zip(top_indices[1:], top_probs[1:])
        ]

        return ChordPrediction(
            chord=top_chord,
            confidence=float(top_probs[0]),
            alternatives=alternatives,
        )

    def _predict_markov(
        self,
        context: List[str],
        num_predictions: int,
        temperature: float,
    ) -> ChordPrediction:
        """Predict using Markov chain."""
        if not context:
            # Random start
            return ChordPrediction(chord="C", confidence=0.5)

        last_chord = context[-1]

        if last_chord not in self._markov_model:
            # Unknown chord, return common next
            return ChordPrediction(
                chord="G",
                confidence=0.3,
                alternatives=[("F", 0.25), ("Am", 0.2)],
            )

        # Get transition probabilities
        transitions = self._markov_model[last_chord]

        # Sort by probability
        sorted_trans = sorted(
            transitions.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        if not sorted_trans:
            return ChordPrediction(chord="C", confidence=0.3)

        top_chord = sorted_trans[0][0]
        top_prob = sorted_trans[0][1]

        alternatives = [
            (chord, prob)
            for chord, prob in sorted_trans[1:num_predictions]
        ]

        return ChordPrediction(
            chord=top_chord,
            confidence=top_prob,
            alternatives=alternatives,
        )

    def predict_progression(
        self,
        context: List[str],
        length: int = 4,
        temperature: float = 1.0,
    ) -> List[ChordPrediction]:
        """
        Predict a chord progression.

        Args:
            context: Starting chords
            length: Number of chords to predict
            temperature: Sampling temperature

        Returns:
            List of chord predictions
        """
        predictions = []
        current_context = list(context)

        for _ in range(length):
            pred = self.predict(current_context, temperature=temperature)
            predictions.append(pred)
            current_context.append(pred.chord)

        return predictions

    def suggest_resolution(
        self,
        context: List[str],
        target_chord: Optional[str] = None,
    ) -> List[ChordPrediction]:
        """
        Suggest chords that lead to resolution.

        Args:
            context: Current chord context
            target_chord: Target resolution chord (default: I chord)

        Returns:
            Suggested resolution path
        """
        if target_chord is None:
            # Infer tonic from context
            target_chord = self._infer_tonic(context)

        # Common resolution patterns
        resolutions = {
            "C": [["G7", "C"], ["Dm7", "G7", "C"], ["F", "G", "C"]],
            "G": [["D7", "G"], ["Am7", "D7", "G"], ["C", "D", "G"]],
            "D": [["A7", "D"], ["Em7", "A7", "D"]],
            "A": [["E7", "A"], ["Bm7", "E7", "A"]],
            "E": [["B7", "E"], ["F#m7", "B7", "E"]],
            "F": [["C7", "F"], ["Gm7", "C7", "F"]],
        }

        patterns = resolutions.get(target_chord, [["G7", "C"]])

        # Choose best pattern based on context
        best_pattern = patterns[0]

        predictions = []
        for chord in best_pattern:
            predictions.append(ChordPrediction(
                chord=chord,
                confidence=0.8,
            ))

        return predictions

    def _infer_tonic(self, context: List[str]) -> str:
        """Infer the tonic/key from chord context."""
        if not context:
            return "C"

        # Simple heuristic: first major chord or most common root
        root_counts = defaultdict(int)
        for chord in context:
            root = chord[0]
            if len(chord) > 1 and chord[1] == 'b':
                root = chord[:2]
            elif len(chord) > 1 and chord[1] == '#':
                root = chord[:2]
            root_counts[root] += 1

        return max(root_counts, key=root_counts.get) if root_counts else "C"


# Convenience functions
def predict_next_chord(
    context: List[str],
    model_name: Optional[str] = None,
) -> ChordPrediction:
    """Predict the next chord in a progression."""
    predictor = ChordPredictor(model_name)
    return predictor.predict(context)


def predict_progression(
    context: List[str],
    length: int = 4,
    model_name: Optional[str] = None,
) -> List[ChordPrediction]:
    """Predict a full chord progression."""
    predictor = ChordPredictor(model_name)
    return predictor.predict_progression(context, length)


def create_chord_model_info(
    name: str,
    path: str,
    backend: ModelBackend = ModelBackend.ONNX,
) -> ModelInfo:
    """
    Create and register a chord prediction model.

    Args:
        name: Model name
        path: Path to model file
        backend: Model backend

    Returns:
        ModelInfo for the registered model
    """
    model = ModelInfo(
        name=name,
        task=ModelTask.CHORD_PREDICTION,
        backend=backend,
        path=path,
        description="Chord sequence prediction model",
    )
    register_model(model)
    return model
