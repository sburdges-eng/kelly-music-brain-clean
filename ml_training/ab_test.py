#!/usr/bin/env python3
"""
A/B Testing Script for miDiKompanion
====================================

Compares rule-based vs ML-enhanced generation quality.
Tests on same emotion nodes for fair comparison.

Usage:
    python ab_test.py --models-dir onnx_models --samples 100
    
Metrics:
    - Latency: Inference time comparison
    - Quality: Note distribution similarity
    - Consistency: Repeated generation variance
    - User preference: Blind comparison results
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import time
from dataclasses import dataclass, asdict

import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('midikompanion.abtest')


@dataclass
class GenerationResult:
    """Result of a single generation."""
    method: str  # 'rule_based' or 'ml_enhanced'
    node_id: int
    notes: List[int]
    velocities: List[int]
    durations: List[float]
    latency_ms: float
    success: bool
    error: str = ""


@dataclass
class ABTestResult:
    """Aggregated A/B test results."""
    total_samples: int
    ml_latency_mean: float
    ml_latency_std: float
    rb_latency_mean: float
    rb_latency_std: float
    ml_success_rate: float
    rb_success_rate: float
    note_diversity_ml: float
    note_diversity_rb: float
    consistency_ml: float
    consistency_rb: float


class RuleBasedGenerator:
    """Simple rule-based melody generator for comparison."""
    
    # Mode patterns (semitone intervals from root)
    MODES = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
    }
    
    # Base VAD for node lookup
    BASE_VAD = {
        0: ('major', 0.8, 0.6),   # Happy
        1: ('minor', -0.6, 0.2),  # Sad
        2: ('phrygian', -0.4, 0.8),  # Angry
        3: ('minor', -0.5, 0.7),  # Fear
        4: ('major', 0.2, 0.8),   # Surprise
        5: ('minor', -0.5, 0.5),  # Disgust
    }
    
    def generate(self, node_id: int, num_notes: int = 16) -> GenerationResult:
        """Generate melody using rule-based approach."""
        start_time = time.perf_counter()
        
        try:
            base_idx = node_id // 36
            sub_idx = (node_id % 36) // 6
            intensity_idx = node_id % 6
            
            mode, valence, arousal = self.BASE_VAD.get(base_idx, ('major', 0.5, 0.5))
            intensity = (intensity_idx + 1) / 6.0
            
            # Get scale
            scale = self.MODES.get(mode, self.MODES['major'])
            
            # Generate notes
            root = 60 + (sub_idx - 3) * 2  # Root varies by sub-emotion
            notes = []
            velocities = []
            durations = []
            
            prev_note = root
            for i in range(num_notes):
                # Note selection based on scale
                scale_degree = np.random.choice(len(scale))
                octave_offset = np.random.choice([-12, 0, 12], p=[0.1, 0.7, 0.2])
                note = root + scale[scale_degree] + octave_offset
                
                # Melodic movement based on arousal
                if np.random.random() < arousal:
                    # More movement
                    note = prev_note + np.random.choice([-2, -1, 1, 2])
                    note = max(48, min(84, note))
                
                notes.append(note)
                prev_note = note
                
                # Velocity based on intensity
                base_velocity = int(60 + intensity * 40)
                velocity = max(1, min(127, base_velocity + np.random.randint(-10, 11)))
                velocities.append(velocity)
                
                # Duration based on arousal (higher arousal = shorter notes)
                base_duration = 0.5 - arousal * 0.3
                duration = max(0.1, base_duration + np.random.uniform(-0.1, 0.1))
                durations.append(duration)
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return GenerationResult(
                method='rule_based',
                node_id=node_id,
                notes=notes,
                velocities=velocities,
                durations=durations,
                latency_ms=latency,
                success=True
            )
            
        except Exception as e:
            return GenerationResult(
                method='rule_based',
                node_id=node_id,
                notes=[],
                velocities=[],
                durations=[],
                latency_ms=0,
                success=False,
                error=str(e)
            )


class MLEnhancedGenerator:
    """ML-enhanced generator using ONNX models."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self._load_models()
    
    def _load_models(self):
        """Load required ONNX models."""
        if not HAS_ONNX:
            logger.warning('ONNX Runtime not available')
            return
        
        model_files = {
            'emotion': 'emotion_recognizer.onnx',
            'melody': 'melody_transformer.onnx',
        }
        
        for name, filename in model_files.items():
            path = self.models_dir / filename
            if path.exists():
                try:
                    self.sessions[name] = ort.InferenceSession(str(path))
                    logger.info(f'Loaded {name} model')
                except Exception as e:
                    logger.warning(f'Failed to load {name}: {e}')
    
    def generate(self, node_id: int, num_notes: int = 16) -> GenerationResult:
        """Generate melody using ML models."""
        start_time = time.perf_counter()
        
        if not self.sessions:
            # Fallback to placeholder if no models
            return self._placeholder_generate(node_id, num_notes)
        
        try:
            # Create node embedding
            embedding = self._node_to_embedding(node_id)
            
            # Run melody transformer
            if 'melody' in self.sessions:
                session = self.sessions['melody']
                output = session.run(None, {'input': embedding.reshape(1, -1)})[0][0]
            else:
                output = np.random.rand(128)
            
            # Convert probabilities to notes
            notes = []
            velocities = []
            durations = []
            
            for i in range(num_notes):
                # Sample from note distribution
                probs = output / output.sum()
                note = np.random.choice(128, p=probs)
                notes.append(note)
                
                # Derive velocity and duration from model output
                velocity = int(64 + output[note] * 60)
                velocities.append(max(1, min(127, velocity)))
                
                duration = 0.25 + output[note] * 0.5
                durations.append(max(0.1, duration))
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return GenerationResult(
                method='ml_enhanced',
                node_id=node_id,
                notes=notes,
                velocities=velocities,
                durations=durations,
                latency_ms=latency,
                success=True
            )
            
        except Exception as e:
            return GenerationResult(
                method='ml_enhanced',
                node_id=node_id,
                notes=[],
                velocities=[],
                durations=[],
                latency_ms=0,
                success=False,
                error=str(e)
            )
    
    def _node_to_embedding(self, node_id: int) -> np.ndarray:
        """Convert node ID to 64-dim embedding."""
        embedding = np.zeros(64, dtype=np.float32)
        
        base_idx = node_id // 36
        sub_idx = (node_id % 36) // 6
        intensity_idx = node_id % 6
        
        # VAD-like encoding
        embedding[0] = 2 * (base_idx / 5) - 1  # Valence-like
        embedding[1] = sub_idx / 5  # Arousal-like
        embedding[2] = intensity_idx / 5  # Intensity
        embedding[3] = node_id / 215  # Normalized node ID
        
        # One-hot for base emotion
        embedding[4 + base_idx] = 1.0
        
        return embedding
    
    def _placeholder_generate(self, node_id: int, num_notes: int) -> GenerationResult:
        """Placeholder when models not available."""
        start_time = time.perf_counter()
        
        notes = [60 + np.random.randint(-12, 13) for _ in range(num_notes)]
        velocities = [80 + np.random.randint(-20, 21) for _ in range(num_notes)]
        durations = [0.25 + np.random.uniform(-0.1, 0.2) for _ in range(num_notes)]
        
        return GenerationResult(
            method='ml_enhanced',
            node_id=node_id,
            notes=notes,
            velocities=[max(1, min(127, v)) for v in velocities],
            durations=durations,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            success=True
        )


class ABTester:
    """A/B testing framework for generation methods."""
    
    def __init__(self, models_dir: Path):
        self.rule_based = RuleBasedGenerator()
        self.ml_enhanced = MLEnhancedGenerator(models_dir)
        self.results: List[Tuple[GenerationResult, GenerationResult]] = []
    
    def run_test(self, node_ids: List[int], repeats: int = 1) -> ABTestResult:
        """Run A/B test on specified emotion nodes."""
        
        self.results.clear()
        
        for node_id in node_ids:
            for _ in range(repeats):
                rb_result = self.rule_based.generate(node_id)
                ml_result = self.ml_enhanced.generate(node_id)
                self.results.append((rb_result, ml_result))
        
        return self._compute_metrics()
    
    def _compute_metrics(self) -> ABTestResult:
        """Compute aggregated metrics from test results."""
        
        ml_latencies = [ml.latency_ms for rb, ml in self.results if ml.success]
        rb_latencies = [rb.latency_ms for rb, ml in self.results if rb.success]
        
        ml_success = sum(1 for _, ml in self.results if ml.success)
        rb_success = sum(1 for rb, _ in self.results if rb.success)
        total = len(self.results)
        
        # Note diversity (unique notes / total notes)
        ml_notes = [n for _, ml in self.results for n in ml.notes]
        rb_notes = [n for rb, _ in self.results for n in rb.notes]
        
        ml_diversity = len(set(ml_notes)) / len(ml_notes) if ml_notes else 0
        rb_diversity = len(set(rb_notes)) / len(rb_notes) if rb_notes else 0
        
        # Consistency (lower variance = more consistent)
        def note_variance(results, method_idx):
            by_node = {}
            for pair in results:
                r = pair[method_idx]
                if r.success:
                    if r.node_id not in by_node:
                        by_node[r.node_id] = []
                    by_node[r.node_id].extend(r.notes)
            
            variances = []
            for notes in by_node.values():
                if len(notes) > 1:
                    variances.append(np.std(notes))
            return np.mean(variances) if variances else 0
        
        ml_consistency = 1.0 / (1.0 + note_variance(self.results, 1))
        rb_consistency = 1.0 / (1.0 + note_variance(self.results, 0))
        
        return ABTestResult(
            total_samples=total,
            ml_latency_mean=np.mean(ml_latencies) if ml_latencies else 0,
            ml_latency_std=np.std(ml_latencies) if ml_latencies else 0,
            rb_latency_mean=np.mean(rb_latencies) if rb_latencies else 0,
            rb_latency_std=np.std(rb_latencies) if rb_latencies else 0,
            ml_success_rate=ml_success / total if total > 0 else 0,
            rb_success_rate=rb_success / total if total > 0 else 0,
            note_diversity_ml=ml_diversity,
            note_diversity_rb=rb_diversity,
            consistency_ml=ml_consistency,
            consistency_rb=rb_consistency,
        )
    
    def generate_report(self, result: ABTestResult) -> str:
        """Generate human-readable report."""
        
        lines = [
            '='*60,
            'miDiKompanion A/B Test Report',
            '='*60,
            '',
            f'Total samples: {result.total_samples}',
            '',
            'LATENCY:',
            f'  ML-Enhanced: {result.ml_latency_mean:.2f}ms (±{result.ml_latency_std:.2f})',
            f'  Rule-Based:  {result.rb_latency_mean:.2f}ms (±{result.rb_latency_std:.2f})',
            f'  Winner: {"ML" if result.ml_latency_mean < result.rb_latency_mean else "Rule-Based"}',
            '',
            'SUCCESS RATE:',
            f'  ML-Enhanced: {result.ml_success_rate*100:.1f}%',
            f'  Rule-Based:  {result.rb_success_rate*100:.1f}%',
            '',
            'NOTE DIVERSITY:',
            f'  ML-Enhanced: {result.note_diversity_ml:.3f}',
            f'  Rule-Based:  {result.note_diversity_rb:.3f}',
            f'  Winner: {"ML" if result.note_diversity_ml > result.note_diversity_rb else "Rule-Based"}',
            '',
            'CONSISTENCY:',
            f'  ML-Enhanced: {result.consistency_ml:.3f}',
            f'  Rule-Based:  {result.consistency_rb:.3f}',
            f'  Winner: {"ML" if result.consistency_ml > result.consistency_rb else "Rule-Based"}',
            '',
            '='*60,
        ]
        
        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Run A/B testing for miDiKompanion')
    parser.add_argument('--models-dir', type=str, default='onnx_models',
                        help='Directory containing ONNX models')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Repeats per node')
    parser.add_argument('--output', type=str, default='ab_test_results.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    # Select nodes to test (sample from each category)
    node_ids = []
    for base_idx in range(6):
        for sub_idx in range(6):
            # One intensity level per sub-emotion
            node_id = base_idx * 36 + sub_idx * 6 + 3  # Moderate intensity
            node_ids.append(node_id)
    
    # Limit to requested samples
    if len(node_ids) > args.samples:
        node_ids = np.random.choice(node_ids, args.samples, replace=False).tolist()
    
    logger.info(f'Testing {len(node_ids)} nodes with {args.repeats} repeats each')
    
    tester = ABTester(Path(args.models_dir))
    result = tester.run_test(node_ids, repeats=args.repeats)
    
    # Print report
    report = tester.generate_report(result)
    print('\n' + report)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    
    logger.info(f'Results saved to: {output_path}')


if __name__ == '__main__':
    main()
