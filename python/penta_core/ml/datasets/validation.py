"""
Dataset Validation for Kelly ML Training.

Validates datasets before training:
- Balance: Equal samples per category
- Diversity: Variety in keys, tempos, articulations
- Quality: File integrity, feature extraction success
- Statistics: Comprehensive dataset analysis
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Report
# =============================================================================


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: str = "warning"  # info, warning, error
    category: str = ""  # balance, diversity, quality, structure
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    
    # Dataset info
    dataset_id: str = ""
    dataset_path: str = ""
    validation_time: str = ""
    
    # Overall status
    is_valid: bool = True
    total_samples: int = 0
    issues_count: int = 0
    
    # Balance check
    balance_score: float = 0.0  # 0-1, 1 = perfectly balanced
    category_counts: Dict[str, int] = field(default_factory=dict)
    min_category_size: int = 0
    max_category_size: int = 0
    
    # Diversity check
    diversity_score: float = 0.0  # 0-1, 1 = highly diverse
    key_distribution: Dict[str, int] = field(default_factory=dict)
    tempo_distribution: Dict[str, int] = field(default_factory=dict)
    mode_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Quality check
    quality_score: float = 0.0  # 0-1, 1 = all files valid
    corrupt_files: List[str] = field(default_factory=list)
    missing_annotations: List[str] = field(default_factory=list)
    
    # Split distribution
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    
    # Duration statistics
    total_duration_hours: float = 0.0
    avg_duration_sec: float = 0.0
    duration_std: float = 0.0
    
    # Issues
    issues: List[ValidationIssue] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def add_issue(
        self,
        severity: str,
        category: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Add a validation issue."""
        self.issues.append(ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            details=details or {},
        ))
        self.issues_count = len(self.issues)
        
        if severity == "error":
            self.is_valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        d = asdict(self)
        d['issues'] = [asdict(i) for i in self.issues]
        return d
    
    def save(self, path: Path):
        """Save report to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "=" * 70)
        print(f"Dataset Validation Report: {self.dataset_id}")
        print("=" * 70)
        
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        print(f"\nStatus: {status}")
        print(f"Total Samples: {self.total_samples}")
        print(f"Total Duration: {self.total_duration_hours:.2f} hours")
        
        print(f"\nScores:")
        print(f"  Balance:   {self.balance_score:.2%}")
        print(f"  Diversity: {self.diversity_score:.2%}")
        print(f"  Quality:   {self.quality_score:.2%}")
        
        print(f"\nSplits:")
        print(f"  Train: {self.train_count}")
        print(f"  Val:   {self.val_count}")
        print(f"  Test:  {self.test_count}")
        
        if self.category_counts:
            print(f"\nCategory Distribution:")
            for cat, count in sorted(self.category_counts.items()):
                bar = "█" * min(50, int(count / max(self.category_counts.values()) * 50))
                print(f"  {cat:15} {count:5} {bar}")
        
        if self.issues:
            print(f"\nIssues ({len(self.issues)}):")
            for issue in self.issues[:10]:
                icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "•")
                print(f"  {icon} [{issue.category}] {issue.message}")
            if len(self.issues) > 10:
                print(f"  ... and {len(self.issues) - 10} more")
        
        if self.recommendations:
            print(f"\nRecommendations:")
            for rec in self.recommendations:
                print(f"  💡 {rec}")
        
        print("\n" + "=" * 70)


# =============================================================================
# Dataset Validator
# =============================================================================


class DatasetValidator:
    """
    Validates datasets for ML training readiness.
    
    Checks:
    - Balance: Are categories evenly distributed?
    - Diversity: Are there varied keys, tempos, etc.?
    - Quality: Are files readable and properly annotated?
    - Structure: Is the dataset format correct?
    
    Usage:
        validator = DatasetValidator()
        report = validator.validate("datasets/emotion_dataset_v1")
        report.print_summary()
    """
    
    def __init__(
        self,
        min_samples_per_category: int = 100,
        balance_threshold: float = 0.5,  # Min ratio of smallest/largest category
        diversity_threshold: float = 0.3,  # Min diversity score
    ):
        self.min_samples_per_category = min_samples_per_category
        self.balance_threshold = balance_threshold
        self.diversity_threshold = diversity_threshold
    
    def validate(self, dataset_path: Path) -> ValidationReport:
        """Validate a dataset directory."""
        from datetime import datetime
        
        dataset_path = Path(dataset_path)
        report = ValidationReport(
            dataset_path=str(dataset_path),
            dataset_id=dataset_path.name,
            validation_time=datetime.now().isoformat(),
        )
        
        # Check structure
        self._check_structure(dataset_path, report)
        
        # Load manifest
        manifest_path = dataset_path / "manifest.json"
        if manifest_path.exists():
            from python.penta_core.ml.datasets.base import load_manifest
            manifest = load_manifest(manifest_path)
            
            # Validate samples
            self._validate_samples(manifest, dataset_path, report)
            
            # Check balance
            self._check_balance(manifest, report)
            
            # Check diversity
            self._check_diversity(manifest, report)
            
            # Check quality
            self._check_quality(manifest, dataset_path, report)
        else:
            # Try to validate from files directly
            self._validate_from_files(dataset_path, report)
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _check_structure(self, dataset_path: Path, report: ValidationReport):
        """Check dataset directory structure."""
        required_dirs = ['raw', 'annotations']
        optional_dirs = ['processed', 'splits']
        
        for dir_name in required_dirs:
            if not (dataset_path / dir_name).exists():
                report.add_issue(
                    "warning",
                    "structure",
                    f"Missing directory: {dir_name}",
                )
        
        if not (dataset_path / "manifest.json").exists():
            report.add_issue(
                "warning",
                "structure",
                "Missing manifest.json - will attempt to validate from files",
            )
    
    def _validate_samples(
        self,
        manifest,
        dataset_path: Path,
        report: ValidationReport,
    ):
        """Validate samples from manifest."""
        report.total_samples = len(manifest.samples)
        
        durations = []
        
        for sample in manifest.samples:
            # Check file exists
            file_path = dataset_path / sample.file_path
            if not file_path.exists():
                report.add_issue(
                    "error",
                    "quality",
                    f"Missing file: {sample.file_path}",
                )
                report.corrupt_files.append(sample.file_path)
            
            # Check annotations
            if not sample.annotations or not sample.annotations.emotion:
                report.missing_annotations.append(sample.sample_id)
            
            # Track splits
            if sample.split == 'train':
                report.train_count += 1
            elif sample.split == 'val':
                report.val_count += 1
            elif sample.split == 'test':
                report.test_count += 1
            
            # Duration
            if sample.duration_sec > 0:
                durations.append(sample.duration_sec)
        
        # Duration statistics
        if durations:
            report.total_duration_hours = sum(durations) / 3600
            report.avg_duration_sec = np.mean(durations)
            report.duration_std = np.std(durations)
    
    def _check_balance(self, manifest, report: ValidationReport):
        """Check category balance."""
        # Count samples per category
        category_counts = Counter()
        
        for sample in manifest.samples:
            if sample.annotations and sample.annotations.emotion:
                category_counts[sample.annotations.emotion] += 1
        
        report.category_counts = dict(category_counts)
        
        if not category_counts:
            report.add_issue(
                "error",
                "balance",
                "No emotion labels found in dataset",
            )
            return
        
        counts = list(category_counts.values())
        report.min_category_size = min(counts)
        report.max_category_size = max(counts)
        
        # Balance score: ratio of min to max
        if report.max_category_size > 0:
            report.balance_score = report.min_category_size / report.max_category_size
        
        # Check for issues
        if report.balance_score < self.balance_threshold:
            report.add_issue(
                "warning",
                "balance",
                f"Dataset is imbalanced (score: {report.balance_score:.2f})",
                {"min": report.min_category_size, "max": report.max_category_size},
            )
        
        # Check minimum samples
        for cat, count in category_counts.items():
            if count < self.min_samples_per_category:
                report.add_issue(
                    "warning",
                    "balance",
                    f"Category '{cat}' has only {count} samples (min: {self.min_samples_per_category})",
                )
    
    def _check_diversity(self, manifest, report: ValidationReport):
        """Check dataset diversity."""
        keys = Counter()
        modes = Counter()
        tempos = Counter()
        grooves = Counter()
        
        for sample in manifest.samples:
            if sample.annotations:
                ann = sample.annotations
                
                if ann.key:
                    keys[ann.key] += 1
                
                if ann.mode:
                    modes[ann.mode] += 1
                
                if ann.tempo_bpm > 0:
                    # Bin tempos
                    tempo_bin = f"{int(ann.tempo_bpm // 20) * 20}-{int(ann.tempo_bpm // 20) * 20 + 20}"
                    tempos[tempo_bin] += 1
                
                if ann.groove_type:
                    grooves[ann.groove_type] += 1
        
        report.key_distribution = dict(keys)
        report.mode_distribution = dict(modes)
        report.tempo_distribution = dict(tempos)
        
        # Calculate diversity score
        diversity_scores = []
        
        # Key diversity (12 possible keys)
        if keys:
            key_entropy = self._entropy(list(keys.values()), 12)
            diversity_scores.append(key_entropy)
        
        # Mode diversity (2 possible modes)
        if modes:
            mode_entropy = self._entropy(list(modes.values()), 2)
            diversity_scores.append(mode_entropy)
        
        # Tempo diversity
        if tempos:
            tempo_entropy = self._entropy(list(tempos.values()), 10)  # ~10 bins
            diversity_scores.append(tempo_entropy)
        
        if diversity_scores:
            report.diversity_score = np.mean(diversity_scores)
        
        # Check for issues
        if len(keys) < 6:
            report.add_issue(
                "warning",
                "diversity",
                f"Low key diversity: only {len(keys)} keys represented",
                {"keys": list(keys.keys())},
            )
        
        if len(modes) < 2:
            report.add_issue(
                "info",
                "diversity",
                f"Limited mode diversity: only {list(modes.keys())} represented",
            )
        
        if report.diversity_score < self.diversity_threshold:
            report.add_issue(
                "warning",
                "diversity",
                f"Low overall diversity score: {report.diversity_score:.2f}",
            )
    
    def _entropy(self, counts: List[int], max_categories: int) -> float:
        """Calculate normalized entropy (0-1)."""
        if not counts or sum(counts) == 0:
            return 0.0
        
        total = sum(counts)
        probs = [c / total for c in counts if c > 0]
        
        entropy = -sum(p * np.log2(p) for p in probs)
        max_entropy = np.log2(max_categories)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _check_quality(
        self,
        manifest,
        dataset_path: Path,
        report: ValidationReport,
    ):
        """Check file quality."""
        valid_count = 0
        
        for sample in manifest.samples:
            file_path = dataset_path / sample.file_path
            
            if file_path.exists():
                # Try to verify file integrity
                try:
                    if sample.file_type == 'midi':
                        self._verify_midi(file_path)
                    elif sample.file_type == 'audio':
                        self._verify_audio(file_path)
                    valid_count += 1
                except Exception as e:
                    report.add_issue(
                        "error",
                        "quality",
                        f"Corrupt file: {sample.file_path}",
                        {"error": str(e)},
                    )
                    report.corrupt_files.append(sample.file_path)
        
        if report.total_samples > 0:
            report.quality_score = valid_count / report.total_samples
    
    def _verify_midi(self, path: Path) -> bool:
        """Verify MIDI file is readable."""
        try:
            import mido
            mid = mido.MidiFile(str(path))
            return len(mid.tracks) > 0
        except ImportError:
            return True  # Skip if mido not available
        except Exception:
            raise ValueError(f"Invalid MIDI file: {path}")
    
    def _verify_audio(self, path: Path) -> bool:
        """Verify audio file is readable."""
        try:
            import librosa
            y, sr = librosa.load(str(path), duration=1)
            return len(y) > 0
        except ImportError:
            return True  # Skip if librosa not available
        except Exception:
            raise ValueError(f"Invalid audio file: {path}")
    
    def _validate_from_files(self, dataset_path: Path, report: ValidationReport):
        """Validate dataset from files when no manifest exists."""
        # Count files
        midi_files = list(dataset_path.glob("**/*.mid")) + list(dataset_path.glob("**/*.midi"))
        audio_files = list(dataset_path.glob("**/*.wav")) + list(dataset_path.glob("**/*.mp3"))
        
        report.total_samples = len(midi_files) + len(audio_files)
        
        if report.total_samples == 0:
            report.add_issue(
                "error",
                "structure",
                "No MIDI or audio files found in dataset",
            )
            return
        
        # Check for annotation files
        annotation_files = list(dataset_path.glob("**/*.json"))
        if len(annotation_files) < report.total_samples:
            report.add_issue(
                "warning",
                "quality",
                f"Missing annotations: {report.total_samples} samples but only {len(annotation_files)} annotation files",
            )
        
        # Try to infer categories from directory structure
        for subdir in (dataset_path / "raw").iterdir() if (dataset_path / "raw").exists() else []:
            if subdir.is_dir():
                count = len(list(subdir.glob("*")))
                if count > 0:
                    report.category_counts[subdir.name] = count
    
    def _generate_recommendations(self, report: ValidationReport):
        """Generate recommendations based on validation results."""
        # Balance recommendations
        if report.balance_score < 0.5:
            low_cats = [c for c, n in report.category_counts.items() 
                       if n < report.max_category_size * 0.5]
            if low_cats:
                report.recommendations.append(
                    f"Add more samples to underrepresented categories: {', '.join(low_cats)}"
                )
            report.recommendations.append(
                "Consider using data augmentation to balance categories"
            )
        
        # Diversity recommendations
        if report.diversity_score < 0.3:
            if len(report.key_distribution) < 6:
                report.recommendations.append(
                    "Add samples in more keys (transpose existing samples)"
                )
            if len(report.tempo_distribution) < 5:
                report.recommendations.append(
                    "Add samples at varied tempos"
                )
        
        # Quality recommendations
        if report.quality_score < 0.95:
            if report.corrupt_files:
                report.recommendations.append(
                    f"Fix or remove {len(report.corrupt_files)} corrupt files"
                )
        
        if report.missing_annotations:
            report.recommendations.append(
                f"Add annotations for {len(report.missing_annotations)} samples"
            )
        
        # Size recommendations
        for cat, count in report.category_counts.items():
            if count < 100:
                report.recommendations.append(
                    f"Category '{cat}' needs at least {100 - count} more samples for reliable training"
                )
        
        # Split recommendations
        total = report.train_count + report.val_count + report.test_count
        if total > 0:
            train_ratio = report.train_count / total
            if train_ratio < 0.7:
                report.recommendations.append(
                    "Consider using 80/10/10 train/val/test split for better training"
                )


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_dataset(dataset_path: Path, **kwargs) -> ValidationReport:
    """Validate a dataset."""
    validator = DatasetValidator(**kwargs)
    return validator.validate(dataset_path)


def check_balance(dataset_path: Path) -> Dict[str, int]:
    """Quick check of category balance."""
    report = validate_dataset(dataset_path)
    return report.category_counts


def check_diversity(dataset_path: Path) -> Dict[str, Any]:
    """Quick check of dataset diversity."""
    report = validate_dataset(dataset_path)
    return {
        'diversity_score': report.diversity_score,
        'keys': report.key_distribution,
        'modes': report.mode_distribution,
        'tempos': report.tempo_distribution,
    }


def print_dataset_stats(dataset_path: Path):
    """Print comprehensive dataset statistics."""
    report = validate_dataset(dataset_path)
    report.print_summary()

