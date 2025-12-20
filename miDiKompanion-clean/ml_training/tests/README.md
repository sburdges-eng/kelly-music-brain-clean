# ML Training Test Suite

Comprehensive test suite for the ML training infrastructure, covering unit tests, integration tests, and performance benchmarks.

## Test Structure

```
tests/
├── __init__.py
├── unit/                    # Unit tests for individual components
│   ├── test_training_utils.py      # EarlyStopping, TrainingMetrics, CheckpointManager
│   ├── test_dataset_loaders.py    # Dataset loader functionality
│   ├── test_rtneural_export.py    # RTNeural JSON export validation
│   └── test_model_architectures.py # Model architecture specifications
├── integration/             # Integration tests for full workflows
│   ├── test_full_pipeline.py      # Training → Export → Validation pipeline
│   ├── test_async_inference.py    # Async inference functionality
│   └── test_roundtrip.py          # Export/import roundtrip validation
└── performance/             # Performance and benchmark tests
    ├── test_full_pipeline_performance.py  # End-to-end performance
    ├── test_inference_latency.py          # Individual model latency
    └── test_memory_usage.py              # Memory footprint validation
```

## Running Tests

### Run All Tests

```bash
cd ml_training
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Performance tests only
pytest tests/performance/ -v
```

### Run Specific Test Files

```bash
# Run a specific test file
pytest tests/unit/test_training_utils.py -v

# Run a specific test class
pytest tests/unit/test_training_utils.py::TestEarlyStopping -v

# Run a specific test method
pytest tests/unit/test_training_utils.py::TestEarlyStopping::test_early_stopping_restores_best_weights -v
```

### Run with Coverage

```bash
# Install pytest-cov if not already installed
pip install pytest-cov

# Run tests with coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ -v --cov=. --cov-report=html
# Open htmlcov/index.html in a browser
```

### Run with Test Runner Script

A convenient test runner script is available:

```bash
# Run all tests
./run_tests.sh all

# Run specific categories
./run_tests.sh unit          # Unit tests only
./run_tests.sh integration   # Integration tests only
./run_tests.sh performance   # Performance tests only
```

The script:

- Checks for pytest installation
- Provides colored output (green for success, red for failures)
- Excludes slow tests by default for unit/integration
- Shows clear error messages

### Run with unittest (Alternative)

Tests are written using `unittest` framework, so they can also be run with:

```bash
python -m unittest discover tests/ -v
```

## Test Coverage

### Test Count Summary

**Total: 61 tests** across all categories:

- **Unit Tests**: 33 tests
  - `test_training_utils.py`: 10 tests (EarlyStopping: 3, TrainingMetrics: 5, CheckpointManager: 2, Evaluation: 2)
  - `test_dataset_loaders.py`: 6 tests (factory, missing files, synthetic datasets, value ranges, DataLoader integration)
  - `test_rtneural_export.py`: 9 tests (JSON validation, metadata, layers, LSTM splitting, weights, activations, parameter counts)
  - `test_model_architectures.py`: 8 tests (all 5 models: specs, output ranges, batch processing, NaN/Inf detection)

- **Integration Tests**: 12 tests
  - `test_full_pipeline.py`: 4 tests (training→export→validation, C++ specs matching, RTNeural structure, Python inference)
  - `test_async_inference.py`: 4 tests (placeholder for async pipeline, non-blocking submit, result availability, result retrieval)
  - `test_roundtrip.py`: 4 tests (Python inference consistency, export structure preservation, JSON loading, output ranges)

- **Performance Tests**: 16 tests
  - `test_full_pipeline_performance.py`: 4 tests (latency, throughput, output validity, consistency)
  - `test_inference_latency.py`: 6 tests (5 individual models + summary, each with mean/P95/max stats)
  - `test_memory_usage.py`: 6 tests (5 individual models + total memory usage)

### Code Coverage

Current coverage (as of last verification):

- **Test files**: 96-99% coverage
- **training_utils.py**: 52% coverage (core functionality covered: EarlyStopping, TrainingMetrics, CheckpointManager)
- **Overall**: 34% coverage (expected for training scripts - many utility scripts not directly tested)

### Coverage Goals

- **Unit tests**: 80%+ coverage for training utilities and dataset loaders
- **Integration tests**: Cover full pipeline (training → export → validation)
- **Performance tests**: Verify latency <50ms, memory <4.5MB targets

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation:

**test_training_utils.py** (10 tests):

- `TestEarlyStopping`: Patience mechanism, min_delta respect, **weight restoration** (verifies deep copy fix)
- `TestTrainingMetrics`: Update tracking, best epoch calculation, JSON/CSV export, plotting (matplotlib optional)
- `TestCheckpointManager`: Save/load checkpoints, best model tracking, cleanup
- `TestEvaluationFunctions`: Cosine similarity computation, model evaluation

**test_dataset_loaders.py** (6 tests):

- Dataset factory function (`create_dataset`)
- Missing file handling (graceful error handling)
- Synthetic dataset creation (`SyntheticEmotionDataset`, `SyntheticMelodyDataset`)
- Value range validation (emotion: [-1,1], notes: [0,1] probabilities)
- DataLoader integration (batch processing)

**test_rtneural_export.py** (9 tests):

- JSON format validation (valid JSON, required keys)
- Metadata structure (model_name, input_size, output_size, parameter_count, framework, export_version)
- Layer structure (dense/LSTM types, in_size/out_size)
- **LSTM weight splitting** (4 gates: input, forget, cell, output for weights_ih/weights_hh)
- Dense layer weight shapes (correct dimensions)
- Activation function validation (tanh, relu, sigmoid, softmax)
- Parameter count matching (model vs exported)
- Input/output size matching (metadata vs layers)

**test_model_architectures.py** (8 tests):

- All 5 models match C++ ModelSpec definitions (input/output dimensions, parameter counts ±5% tolerance)
- Output range validation (tanh: [-1,1], sigmoid/softmax: [0,1])
- Batch processing (1, 4, 16, 32 batch sizes)
- NaN/Inf detection (all models produce valid outputs)

### Integration Tests (`tests/integration/`)

Test complete workflows:

**test_full_pipeline.py** (4 tests):

- `test_training_export_validation`: Complete workflow (Train → Export → Validate JSON structure)
- `test_exported_models_match_specs`: Verify exported models match C++ ModelSpec definitions (input/output/params)
- `test_rtneural_json_structure`: Validate RTNeural JSON structure (layers array, metadata object, layer types)
- `test_python_inference_matches_export`: Verify Python model inference produces valid outputs (no NaN/Inf, correct shapes)

**test_async_inference.py** (4 tests - **Placeholder**):

- `test_async_inference_placeholder`: Placeholder for async inference pipeline (awaiting C++ `AsyncMLPipeline` implementation)
- `test_non_blocking_submit`: Verify feature submission is non-blocking
- `test_result_availability_check`: Verify result availability can be checked without blocking
- `test_result_retrieval`: Verify results can be retrieved without blocking

**test_roundtrip.py** (4 tests):

- `test_python_inference_consistency`: Verify Python model inference is deterministic
- `test_export_preserves_structure`: Verify export preserves model structure and weights
- `test_json_can_be_loaded`: Verify exported JSON can be loaded and parsed correctly
- `test_model_output_ranges`: Verify model outputs are in expected ranges after export/load

### Performance Tests (`tests/performance/`)

Benchmark performance characteristics:

**test_full_pipeline_performance.py** (4 tests):

- `test_full_pipeline_latency`: Full pipeline latency (<50ms target, 100 iterations with 10 warmup)
- `test_pipeline_throughput`: Pipeline throughput (>20 samples/second target, 1000 samples)
- `test_pipeline_output_validity`: Verify all outputs are valid (correct shapes, no NaN/Inf)
- `test_pipeline_consistency`: Verify pipeline produces deterministic results (identical outputs for same input)

**test_inference_latency.py** (6 tests):

- Individual model latency tests (5 models): Mean, median, P95, P99, max statistics
- `test_all_models_latency_summary`: Total pipeline latency summary (<50ms target)
- Each test: 1000 iterations with 100 warmup, target <10ms per model (with 2x tolerance for CI variability)

**test_memory_usage.py** (6 tests):

- Individual model memory tests (5 models): Parameter counts and memory in MB
- `test_total_memory_usage`: Total memory usage (<4.5MB target, actual ~4.39MB)
- Memory calculation: 4 bytes per float32 parameter
- Thresholds: 2.5MB for EmotionRecognizer/MelodyTransformer, 0.5MB for others, 4.5MB total

## Adding New Tests

### Test File Structure

```python
#!/usr/bin/env python3
"""
Test Description
================
Brief description of what this test file covers.
"""

import unittest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from module_to_test import ClassToTest


class TestClassName(unittest.TestCase):
    """Test class description."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize test data, models, etc.

    def tearDown(self):
        """Clean up after tests."""
        # Clean up resources if needed

    def test_specific_feature(self):
        """Test description."""
        # Test implementation
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
```

### Test Naming Conventions

- Test classes: `TestClassName` (PascalCase)
- Test methods: `test_feature_name` (snake_case, must start with `test_`)
- Use descriptive names that explain what is being tested

### Best Practices

1. **Test Isolation**: Each test should be independent and not rely on other tests
2. **Clean Setup/Teardown**: Use `setUp()` and `tearDown()` for fixtures
3. **Clear Assertions**: Use descriptive assertion messages
4. **Test Edge Cases**: Include tests for error conditions and edge cases
5. **Mock External Dependencies**: Use mocks for file I/O, network calls, etc.

## Debugging Test Failures

### Verbose Output

```bash
# Show detailed output
pytest tests/ -v -s

# Show print statements
pytest tests/ -v -s --capture=no
```

### Traceback Options

```bash
# Short traceback (default)
pytest tests/ -v --tb=short

# Long traceback
pytest tests/ -v --tb=long

# Line-by-line traceback
pytest tests/ -v --tb=line

# No traceback (just summary)
pytest tests/ -v --tb=no
```

### Run Last Failed Tests

```bash
# Run only tests that failed in the last run
pytest tests/ --lf

# Run failed tests first, then others
pytest tests/ --ff
```

### Debug Specific Test

```bash
# Drop into debugger on failure
pytest tests/ -v --pdb

# Drop into debugger at start of test
pytest tests/ -v --trace
```

## CI/CD Integration

Tests are automatically run in GitHub Actions on:

- Push to `main`, `develop`, or `miDiKompanion` branches
- Pull requests to these branches
- Changes to `ml_framework/**` or `ml_training/**` paths
- Manual workflow dispatch

See `.github/workflows/ml_tests.yml` for configuration details.

### CI Test Matrix

- **Python versions**: 3.9, 3.11
- **Operating systems**: Ubuntu Latest, macOS Latest
- **Dependencies**: Installed from `ml_framework/requirements.txt` and `ml_training/requirements.txt`

## Known Issues and Limitations

### Optional Dependencies

Some dataset loaders require optional dependencies (e.g., `music21`). Tests use `@unittest.skipUnless` to handle missing dependencies gracefully.

### Performance Test Variability

Performance tests may show some variability due to system load. The tests use reasonable thresholds to account for this.

### Test Data

Most tests use synthetic data generated on-the-fly. Real dataset tests are skipped if datasets are not available.

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the ml_training directory
cd ml_training

# Check Python path
python -c "import sys; print(sys.path)"

# Install dependencies
pip install -r requirements.txt
```

### Test Discovery Issues

```bash
# Clear pytest cache
rm -rf .pytest_cache

# Run with explicit path
pytest tests/unit/test_training_utils.py -v
```

### Memory Issues

If tests fail due to memory issues:

```bash
# Run tests one at a time
pytest tests/ -v --forked

# Or run specific test categories separately
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v
```

## Test Implementation Details

### Key Test Features

1. **EarlyStopping Weight Restoration Fix** ✅
   - Test: `test_training_utils.py::TestEarlyStopping::test_early_stopping_restores_best_weights`
   - Verifies that `EarlyStopping._save_checkpoint()` uses deep copy (`.clone()`) to prevent weight reference sharing
   - Location: `training_utils.py:289-296`

2. **Full Pipeline Dimension Fix** ✅
   - Test: `test_full_pipeline_performance.py::TestFullPipelinePerformance::run_full_pipeline`
   - Verifies correct tensor slicing: `emotion[:, :16]` (feature dimension, not batch)
   - Creates proper 32-dim compact context for DynamicsEngine
   - Location: `test_full_pipeline_performance.py:59`

3. **Memory Threshold Updates** ✅
   - Test: `test_memory_usage.py::TestMemoryUsage`
   - Thresholds: 2.5MB for EmotionRecognizer/MelodyTransformer, 4.5MB total
   - Actual: ~4.39MB total (within target)

### Test Fixtures and Setup

All tests use proper setup/teardown:

- `setUp()`: Creates temporary directories, initializes models
- `tearDown()`: Cleans up temporary files and directories
- Uses `tempfile.mkdtemp()` for isolated test environments
- Tests are independent (no shared state)

### Optional Dependencies Handling

Tests gracefully handle missing optional dependencies:

- Dataset loaders: `@unittest.skipUnless(DATASET_LOADERS_AVAILABLE, ...)`
- Matplotlib: Try/except blocks for plotting tests
- Real datasets: Tests skip if datasets not available

## Related Documentation

- [Main README](../README.md) - Training guide and usage
- [Training Utils](../training_utils.py) - Training utility functions (EarlyStopping, TrainingMetrics, CheckpointManager)
- [Dataset Loaders](../dataset_loaders.py) - Dataset loading functionality (DEAM, PMEmo, Lakh MIDI, MAESTRO, Groove MIDI)
- [Model Architectures](../train_all_models.py) - Model definitions (5 models: EmotionRecognizer, MelodyTransformer, HarmonyPredictor, DynamicsEngine, GroovePredictor)
- [Test Suite Status](../TEST_SUITE_STATUS.md) - Verification status and coverage details
