# Sprint 1 – Core Testing & Quality

## Overview

Sprint 1 focuses on establishing the foundational testing infrastructure and ensuring code quality across the Python-based music-brain and penta-core components.

## Objectives

1. **Test Coverage**: Achieve comprehensive test coverage for core modules
2. **Quality Assurance**: Validate all critical functionality works as expected
3. **CI Integration**: Ensure automated testing runs on every commit

## Test Scope

### music-brain Tests (`tests_music-brain/`)

- **Basic Functionality** (`test_basic.py`)
  - Core data structures
  - MIDI I/O operations
  - Utility functions

- **Bridge Integration** (`test_bridge_integration.py`)
  - HarmonyPlan class validation
  - TherapySession API
  - MIDI graceful degradation

- **CLI Commands** (`test_cli_commands.py`, `test_cli_flow.py`)
  - All 6 CLI commands (extract, apply, analyze, diagnose, reharm, intent)
  - End-to-end command workflows
  - Error handling and edge cases

- **Engine Components** (`test_comprehensive_engine.py`)
  - Orchestrator pipeline
  - Genre detection
  - Synesthesia fallback system

- **Groove System** (`test_groove_engine.py`, `test_groove_extractor.py`)
  - Groove extraction from MIDI
  - Groove application with templates
  - Genre-specific groove patterns

- **Intent Processing** (`test_intent_processor.py`, `test_intent_schema.py`)
  - Three-phase intent schema validation
  - Rule-breaking logic
  - Emotional-to-musical mapping

- **DAW Integration** (`test_daw_integration.py`)
  - Ableton bridge OSC/MIDI communication
  - Logic Pro X workflow
  - Plugin parameter mapping

- **Error Handling** (`test_error_paths.py`)
  - Graceful error recovery
  - Invalid input handling
  - Edge case validation

### penta-core Tests (`tests_penta-core/`)

- **Python API Validation**
  - Module import tests
  - API wrapper functionality
  - Integration with music-brain

## Success Criteria

- ✅ All `tests_music-brain/` tests pass
- ✅ `tests_penta-core/` tests pass or gracefully skip if C++ not built
- ✅ Test coverage > 80% for critical modules
- ✅ No breaking changes to existing functionality

## Workflow Configuration

```yaml
sprint1_test_quality:
  name: "Sprint 1 – Core Testing & Quality"
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.9'
    - run: pip install -e .[dev,audio,all]
    - run: pytest tests_music-brain/ -v
    - run: pytest tests_penta-core/ -v || true
```

## Key Deliverables

1. **Comprehensive test suite** covering all major modules
2. **CI/CD pipeline** running tests on every commit
3. **Test documentation** explaining what each test validates
4. **Bug reports** for any failing tests with reproduction steps

## Dependencies

- Python 3.9+
- pytest
- All required packages from `requirements_music-brain.txt`
- Optional: audio packages (librosa, soundfile) for audio tests

## Related Documentation

- [Comprehensive System Requirements](comprehensive-system-requirements.md)
- [Phase 3 Implementation Summary](PHASE3_SUMMARY.md)
- [Build Instructions](BUILD.md)

## Notes

- Tests should run in < 60 seconds for fast feedback
- Use `pytest -v` for verbose output
- Use `pytest -k <pattern>` to run specific tests
- C++ tests may be skipped if penta-core C++ engine not built
