# Copilot Instructions
## Architecture Snapshot
- Kelly pairs Python orchestration with a performance C++ engine; [music_brain](music_brain) manages intent analysis, session logic, FastAPI, and CLI flows, while real-time DSP lives in [cpp_music_brain](cpp_music_brain) and bridges into [penta_core](penta_core) via pybind11.
- Emotional processing follows the Wound → Emotion → Rule-break pipeline documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/PHASE3_DESIGN.md](docs/PHASE3_DESIGN.md); maintain consistency with emotion JSON assets in [data](data) and the runtime copy in [emotion_thesaurus](emotion_thesaurus).
- Real-time constraints in the C++ layer forbid allocations, locks, or logging inside audio callbacks; consult [docs/PHASE3_DESIGN.md](docs/PHASE3_DESIGN.md) before touching harmony, groove, or diagnostics modules.
- Python song generation stitches together [music_brain/session](music_brain/session), [music_brain/arrangement](music_brain/arrangement), and [music_brain/groove](music_brain/groove); generators expect structured dataclasses and should return to_dict()-friendly payloads.

## Core Workflows
- Install tooling with pip install -e . from the repo root (Python 3.11+); enable optional extras as needed via midikompanion[dev] and midikompanion[research] defined in [pyproject.toml](pyproject.toml).
- Launch the API server with python -m music_brain.api; ensure [emotion_thesaurus](emotion_thesaurus) exists (copy JSONs from [data](data) as described in [BUILD_API_SERVER.md](BUILD_API_SERVER.md)) and watch the structured logs emitted by music_brain.api.
- Exercise the CLI using python -m music_brain.cli commands; modules are lazily imported so add new subcommands by wiring lightweight factories similar to existing get_*_module helpers.
- Run Python tests with pytest tests tests_music-brain tests_penta-core; the suite assumes emotion assets are present and may call into pybind-backed modules.
- Build the C++ engine with cmake -B build -DCMAKE_BUILD_TYPE=Debug -DPENTA_BUILD_TESTS=ON followed by cmake --build build and ctest as captured in [docs/BUILD.md](docs/BUILD.md); target-specific builds (plugins, bindings, benchmarks) rely on the same cache.

## Coding Patterns
- File IO should flow through [music_brain/utils/path_utils.py](music_brain/utils/path_utils.py) safe_path helpers to normalize cross-platform paths (used extensively inside CLI and API routines).
- Metrics and rate limiting are centralized in [music_brain/metrics.py](music_brain/metrics.py) and middleware under [music_brain/middleware](music_brain/middleware); reuse these utilities when adding endpoints to keep instrumentation consistent.
- Session logic depends on the state machines in [music_brain/session](music_brain/session) (phases, interrogator, rule-breaking teacher); new conversational features should update both context objects and the Quick Questions catalog.
- Harmony and groove analysis algorithms live in [penta_core/harmony](penta_core/harmony) and [penta_core/groove](penta_core/groove); Python surfaces wrap underlying C++ implementations, so respect existing method signatures to avoid breaking bindings.
- C++ audio paths live under [cpp_music_brain/src](cpp_music_brain/src) with tests in [cpp_music_brain/tests](cpp_music_brain/tests); follow JUCE-style separation of `processBlock` (real-time) and editor/UI threads, and keep readerwriterqueue usage for message passing.

## Reference Guides
- Use [docs/COMPREHENSIVE_DEVELOPMENT_WORKFLOW.md](docs/COMPREHENSIVE_DEVELOPMENT_WORKFLOW.md) for end-to-end process expectations, including sprint docs and data prerequisites.
- Cross-check plugin specifics, lock-free patterns, and OSC messaging in [docs/JUCE_PLUGIN_ARCHITECTURE.md](docs/JUCE_PLUGIN_ARCHITECTURE.md) and the OSC section of [docs/PHASE3_DESIGN.md](docs/PHASE3_DESIGN.md).
- Confirm contribution standards, formatting defaults (Black, Ruff, clang-format), and mypy settings in [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) and [pyproject.toml](pyproject.toml) before submitting changes.
- Data science and ML assets live in [ml_training](ml_training) and [ml_framework](ml_framework); these directories are large, so prefer targeted edits guided by their local README files surfaced by the repository search.
- When unsure about prior decisions, review the historical context in [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md), [docs/PRODUCTION_READY.md](docs/PRODUCTION_READY.md), and the sprint summaries inside [docs](docs).
