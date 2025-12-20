# Kelly Music Brain - Comprehensive Development Workflow

## Project Overview
Kelly Music Brain is an AI-powered music production system integrating multiple components: music theory engine (penta-core), DAW integration (iDAW), and intelligent music brain capabilities.

## Development Workflow Structure

### 1. Planning & Design Phase

#### 1.1 Requirements Gathering
- Review existing documentation in `/docs`, `/docs_music-brain`, `/docs_penta-core`
- Consult:
  - `PROJECT_ROADMAP.md` for long-term vision
  - `ROADMAP_18_MONTHS.md` for timeline
  - `PHASE_2_PLAN.md` for current phase objectives
  - `REFINED_PRIORITY_PLANS.md` for prioritized tasks

#### 1.2 Architecture Planning
- Reference `MAIN_DOCUMENTATION.md` for system overview
- Review `Integration_Architecture.md` for component integration
- Check `cpp_audio_architecture.md` for audio processing design
- Consult `hybrid_development_roadmap.md` for Python/C++ strategy

#### 1.3 Sprint Planning
- Use sprint documents: `Sprint_1_Core_Testing_and_Quality.md` through `Sprint_8_Enterprise_Ecosystem.md`
- Reference `COMPREHENSIVE_TODO.md` for task breakdown
- Check `TODO_BEGIN_ALL.txt` for immediate priorities
- Review `progress.md` for current status

### 2. Development Environment Setup

#### 2.1 Platform-Specific Setup
**macOS:**
```bash
./install_macos.sh
./build_macos_app.sh
```

**Linux:**
```bash
./install_linux.sh
```

**Windows:**
```powershell
./install_windows.ps1
```

#### 2.2 Development Tools
- **Python Environment:**
  - Install dependencies: `pip install -r requirements.txt`
  - Music Brain specific: `pip install -r requirements_music-brain.txt`
  - Penta Core specific: `pip install -r requirements_penta-core.txt`

- **C++ Build System:**
  - CMake configuration: `CMakeLists.txt`
  - File I/O module: `CMakeLists_fileio.txt`
  - Review `BUILD.md` and `JUCE_SETUP.md`

- **Docker Environment:**
  - Use `docker-compose.yml` for containerized development
  - Training environment: `Dockerfile.training` or `Dockerfile.training.cuda`
  - Analyzer: `Dockerfile.kb-analyzer`

### 3. Development Workflow

#### 3.1 Feature Development Process

**Step 1: Issue/Task Selection**
1. Review `COMPREHENSIVE_TODO.md` and `TODO_ANALYSIS.md`
2. Check `blockers.md` for known issues
3. Reference `dependencies.md` for component dependencies

**Step 2: Branch Strategy**
```bash
git checkout -b feature/<feature-name>
# or
git checkout -b fix/<bug-name>
# or
git checkout -b sprint/<sprint-number>-<task-name>
```

**Step 3: Implementation**

**For Python Modules:**
- Core music engine: `/music_brain/` or `/penta_core/`
- Review existing modules:
  - `chord.py`, `progression.py`, `harmony.py` - Music theory
  - `feel.py`, `groove.py` - Feel and rhythm
  - `emotional_mapping.py`, `emotion_thesaurus.py` - Emotion processing
  - `intent_processor.py`, `intent_schema.py` - Intent parsing
  - `orchestrator.py`, `generator.py` - Music generation

**For C++ Components:**
- Audio processing: `/cpp_music_brain/`
- VST Plugin: Reference `VST_PLUGIN_IMPLEMENTATION_PLAN.md`
- Core files: `PluginProcessor.cpp`, `PluginEditor.cpp`
- Audio components: `VoiceProcessor.cpp`, `WavetableSynth.cpp`

**For DAW Integration:**
- Logic Pro: `logic_pro.py`, `Logic Pro Settings.md`
- iDAW Pipeline: `idaw_complete_pipeline.py`
- Library integration: `idaw_library_integration.py`
- UI components: `idaw_ableton_ui.py`

#### 3.2 Code Standards

**Python:**
- Follow PEP 8 guidelines
- Use type hints where applicable
- Document with docstrings
- Leverage existing utilities:
  - `Logger.py` for logging
  - `cache.py` for caching
  - `metrics.py` for performance tracking

**C++:**
- Follow JUCE conventions
- Document with Doxygen comments (see `Doxyfile`)
- Use modern C++ practices
- Reference existing implementations in `/src/`

### 4. Testing Protocol

#### 4.1 Test Suite Structure
- Unit tests: `/tests/`, `/tests_music-brain/`, `/tests_penta-core/`
- Performance tests: `test_performance.py`, `test_performance_optimizations.py`
- Basic integration: `test_basic.py`
- Test runner: `RunTests.cpp` for C++ components

#### 4.2 Testing Workflow
```bash
# Python tests
pytest tests/
pytest tests_music-brain/
pytest tests_penta-core/

# Performance benchmarks
python test_performance.py

# C++ tests
# Build and run via CMake test target
```

#### 4.3 Test Documentation
- Review `TEST_README.md` for test guidelines
- Check `TEST_HARNESS_README.md` for harness usage
- Reference `TRIAL_RUN_CHECKLIST.md` for integration testing

### 5. Data & Knowledge Management

#### 5.1 Knowledge Bases
- Music theory: `scales_database.json` (1.9MB+ comprehensive)
- Chord progressions: `chord_progressions.json`, `chord_progression_families.json`
- Emotional mapping: `scale_emotional_map.json`, emotion files (`happy.json`, `sad.json`, etc.)
- Genre templates: `genre_pocket_maps.json`, `genre_templates.py`
- Vernacular: `vernacular_database.json`, `music_vernacular_database.md`

#### 5.2 Sample Library Management
- Downloader: `sample_downloader.py`
- Cataloger: `audio_cataloger.py`
- Analyzer: `audio_analyzer_starter.py`
- Freesound integration: `FREESOUND_PACK_LIST.md`

### 6. Integration & Build Process

#### 6.1 Component Integration
**DAW Integration:**
1. Review `DAW_INTEGRATION.md` and `DAIW_INTEGRATION.md`
2. Implement using `BridgeClient.cpp` for communication
3. Test with `daiw_listener_public.py`

**Music Brain Server:**
```bash
python brain_server.py
# or
python server.py
```

**API Layer:**
```bash
python api.py  # REST API
python app.py  # Flask application
```

#### 6.2 Build Verification
- Check `BUILD_COMPLETE.md` for completion criteria
- Review `INTEGRATION_COMPLETE.md` for integration status
- Validate with `validate_merge.py`

### 7. Documentation Requirements

#### 7.1 Code Documentation
- Update relevant module documentation
- Add examples to `/examples/`, `/examples_music-brain/`, `/examples_penta-core/`
- Update API documentation in `/docs/`

#### 7.2 User Documentation
**Knowledge Base:**
- Obsidian vault: `/vault/`
- Templates: `/Templates/`
- Guides: Production, songwriting, theory guides (multiple .md files)
- Tools: `AUDIO_ANALYZER_TOOLS.md`, `GROOVE_MODULE_GUIDE.md`

**AI Assistant Setup:**
- `AI Assistant Setup Guide.md`
- `ChatGPT_Custom_GPT_Instructions.md`
- `CLAUDE_AGENT_GUIDE.md`
- `Gemini_Setup.md`

#### 7.3 Change Documentation
- Update `CHANGE_LIST.md` with changes
- Add breaking changes to `BREAKING_CHANGES.md`
- Document fixes in relevant summaries (`BRIDGE_FIXS_SUMMARY.md`, etc.)

### 8. Quality Assurance

#### 8.1 Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass (unit, integration, performance)
- [ ] Documentation updated
- [ ] No security vulnerabilities (check `SECURITY_SUMMARY.md`)
- [ ] Performance benchmarks met (check `PERFORMANCE_SUMMARY.md`)
- [ ] Dependencies documented in `dependencies.md`

#### 8.2 Performance Validation
- Run `test_performance.py`
- Check against `PERFORMANCE_IMPROVEMENTS.md`
- Review `PERFORMANCE_OPTIMIZATIONS.md` for optimization strategies
- Validate with benchmarks in `/benchmarks/`

#### 8.3 Troubleshooting
- Consult `TROUBLESHOOTING.md` for common issues
- Review `critical_broken_code_analysis.md` for known problems
- Check `blockers.md` for current blockers

### 9. Release Process

#### 9.1 Pre-Release Checklist
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Examples working
- [ ] Performance benchmarks met
- [ ] Security review complete
- [ ] CHANGELOG updated

#### 9.2 Version Management
- Update `VERSION` file
- Update `pyproject.toml` or platform-specific configs
- Tag release in git

#### 9.3 Deployment
**Python Package:**
```bash
python setup.py sdist bdist_wheel
```

**macOS App:**
```bash
./build_macos_app.sh
python setup_macos.py
```

**Docker:**
```bash
docker-compose build
docker-compose up
```

### 10. Specialized Workflows

#### 10.1 Music Generation Workflow
```python
# Use existing generators
from generator import MusicGenerator
from intent_processor import IntentProcessor
from orchestrator import Orchestrator

# Process user intent
intent = IntentProcessor.process("Create uplifting pop song")

# Generate music
music = MusicGenerator.generate(intent)

# Orchestrate
final = Orchestrator.orchestrate(music)
```

#### 10.2 Audio Analysis Workflow
```python
from analyzer import AudioAnalyzer
from extractor import FeatureExtractor

# Analyze audio
features = AudioAnalyzer.analyze('track.wav')
feel = FeatureExtractor.extract_feel(features)
```

#### 10.3 DAW Integration Workflow
1. Launch DAW listener: `python daiw_listener_public.py`
2. Configure in DAW using `Logic Pro Settings.md`
3. Use menu bar app: `python daiw_menubar.py`

#### 10.4 Custom Agent Workflow
- Review `agent.py` for agent framework
- Use `cpp_planner.py` for C++ planning tasks
- Check `ai_specializations.py` for specialized behaviors

### 11. Mobile & Web Development

#### 11.1 Mobile Apps
- iOS: `/iOS/` directory
- macOS: `/macOS/` directory
- Cross-platform: `/mobile/` directory

#### 11.2 Web Interface
- Frontend: `/src/` with React/TypeScript
- Vite config: `vite.config.ts`
- Tailwind: `tailwind.config.js`
- API: `api.py`

### 12. Continuous Improvement

#### 12.1 Sprint Reviews
- Review sprint completion: `SPRINT_5_COMPLETION_SUMMARY.md`
- Update roadmap: `PROJECT_ROADMAP.md`
- Plan next sprint using sprint templates

#### 12.2 Technical Debt
- Track in `TODO_COMPLETION_SUMMARY.md`
- Review `WIP_REVIEW_SUMMARY.md` for work in progress
- Clean up per `CLEANUP_SUMMARY.md` and `FILE_CLEANUP_SUMMARY.md`

#### 12.3 Optimization Cycles
- Regular performance reviews
- Refactoring sessions
- Dependency updates
- Security audits

### 13. Collaboration & Communication

#### 13.1 Team Coordination
- Use `Collaboration Template.md` for sessions
- Document in `Session Notes Template.md`
- Track progress in `Weekly Review Template.md`

#### 13.2 Knowledge Sharing
- Maintain Learning Topics: `Learning Topic Template.md`
- Share insights via `Teaching` module (`teaching.py`, `teaching_tools.py`)
- Update guides and references

### 14. Reference Quick Links

**Core Documentation:**
- [README](README.md) | [Main Docs](MAIN_DOCUMENTATION.md)
- [Install Guide](INSTALL.md) | [Build Guide](BUILD.md)
- [Quickstart](QUICKSTART.md) | [Phase 2 Quickstart](PHASE_2_QUICKSTART.md)

**Development:**
- [Project Roadmap](PROJECT_ROADMAP.md) | [18-Month Plan](ROADMAP_18_MONTHS.md)
- [TODO Analysis](TODO_ANALYSIS.md) | [Dependencies](dependencies.md)

**Integration:**
- [DAW Integration](DAW_INTEGRATION.md) | [VST Plan](VST_PLUGIN_IMPLEMENTATION_PLAN.md)

**Guides:**
- Production workflows in `/Production_Workflows/`
- Songwriting guides in `/Songwriting_Guides/`
- Theory reference in `/Theory_Reference/`

## Workflow Automation

### Daily Development Cycle
1. Pull latest changes
2. Review daily tasks from `COMPREHENSIVE_TODO.md`
3. Check `blockers.md` for issues
4. Develop feature/fix
5. Run tests
6. Update documentation
7. Commit with descriptive message
8. Push and create PR

### Weekly Cycle
1. Sprint planning review
2. Progress update in `progress.md`
3. Performance benchmark check
4. Documentation review
5. Team sync
6. Weekly review using template

### Sprint Cycle (2-week sprints)
1. Sprint planning
2. Daily standups
3. Mid-sprint review
4. Sprint completion review
5. Retrospective
6. Sprint summary documentation

---

## Getting Started Today

**First Time Setup:**
```bash
# 1. Clone repository
git clone <repo-url>
cd kelly-music-brain-clean

# 2. Install dependencies (macOS example)
./install_macos.sh

# 3. Run tests to verify
pytest tests/

# 4. Start development server
python brain_server.py

# 5. Review START_HERE.txt for orientation
cat START_HERE.txt
```

**Begin Development:**
1. Read `START_HERE.txt`
2. Review `PROJECT_ROADMAP.md`
3. Check current sprint document
4. Pick task from `COMPREHENSIVE_TODO.md`
5. Follow workflow above

---

*This workflow document integrates all aspects of the Kelly Music Brain development process. Update as the project evolves.*
