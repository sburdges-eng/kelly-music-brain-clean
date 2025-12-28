# Kelly Project - Team Roles & Responsibilities

**Last Updated**: December 28, 2024  
**Project**: Kelly - Therapeutic iDAW (Desktop)  
**Repository**: `sburdges-eng/kelly-music-brain-clean`

---

## Overview

This document defines the team structure and role responsibilities for the Kelly project. Kelly is a multi-disciplinary project combining music theory, AI/ML, audio engineering, and software development to create a therapeutic desktop application that translates emotional intent into music.

### Project Architecture Stack

- **Frontend**: React + TypeScript (Vite)
- **Desktop Bridge**: Tauri 2 (Rust)
- **Backend API**: Python 3.11+ (Music Brain)
- **Audio Engine**: C++ (JUCE framework)
- **Machine Learning**: Python (PyTorch/TensorFlow)
- **Platform**: Cross-platform (macOS, Windows, Linux)

---

## Core Team Roles

### 1. Project Lead / Product Manager

**Responsibilities:**
- Define product vision and roadmap alignment with "Interrogate Before Generate" philosophy
- Prioritize features and manage sprint planning
- Coordinate between technical teams and stakeholders
- Ensure therapeutic integrity of the application
- Manage project timeline and deliverables
- Review and approve major architectural decisions

**Skills Required:**
- Product management experience
- Understanding of music production workflows
- Familiarity with therapeutic/emotional design principles
- Strong communication and coordination skills

**Key Deliverables:**
- Sprint planning and roadmaps
- Feature prioritization
- Stakeholder communication
- Project status reports

---

### 2. Music Theory / Composition Lead

**Responsibilities:**
- Design and maintain the emotional-to-musical mapping system
- Define chord progression families and harmonic rules
- Create and validate rule-breaking justifications
- Develop groove templates and timing feel definitions
- Review all music generation algorithms for emotional authenticity
- Maintain the vault of songwriting guides and theory references
- Ensure music theory accuracy across all components

**Skills Required:**
- Advanced music theory knowledge
- Professional composition experience
- Understanding of multiple genres (lo-fi, indie, EDM, jazz, etc.)
- Familiarity with DAW workflows
- Emotional intelligence and therapeutic music understanding

**Key Deliverables:**
- Chord progression databases (`chord_progression_families.json`)
- Emotional mapping definitions (`emotional_mapping.py`)
- Groove templates (`groove/templates.py`)
- Songwriting guides in vault
- Music theory validation for generated output

**Related Files:**
- `music_brain/data/emotional_mapping.py`
- `music_brain/data/chord_progression_families.json`
- `vault/Songwriting_Guides/`
- `vault/Theory_Reference/`

---

### 3. Python Backend Engineer (Music Brain API)

**Responsibilities:**
- Develop and maintain the Music Brain Python API (`http://127.0.0.1:8000`)
- Implement intent processing system (3-phase schema)
- Build harmony generation algorithms
- Develop groove extraction and application systems
- Create chord analysis and diagnostic tools
- Implement MIDI I/O operations
- Write and maintain Python unit tests
- Ensure code quality (Black, Ruff, mypy)

**Skills Required:**
- Python 3.11+ expertise
- Experience with music libraries (mido, music21, librosa)
- REST API development
- MIDI protocol understanding
- Test-driven development
- Familiarity with audio signal processing concepts

**Key Deliverables:**
- Music Brain API endpoints (`/emotions`, `/generate`, `/interrogate`)
- Intent schema implementation (`session/intent_schema.py`)
- Harmony generator (`structure/progression.py`)
- Groove system (`groove/extractor.py`, `groove/applicator.py`)
- CLI tools (`cli.py`)

**Related Files:**
- `music_brain/` (all Python modules)
- `pyproject.toml`
- `tests/python/`

---

### 4. Frontend Engineer (React/TypeScript)

**Responsibilities:**
- Develop and maintain React UI components
- Implement Tauri command integrations
- Build responsive dark-themed interface
- Create visualization components for musical data
- Implement error boundaries and API status indicators
- Ensure accessibility and UX best practices
- Write frontend unit and integration tests

**Skills Required:**
- React 19+ and TypeScript expertise
- Vite build system knowledge
- Tauri desktop app development
- Modern CSS/Tailwind CSS
- UI/UX design sensibility
- Understanding of music production UI patterns

**Key Deliverables:**
- React components in `src/`
- Tauri command hooks (`useMusicBrain`)
- UI routing and navigation
- Visual feedback systems
- Responsive layouts

**Related Files:**
- `src/` (React components)
- `package.json`
- `vite.config.ts`
- `tailwind.config.js`

---

### 5. Desktop Systems Engineer (Rust/Tauri)

**Responsibilities:**
- Develop and maintain Tauri Rust backend
- Implement Tauri commands (`get_emotions`, `generate_music`, `interrogate`)
- Handle HTTP communication with Music Brain API
- Manage desktop OS integrations (file system, notifications)
- Build cross-platform compatibility
- Handle application packaging and distribution
- Optimize desktop performance

**Skills Required:**
- Rust programming expertise
- Tauri 2 framework experience
- Cross-platform desktop development
- HTTP/REST API integration
- Understanding of OS-level APIs (macOS, Windows, Linux)

**Key Deliverables:**
- Tauri commands in `src-tauri/src/`
- Desktop integration features
- Build configurations
- Platform-specific implementations

**Related Files:**
- `src-tauri/`
- `src-tauri/tauri.conf.json`
- `src-tauri/Cargo.toml`

---

### 6. Audio Engine Engineer (C++/JUCE)

**Responsibilities:**
- Develop C++ audio processing engine using JUCE
- Implement real-time audio playback and synthesis
- Build VST3/AU plugin architecture (future phase)
- Optimize low-latency audio performance
- Create audio effect processors
- Integrate with Python Music Brain via C++ bindings
- Implement audio analysis features (spectrum, beat detection)

**Skills Required:**
- C++20 expertise
- JUCE framework experience
- Audio DSP knowledge
- Plugin development (VST3, AU)
- Real-time programming skills
- Understanding of audio threading and latency optimization

**Key Deliverables:**
- Audio engine in `cpp_music_brain/`
- JUCE-based audio processors
- Plugin implementations (future)
- Audio analysis tools

**Related Files:**
- `cpp_music_brain/src/`
- `cpp_music_brain/include/`
- `CMakeLists.txt`
- `external/JUCE/`

---

### 7. Machine Learning Engineer

**Responsibilities:**
- Develop emotion classification models
- Train models on emotional-musical datasets
- Implement ML-based chord prediction
- Create tempo and key detection algorithms
- Build audio feature extraction pipelines
- Optimize model performance for desktop deployment
- Maintain ML training pipelines and datasets

**Skills Required:**
- Python ML frameworks (PyTorch/TensorFlow)
- Audio ML experience (librosa, essentia)
- Model training and optimization
- Dataset curation and annotation
- Understanding of music information retrieval (MIR)

**Key Deliverables:**
- Emotion classification models
- Audio analysis models
- Training scripts in `ML Kelly Training/`
- Model cards in `docs/model_cards/`
- Dataset management

**Related Files:**
- `models/`
- `ML Kelly Training/`
- `checkpoints/`
- `datasets/`
- `python/penta_core/teachers/`

---

### 8. DevOps / Build Engineer

**Responsibilities:**
- Maintain CI/CD pipelines (GitHub Actions)
- Manage build systems (CMake, npm, Cargo)
- Configure development environments
- Handle dependency management
- Set up testing infrastructure
- Manage deployment workflows
- Ensure cross-platform builds work correctly

**Skills Required:**
- CI/CD expertise (GitHub Actions)
- Build systems (CMake, npm, Cargo, setuptools)
- Docker/containerization
- Cross-platform build experience
- Python packaging knowledge
- Rust toolchain management

**Key Deliverables:**
- GitHub Actions workflows (`.github/workflows/`)
- Build configurations
- Development environment setup scripts
- Deployment configurations

**Related Files:**
- `.github/workflows/`
- `CMakeLists.txt`
- `pyproject.toml`
- `package.json`
- `Cargo.toml`
- `environment.yml`

---

### 9. QA / Test Engineer

**Responsibilities:**
- Write and maintain comprehensive test suites
- Develop integration tests across Python, Rust, and C++ components
- Create end-to-end testing scenarios
- Perform manual testing of UI/UX flows
- Document bugs and regression issues
- Validate emotional authenticity of generated music
- Ensure test coverage >80%

**Skills Required:**
- Testing frameworks (pytest, Catch2, Jest)
- Test automation
- Manual QA methodologies
- Understanding of music theory for validation
- Bug tracking and documentation

**Key Deliverables:**
- Unit tests in `tests/python/`, `tests/cpp/`
- Integration test suites
- Test coverage reports
- Bug documentation
- QA test plans

**Related Files:**
- `tests/`
- `pyproject.toml` (pytest config)
- `cpp_music_brain/tests/`

---

### 10. Technical Writer / Documentation Lead

**Responsibilities:**
- Maintain comprehensive project documentation
- Write API documentation
- Create user guides and tutorials
- Document development setup procedures
- Maintain architectural decision records
- Write onboarding materials for new contributors
- Keep CONTRIBUTING.md and README.md up to date

**Skills Required:**
- Technical writing expertise
- Understanding of music production concepts
- Familiarity with developer documentation standards
- Markdown proficiency
- Ability to explain complex concepts clearly

**Key Deliverables:**
- README.md, CONTRIBUTING.md
- API documentation
- User guides in `docs/`
- Vault documentation (`vault/`)
- Code comments and docstrings

**Related Files:**
- `docs/`
- `README.md`
- `CONTRIBUTING.md`
- `vault/`
- `.github/copilot-instructions.md`

---

### 11. UI/UX Designer

**Responsibilities:**
- Design therapeutic, emotion-focused interfaces
- Create dark theme color palettes
- Design visual representations of musical concepts
- Develop user flow diagrams
- Ensure accessibility standards (WCAG)
- Create mockups and prototypes
- Conduct user research and testing
- Design icon sets and visual assets

**Skills Required:**
- UI/UX design experience
- Figma or similar design tools
- Understanding of music production UIs (Ableton, Logic Pro)
- Accessibility knowledge
- Emotional design principles
- Visual design skills

**Key Deliverables:**
- UI mockups and designs
- Design system documentation
- Icon assets in `assets/`
- User flow diagrams
- Accessibility audit reports

**Related Files:**
- `assets/`
- `public/`
- `src-tauri/icons/`

---

## Supporting Roles

### 12. Community Manager

**Responsibilities:**
- Manage GitHub issues and pull requests
- Facilitate community discussions
- Onboard new contributors
- Organize community events (e.g., office hours)
- Moderate forums and communication channels
- Track contributor recognition

**Skills Required:**
- Community management experience
- Strong communication skills
- Understanding of open-source workflows
- Conflict resolution abilities

---

### 13. Security Specialist (Part-time/Advisory)

**Responsibilities:**
- Review code for security vulnerabilities
- Conduct security audits
- Manage dependency security updates
- Ensure safe handling of user data
- Review API security practices

**Skills Required:**
- Security auditing experience
- Understanding of OWASP guidelines
- Familiarity with Rust, Python, C++ security practices

---

## Team Structure Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   Project Lead / PM                      │
│              (Vision, Roadmap, Coordination)             │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐   ┌───▼────┐   ┌───▼─────┐
    │  Music  │   │ Tech   │   │  UX/UI  │
    │  Theory │   │ Leads  │   │ Design  │
    │  Lead   │   │        │   │  Lead   │
    └────┬────┘   └───┬────┘   └────┬────┘
         │            │              │
         │     ┌──────┼──────┐       │
         │     │      │      │       │
    ┌────▼─────▼──┐ ┌▼──┐  ┌▼───┐  ┌▼─────┐
    │   Python    │ │Rust│ │C++ │  │ UI   │
    │   Backend   │ │Tauri│ │JUCE│  │React │
    │   Engineer  │ │Eng.│ │Eng.│  │ Eng. │
    └─────────────┘ └────┘ └────┘  └──────┘
         │
    ┌────▼────┐
    │   ML    │
    │Engineer │
    └─────────┘

    Supporting Teams:
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ DevOps/  │  │   QA /   │  │  Tech    │
    │  Build   │  │  Test    │  │  Writer  │
    └──────────┘  └──────────┘  └──────────┘
```

---

## Role Assignment Guidelines

### For New Contributors

1. Review the role descriptions above
2. Check the `CONTRIBUTING.md` file for development setup
3. Look at open issues tagged with `good-first-issue`
4. Join discussions in relevant areas
5. Start with small contributions before taking on major roles

### For Project Maintainers

1. Assign roles based on demonstrated expertise and contribution history
2. Ensure each core role has at least one assigned person
3. Create redundancy for critical roles (primary + backup)
4. Document role transitions and knowledge transfer
5. Review role assignments quarterly

---

## Communication & Coordination

### Primary Channels

- **GitHub Issues**: Feature requests, bugs, discussions
- **GitHub Discussions**: Community Q&A, RFCs
- **Pull Requests**: Code review and technical discussions

### Meeting Cadence (Recommended)

- **Weekly**: Engineering sync (all technical roles)
- **Bi-weekly**: Sprint planning (PM, leads)
- **Monthly**: Architecture review (technical leads)
- **Quarterly**: Roadmap planning (all roles)

---

## Onboarding Process

### New Team Member Checklist

- [ ] Read README.md and CONTRIBUTING.md
- [ ] Complete development environment setup
- [ ] Review project architecture docs (`docs/ARCHITECTURE.md`)
- [ ] Read `.github/copilot-instructions.md` for project philosophy
- [ ] Complete a "good first issue" to understand workflow
- [ ] Shadow an experienced team member in your role
- [ ] Attend weekly engineering sync
- [ ] Review vault documentation for music theory context

---

## Current Team Status

**Note**: This is a template. Maintainers should update this section with actual team member assignments.

| Role | Primary | Backup | Status |
|------|---------|--------|--------|
| Project Lead / PM | TBD | - | Open |
| Music Theory Lead | TBD | - | Open |
| Python Backend Engineer | TBD | - | Open |
| Frontend Engineer (React) | TBD | - | Open |
| Desktop Systems Engineer (Rust) | TBD | - | Open |
| Audio Engine Engineer (C++) | TBD | - | Open |
| ML Engineer | TBD | - | Open |
| DevOps / Build Engineer | TBD | - | Open |
| QA / Test Engineer | TBD | - | Open |
| Technical Writer | TBD | - | Open |
| UI/UX Designer | TBD | - | Open |

---

## Skill Development & Training

### Recommended Learning Paths

**For Music Theory Role:**
- Music theory fundamentals (harmony, counterpoint)
- Emotional music composition techniques
- DAW workflows and production

**For Backend Engineers:**
- Python audio libraries (librosa, mido, music21)
- MIDI protocol specifications
- Music information retrieval (MIR)

**For Frontend Engineers:**
- Tauri desktop app development
- Music visualization techniques
- Accessibility in audio applications

**For Audio Engineers:**
- JUCE framework tutorials
- Audio DSP fundamentals
- Plugin development standards

---

## Questions & Contact

For questions about team roles or to express interest in joining the team:

1. Open a GitHub Discussion in the "Team & Roles" category
2. Review open issues for role-specific work
3. Contact the Project Lead (once assigned)

---

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2024-12-28 | 1.0 | Initial team roles document | GitHub Copilot |

---

**License**: MIT  
**Project**: Kelly - Therapeutic iDAW
