# Contributing to Kelly

Thank you for your interest in contributing to Kelly!

## Team Roles & Responsibilities

Before contributing, please review our [Team Roles document](../TEAM_ROLES.md) to understand the different roles and responsibilities within the project. This will help you identify where your skills and interests align with the project's needs.

## Development Setup

### Prerequisites

- Python 3.11+
- CMake 3.27+
- Qt 6
- C++20 compatible compiler

### Python Setup

```bash
# Clone the repository
git clone https://github.com/sburdges-eng/Kelly.git
cd Kelly

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/python -v
```

### C++ Setup

```bash
# Initialize submodules (JUCE, Catch2, etc.)
git submodule update --init --recursive

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON

# Build
cmake --build build

# Run tests
cd build && ctest -V
```

## Code Style

### Python
- Follow PEP 8 guidelines
- Use Black for formatting: `black src/kelly tests/python`
- Use Ruff for linting: `ruff check src/kelly tests/python`
- Use type hints and mypy: `mypy src/kelly`

### C++
- Follow C++20 best practices
- Use clang-format for formatting
- Use const correctness
- Prefer modern C++ idioms

## Testing

All contributions should include tests:

- Python: pytest-based tests in `tests/python/`
- C++: Catch2-based tests in `tests/cpp/`
- Coverage: Aim for >80% code coverage

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests
5. Ensure all tests pass
6. Run linters and formatters
7. Commit with descriptive messages
8. Push to your fork
9. Create a Pull Request

## Commit Messages

Use clear, descriptive commit messages:

```
Add emotion mapping for surprise category

- Implement surprise emotion nodes
- Add tests for surprise mapping
- Update documentation
```

## Areas for Contribution

See our [Team Roles document](../TEAM_ROLES.md) for detailed information about different contribution areas. Key areas include:

### Music Theory & Composition
- Expanding the emotion thesaurus to full 216 nodes
- Adding more groove templates and genre-specific patterns
- Validating emotional-to-musical mappings
- Creating songwriting guides

### Software Development
- **Python Backend**: Improving MIDI generation algorithms, expanding the Music Brain API
- **Frontend (React/TypeScript)**: Enhancing the GUI interface, improving UX
- **Audio Engine (C++/JUCE)**: Adding audio analysis features, implementing real-time processing
- **Desktop Integration (Rust/Tauri)**: Improving OS integrations, optimizing performance

### Machine Learning
- Developing emotion classification models
- Implementing audio feature extraction
- Training chord and tempo detection models

### Documentation & Testing
- Writing documentation and examples
- Improving test coverage (target >80%)
- Creating user guides and tutorials

## Getting Started

1. **Choose Your Role**: Review the [Team Roles document](../TEAM_ROLES.md) to find where you'd like to contribute
2. **Start Small**: Look for issues tagged with `good-first-issue` to get familiar with the codebase
3. **Understand the Philosophy**: Read the project README and `.github/copilot-instructions.md` to understand the "Interrogate Before Generate" philosophy
4. **Join Discussions**: Participate in GitHub Discussions to connect with the community

## Questions?

Feel free to open an issue or GitHub Discussion before starting work on major changes. For team-related questions, refer to the [Team Roles document](../TEAM_ROLES.md).
