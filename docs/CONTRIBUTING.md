# Contributing to Kelly

Thank you for your interest in contributing to Kelly!

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

- Expanding the emotion thesaurus to full 216 nodes
- Adding more groove templates
- Improving MIDI generation algorithms
- Enhancing the GUI interface
- Adding audio analysis features (librosa integration)
- Writing documentation and examples
- Improving test coverage

## Questions?

Feel free to open an issue for discussion before starting work on major changes.
