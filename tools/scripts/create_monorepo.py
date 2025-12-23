#!/usr/bin/env python3
"""
Monorepo Scaffold Generator

Creates complete directory structure for Kelly Music Brain 2.0 monorepo with:
- packages/{core,cli,desktop,plugins,mobile,web}
- Package-level configuration files
- Root workspace configuration
- .gitignore patterns
- Pre-commit hooks setup

Usage:
    python create_monorepo.py --output /path/to/new/repo
    python create_monorepo.py --output . --dry-run
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List


class MonorepoGenerator:
    """Generate monorepo directory structure and configuration files."""
    
    def __init__(self, output_dir: str, dry_run: bool = False):
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        
    def create_directory(self, path: Path):
        """Create a directory."""
        if self.dry_run:
            print(f"[DRY RUN] Would create directory: {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {path}")
    
    def create_file(self, path: Path, content: str):
        """Create a file with content."""
        if self.dry_run:
            print(f"[DRY RUN] Would create file: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            print(f"✓ Created file: {path}")
    
    def generate_structure(self):
        """Generate complete monorepo structure."""
        print(f"Generating Kelly Music Brain 2.0 monorepo at: {self.output_dir}")
        
        # Root directories
        root_dirs = [
            'packages/core/python/kelly_core',
            'packages/core/cpp/include/kelly',
            'packages/core/cpp/src',
            'packages/cli/kelly_cli',
            'packages/desktop/src/gui',
            'packages/desktop/src/audio',
            'packages/plugins/vst3',
            'packages/plugins/clap',
            'packages/mobile/ios',
            'packages/mobile/android',
            'packages/web/src',
            'data/emotions',
            'data/scales',
            'data/genres',
            'data/rule_breaking',
            'vault/Songwriting_Guides',
            'vault/Theory_Reference',
            'tests/python',
            'tests/cpp',
            'tools/scripts',
            'tools/templates',
            'docs/architecture',
            'docs/api',
            'docs/guides',
            'examples/python_api',
            'examples/cli_usage',
            'examples/plugin_integration',
            'archive',
            '.github/workflows',
            '.github/ISSUE_TEMPLATE'
        ]
        
        for dir_path in root_dirs:
            self.create_directory(self.output_dir / dir_path)
        
        # Generate configuration files
        self.generate_root_configs()
        self.generate_package_configs()
        self.generate_github_workflows()
        self.generate_gitignore()
        self.generate_documentation()
        
        print("\n✅ Monorepo structure generation complete!")
    
    def generate_root_configs(self):
        """Generate root-level configuration files."""
        
        # Root pyproject.toml
        pyproject_content = '''[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "kelly-music-brain"
version = "2.0.0"
description = "Therapeutic iDAW with emotion-driven music generation"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Kelly Development Team"}
]
keywords = ["music", "emotion", "therapy", "daw", "midi", "audio"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Sound/Audio",
]

dependencies = [
    "music21>=9.1.0",
    "librosa>=0.10.0",
    "mido>=1.3.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
    "pre-commit>=3.5.0",
]

[project.scripts]
kelly = "kelly_cli.main:app"
daiw = "kelly_cli.main:app"  # Legacy alias

[tool.setuptools.packages.find]
where = ["packages/core/python", "packages/cli"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=kelly --cov-report=term-missing --cov-report=xml"

[tool.black]
line-length = 100
target-version = ['py311']
include = '\\.pyi?$'

[tool.ruff]
line-length = 100
target-version = "py311"
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
]
ignore = [
    "E501",  # line too long, handled by black
]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
'''
        self.create_file(self.output_dir / 'pyproject.toml', pyproject_content)
        
        # Root CMakeLists.txt
        cmake_content = '''cmake_minimum_required(VERSION 3.27)
project(KellyMusicBrain VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_CLI "Build CLI application" ON)
option(BUILD_DESKTOP "Build desktop application" ON)
option(BUILD_PLUGINS "Build VST3/CLAP plugins" ON)
option(BUILD_MOBILE "Build mobile apps" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(ENABLE_TRACY "Enable Tracy profiling" OFF)

# Find packages
find_package(Qt6 COMPONENTS Core Widgets Multimedia QUIET)

# JUCE setup for plugins and desktop
if(BUILD_PLUGINS OR BUILD_DESKTOP)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/JUCE")
        add_subdirectory(external/JUCE EXCLUDE_FROM_ALL)
    endif()
endif()

# Core library
add_subdirectory(packages/core/cpp)

# CLI (Python-based, installed via pip)
if(BUILD_CLI)
    message(STATUS "CLI will be installed via Python package manager")
endif()

# Desktop application
if(BUILD_DESKTOP AND Qt6_FOUND)
    add_subdirectory(packages/desktop)
else()
    message(STATUS "Desktop app disabled or Qt6 not found")
endif()

# Plugins
if(BUILD_PLUGINS)
    add_subdirectory(packages/plugins)
else()
    message(STATUS "Plugins disabled")
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Kelly Music Brain 2.0 Configuration:")
message(STATUS "  Build CLI:      ${BUILD_CLI}")
message(STATUS "  Build Desktop:  ${BUILD_DESKTOP}")
message(STATUS "  Build Plugins:  ${BUILD_PLUGINS}")
message(STATUS "  Build Mobile:   ${BUILD_MOBILE}")
message(STATUS "  Build Tests:    ${BUILD_TESTS}")
message(STATUS "  Tracy Profiling: ${ENABLE_TRACY}")
message(STATUS "")
'''
        self.create_file(self.output_dir / 'CMakeLists.txt', cmake_content)
        
        # Root package.json for web workspace
        package_json_content = '''{
  "name": "kelly-music-brain",
  "version": "2.0.0",
  "private": true,
  "description": "Kelly Music Brain 2.0 - Therapeutic iDAW",
  "workspaces": [
    "packages/web"
  ],
  "scripts": {
    "dev": "pnpm --filter web dev",
    "build": "pnpm --filter '*' build",
    "test": "pnpm --filter '*' test",
    "lint": "pnpm --filter '*' lint"
  },
  "engines": {
    "node": ">=18.0.0",
    "pnpm": ">=8.0.0"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/sburdges-eng/kelly-music-brain-clean.git"
  },
  "license": "MIT"
}
'''
        self.create_file(self.output_dir / 'package.json', package_json_content)
        
        # .pre-commit-config.yaml
        precommit_content = '''repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-clang-format
    rev: v17.0.6
    hooks:
      - id: clang-format
        types_or: [c++, c]
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
'''
        self.create_file(self.output_dir / '.pre-commit-config.yaml', precommit_content)
    
    def generate_package_configs(self):
        """Generate package-level configuration files."""
        
        # Core Python package
        core_py_pyproject = '''[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "kelly-core"
version = "2.0.0"
description = "Kelly Music Brain core emotion and music processing"
requires-python = ">=3.11"

dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "music21>=9.1.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["kelly_core*"]
'''
        self.create_file(
            self.output_dir / 'packages/core/python/pyproject.toml',
            core_py_pyproject
        )
        
        core_py_readme = '''# Kelly Core (Python)

Core emotion processing and music theory engine for Kelly Music Brain 2.0.

## Features

- Emotional state mapping
- Music theory analysis
- Chord progression generation
- Groove extraction and application
'''
        self.create_file(
            self.output_dir / 'packages/core/python/README.md',
            core_py_readme
        )
        
        # Core C++ package
        core_cpp_cmake = '''cmake_minimum_required(VERSION 3.27)
project(KellyCore VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Create library
add_library(KellyCore STATIC ${SOURCES})

target_include_directories(KellyCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link dependencies
target_link_libraries(KellyCore PUBLIC
    # Add dependencies here
)

# Install
install(TARGETS KellyCore
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/kelly
    DESTINATION include
)
'''
        self.create_file(
            self.output_dir / 'packages/core/cpp/CMakeLists.txt',
            core_cpp_cmake
        )
        
        core_cpp_readme = '''# Kelly Core (C++)

High-performance audio processing and DSP for Kelly Music Brain 2.0.

## Features

- Real-time audio processing
- MIDI event handling
- DSP algorithms
- Audio format I/O
'''
        self.create_file(
            self.output_dir / 'packages/core/cpp/README.md',
            core_cpp_readme
        )
        
        # CLI package
        cli_pyproject = '''[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "kelly-cli"
version = "2.0.0"
description = "Kelly Music Brain command-line interface"
requires-python = ">=3.11"

dependencies = [
    "kelly-core==2.0.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
]

[project.scripts]
kelly = "kelly_cli.main:app"
daiw = "kelly_cli.main:app"

[tool.setuptools.packages.find]
where = ["."]
include = ["kelly_cli*"]
'''
        self.create_file(
            self.output_dir / 'packages/cli/pyproject.toml',
            cli_pyproject
        )
        
        # Desktop package
        desktop_cmake = '''cmake_minimum_required(VERSION 3.27)
project(KellyDesktop VERSION 2.0.0 LANGUAGES CXX)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets Multimedia)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(KellyDesktop ${SOURCES})

target_link_libraries(KellyDesktop PRIVATE
    KellyCore
    Qt6::Core
    Qt6::Widgets
    Qt6::Multimedia
)

install(TARGETS KellyDesktop
    RUNTIME DESTINATION bin
)
'''
        self.create_file(
            self.output_dir / 'packages/desktop/CMakeLists.txt',
            desktop_cmake
        )
        
        # Plugins package
        plugins_cmake = '''cmake_minimum_required(VERSION 3.27)
project(KellyPlugins VERSION 2.0.0 LANGUAGES CXX)

# VST3 plugin
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vst3")
    add_subdirectory(vst3)
endif()

# CLAP plugin
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/clap")
    add_subdirectory(clap)
endif()
'''
        self.create_file(
            self.output_dir / 'packages/plugins/CMakeLists.txt',
            plugins_cmake
        )
        
        # Web package
        web_package_json = '''{
  "name": "@kelly/web",
  "version": "2.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0"
  }
}
'''
        self.create_file(
            self.output_dir / 'packages/web/package.json',
            web_package_json
        )
    
    def generate_github_workflows(self):
        """Generate GitHub Actions workflow files."""
        
        # Python CI
        ci_python = '''name: Python CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Lint with ruff
        run: ruff check packages/core/python packages/cli tests/python
      
      - name: Format check with black
        run: black --check packages/core/python packages/cli tests/python
      
      - name: Type check with mypy
        run: mypy packages/core/python packages/cli
        continue-on-error: true
      
      - name: Run pytest
        run: pytest tests/python -v --cov --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: python
'''
        self.create_file(
            self.output_dir / '.github/workflows/ci-python.yml',
            ci_python
        )
        
        # C++ CI
        ci_cpp = '''name: C++ CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        build_type: [Debug, Release]
    
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      
      - name: Install dependencies (Ubuntu)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake ninja-build qt6-base-dev libasound2-dev
      
      - name: Install dependencies (macOS)
        if: runner.os == 'macOS'
        run: brew install cmake ninja qt@6
      
      - name: Install dependencies (Windows)
        if: runner.os == 'Windows'
        run: choco install cmake ninja
      
      - name: Configure CMake
        run: |
          cmake -B build -G Ninja \\
            -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} \\
            -DBUILD_TESTS=ON \\
            -DBUILD_PLUGINS=OFF
      
      - name: Build
        run: cmake --build build --config ${{ matrix.build_type }}
      
      - name: Run tests
        run: |
          cd build
          ctest --output-on-failure -C ${{ matrix.build_type }}
'''
        self.create_file(
            self.output_dir / '.github/workflows/ci-cpp.yml',
            ci_cpp
        )
        
        # Plugin build workflow
        build_plugins = '''name: Build Plugins

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build-vst3:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [macos-latest, windows-latest]
    
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      
      - name: Install dependencies (macOS)
        if: runner.os == 'macOS'
        run: brew install cmake ninja qt@6
      
      - name: Install dependencies (Windows)
        if: runner.os == 'Windows'
        run: choco install cmake ninja
      
      - name: Configure CMake
        run: |
          cmake -B build -G Ninja \\
            -DCMAKE_BUILD_TYPE=Release \\
            -DBUILD_PLUGINS=ON \\
            -DBUILD_TESTS=OFF
      
      - name: Build VST3
        run: cmake --build build --config Release --target KellyPlugin
      
      - name: Sign plugin (macOS)
        if: runner.os == 'macOS'
        run: |
          # TODO: Add codesigning
          echo "Skipping codesigning for now"
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: kelly-plugin-${{ matrix.os }}
          path: |
            build/**/*.vst3
            build/**/*.clap
'''
        self.create_file(
            self.output_dir / '.github/workflows/build-plugins.yml',
            build_plugins
        )
        
        # Release workflow
        release = '''name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  publish-pypi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Build Python packages
        run: |
          pip install build
          python -m build
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
  
  create-release:
    runs-on: ubuntu-latest
    needs: [publish-pypi]
    steps:
      - uses: actions/checkout@v4
      
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          generate_release_notes: true
'''
        self.create_file(
            self.output_dir / '.github/workflows/release.yml',
            release
        )
    
    def generate_gitignore(self):
        """Generate comprehensive .gitignore."""
        gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
.pytest_cache/
.coverage
.coverage.*
htmlcov/
*.cover
.mypy_cache/
.ruff_cache/
.dmypy.json
dmypy.json

# C++
*.o
*.obj
*.exe
*.out
*.app
*.dll
*.so
*.dylib
*.a
*.lib
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
compile_commands.json
CTestTestfile.cmake
_deps/

# Build directories
build/
Build/
out/
.cache/
.ccache/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
.pnpm-store/

# Audio/MIDI files (optional)
*.wav
*.mp3
*.flac
*.aiff
*.mid
*.midi

# Temp files
*.tmp
*.temp
.tmp/
tmp/

# OS
Thumbs.db
.DS_Store
'''
        self.create_file(self.output_dir / '.gitignore', gitignore_content)
    
    def generate_documentation(self):
        """Generate documentation files."""
        
        # Root README
        readme = '''# Kelly Music Brain 2.0 🎵

> Therapeutic iDAW: Translate emotions into music

Kelly Music Brain 2.0 is a comprehensive emotion-driven music creation platform combining:
- 216-node emotional taxonomy (evolved from DAiW's 7-node system)
- Real-time audio processing and DSP
- VST3/CLAP plugin support
- Desktop application with Qt6 GUI
- Python CLI and API
- Mobile companion apps (iOS/Android)
- Web interface

## 🚀 Quick Start

### Installation

```bash
# Install Python CLI and core
pip install kelly-music-brain

# Or install from source
git clone https://github.com/sburdges-eng/kelly-music-brain-clean.git
cd kelly-music-brain-clean
pip install -e ".[dev]"
```

### Basic Usage

```bash
# CLI
kelly intent new --title "My Song" --emotion grief

# Python API
from kelly_core import EmotionalState, generate_progression
state = EmotionalState(valence=-0.8, arousal=0.3)
progression = generate_progression(state)
```

## 📦 Project Structure

```
kelly-music-brain/
├── packages/
│   ├── core/          # Core emotion and music processing
│   ├── cli/           # Command-line interface
│   ├── desktop/       # Qt6 desktop application
│   ├── plugins/       # VST3/CLAP plugins
│   ├── mobile/        # iOS/Android apps
│   └── web/           # Web interface
├── data/              # Emotion maps, scales, genres
├── vault/             # Knowledge base
├── docs/              # Documentation
└── tests/             # Test suites
```

## 🛠️ Development

See [docs/guides/development.md](docs/guides/development.md) for detailed setup.

### Build from Source

```bash
# Python packages
pip install -e ".[dev]"

# C++ desktop/plugins
cmake -B build -DBUILD_PLUGINS=ON
cmake --build build
```

## 📖 Documentation

- [Installation Guide](docs/guides/installation.md)
- [CLI Usage](docs/guides/cli.md)
- [Plugin Guide](docs/guides/plugins.md)
- [API Reference](docs/api/)
- [Migrating from DAiW 1.0](docs/MIGRATION.md)

## 🔄 Migrating from DAiW

See [MIGRATION.md](docs/MIGRATION.md) for complete migration guide.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built on the foundation of DAiW-Music-Brain 1.0.
'''
        self.create_file(self.output_dir / 'README.md', readme)
        
        # Migration guide
        migration = '''# Migrating from DAiW 1.0 to Kelly 2.0

## Overview

Kelly Music Brain 2.0 is the evolution of DAiW-Music-Brain, incorporating all DAiW functionality plus:
- 216-node emotional taxonomy (vs. 7-node in DAiW)
- Real-time audio processing (C++ DSP engine)
- VST3/CLAP plugin support
- Desktop application
- Mobile apps
- Web interface

## Breaking Changes

### Package Names

- `music_brain` → `kelly_core`
- `daiw` CLI → `kelly` CLI (with `daiw` alias for backwards compatibility)

### Import Paths

```python
# Old (DAiW 1.0)
from music_brain.session.intent_schema import CompleteSongIntent
from music_brain.groove.applicator import GrooveApplicator

# New (Kelly 2.0)
from kelly_core.session.intent_schema import CompleteSongIntent
from kelly_core.groove.applicator import GrooveApplicator
```

### Configuration Files

- Intent schema structure unchanged
- Data file formats unchanged
- MIDI I/O compatible

## Migration Steps

### 1. Update Dependencies

```bash
# Remove old package
pip uninstall daiw-music-brain

# Install Kelly 2.0
pip install kelly-music-brain
```

### 2. Update Import Statements

Use the provided migration script:

```bash
python tools/scripts/migrate_imports.py /path/to/your/code
```

### 3. Update CLI Commands

Most commands remain the same:

```bash
# Old
daiw intent new --title "My Song"

# New (both work)
kelly intent new --title "My Song"
daiw intent new --title "My Song"  # Legacy alias
```

### 4. Update Data Files

Data files (JSON/YAML intents) are compatible. No changes needed.

## New Features

### Enhanced Emotion System

Kelly 2.0 uses a 216-node emotional taxonomy:

```python
from kelly_core.emotions import EmotionalState

state = EmotionalState(
    core_emotion='grief',
    valence=-0.8,
    arousal=0.3,
    dominance=0.2
)
```

### Real-Time Audio Processing

```python
from kelly_core.audio import AudioProcessor

processor = AudioProcessor()
processor.load_midi('input.mid')
audio = processor.render()
```

### Plugin Integration

Use Kelly as a VST3/CLAP plugin in your DAW.

## Support

For migration issues, see:
- [GitHub Issues](https://github.com/sburdges-eng/kelly-music-brain-clean/issues)
- [Documentation](docs/)

## Timeline

- **DAiW 1.0**: Legacy support on `1.x-maintenance` branch
- **Kelly 2.0**: Active development on `main` branch
'''
        self.create_file(self.output_dir / 'docs/MIGRATION.md', migration)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Kelly Music Brain 2.0 monorepo structure"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output directory for monorepo (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without creating files'
    )
    
    args = parser.parse_args()
    
    generator = MonorepoGenerator(args.output, dry_run=args.dry_run)
    generator.generate_structure()
    
    if not args.dry_run:
        print("\n📚 Next steps:")
        print("  1. Review generated structure")
        print("  2. Run: git init (if new repo)")
        print("  3. Run: pre-commit install")
        print("  4. Run: pip install -e '.[dev]'")
        print("  5. Start development!")


if __name__ == '__main__':
    main()
