#!/bin/bash
set -e

echo "🎵 Kelly Music Brain 2.0 - Post-Create Setup"
echo "============================================="

# Initialize git submodules
echo "📦 Initializing git submodules..."
if [ -f .gitmodules ]; then
    git submodule update --init --recursive
fi

# Configure git
echo "🔧 Configuring git..."
git config --global --add safe.directory /workspace
git lfs install

# Set up Python environment
echo "🐍 Setting up Python environment..."
if [ -f pyproject.toml ]; then
    # Install project in editable mode with dev dependencies
    pip install -e ".[dev]" || echo "⚠️  pip install failed, trying poetry..."
    
    # Try poetry if pip fails
    if command -v poetry &> /dev/null; then
        poetry install || echo "⚠️  Poetry install failed"
    fi
fi

# Set up pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
if [ -f .pre-commit-config.yaml ]; then
    pip install pre-commit
    pre-commit install || echo "⚠️  Pre-commit install failed"
fi

# Create common directories if they don't exist
echo "📁 Creating directory structure..."
mkdir -p tools/scripts
mkdir -p tools/templates
mkdir -p packages
mkdir -p data
mkdir -p docs/architecture
mkdir -p docs/api
mkdir -p docs/guides
mkdir -p tests/python
mkdir -p tests/cpp
mkdir -p examples

# Set up ccache
echo "⚡ Configuring ccache..."
mkdir -p .ccache
ccache --max-size=5G

# Display environment info
echo ""
echo "✅ Setup complete!"
echo ""
echo "Environment Information:"
echo "------------------------"
echo "Python: $(python3 --version)"
echo "Node.js: $(node --version)"
echo "npm: $(npm --version)"
echo "pnpm: $(pnpm --version)"
echo "CMake: $(cmake --version | head -n1)"
echo "gcc: $(gcc --version | head -n1)"
echo "clang: $(clang --version | head -n1)"
echo ""
echo "Poetry: $(poetry --version 2>/dev/null || echo 'Not installed')"
echo "git-filter-repo: $(git-filter-repo --version 2>/dev/null || echo 'Not installed')"
echo ""
echo "📚 Quick Start:"
echo "  - Run tools: cd tools/scripts && python deduplicate.py"
echo "  - Build C++: cmake -B build && cmake --build build"
echo "  - Run tests: pytest tests/python"
echo ""
echo "Happy coding! 🎹"
