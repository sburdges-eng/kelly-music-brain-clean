#!/bin/bash
# Run full code quality suite
# Usage: ./scripts/run_code_quality.sh [--fix]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

FIX_MODE=false
if [ "$1" == "--fix" ]; then
    FIX_MODE=true
fi

echo "=========================================="
echo "Code Quality Suite"
echo "=========================================="
echo ""

# Black - Code formatting
echo "1. Running Black (code formatting)..."
if [ "$FIX_MODE" = true ]; then
    black . || echo "⚠️  Black found formatting issues (run with --fix to auto-fix)"
else
    black --check . || echo "⚠️  Black found formatting issues (run with --fix to auto-fix)"
fi
echo "✅ Black check complete"
echo ""

# Ruff - Linting
echo "2. Running Ruff (linting)..."
if [ "$FIX_MODE" = true ]; then
    ruff check . --fix || echo "⚠️  Ruff found linting issues"
else
    ruff check . || echo "⚠️  Ruff found linting issues (run with --fix to auto-fix)"
fi
echo "✅ Ruff check complete"
echo ""

# isort - Import sorting
echo "3. Running isort (import sorting)..."
if [ "$FIX_MODE" = true ]; then
    isort . || echo "⚠️  isort found import sorting issues"
else
    isort --check-only . || echo "⚠️  isort found import sorting issues (run with --fix to auto-fix)"
fi
echo "✅ isort check complete"
echo ""

# MyPy - Type checking
echo "4. Running MyPy (type checking)..."
mypy music_brain penta_core --ignore-missing-imports || echo "⚠️  MyPy found type checking issues"
echo "✅ MyPy check complete"
echo ""

# Pytest - Run tests
echo "5. Running Pytest (test suite)..."
pytest tests_music-brain/ -v --tb=short || echo "⚠️  Some tests failed"
echo "✅ Test suite complete"
echo ""

echo "=========================================="
echo "Code Quality Suite Complete"
echo "=========================================="
echo ""
echo "To auto-fix issues, run: ./scripts/run_code_quality.sh --fix"
