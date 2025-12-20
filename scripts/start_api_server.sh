#!/bin/bash
# Start the Music Brain API Server
# 
# Usage:
#   ./scripts/start_api_server.sh              # Start with defaults (127.0.0.1:8000)
#   PORT=8080 ./scripts/start_api_server.sh   # Start on custom port
#   HOST=0.0.0.0 ./scripts/start_api_server.sh # Listen on all interfaces

set -e  # Exit on error

# Get script directory and change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Configuration (can be overridden by environment variables)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-true}"

# Find and activate virtual environment
VENV_DIR=""
for venv_name in ".venv" "venv" "env"; do
    if [ -d "$venv_name" ]; then
        VENV_DIR="$venv_name"
        break
    fi
done

if [ -z "$VENV_DIR" ]; then
    echo "❌ Error: Virtual environment not found."
    echo "   Please create one first:"
    echo "   python3 -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -e ."
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check if required packages are installed
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "❌ Error: Required packages not found."
    echo "   Installing FastAPI and uvicorn..."
    pip install fastapi uvicorn || {
        echo "❌ Failed to install dependencies"
        exit 1
    }
fi

# Check if music_brain.api module exists
if ! python3 -c "import music_brain.api" 2>/dev/null; then
    echo "❌ Error: music_brain.api module not found."
    echo "   Make sure the project is installed: pip install -e ."
    exit 1
fi

# Check if port is already in use
if command -v lsof >/dev/null 2>&1; then
    if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Warning: Port $PORT is already in use."
        echo "   Please stop the existing server or use a different port:"
        echo "   PORT=8080 $0"
        exit 1
    fi
fi

# Build reload flag
RELOAD_FLAG=""
if [ "$RELOAD" = "true" ]; then
    RELOAD_FLAG="--reload"
fi

# Display startup information
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎵  Music Brain API Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 Server:     http://$HOST:$PORT"
echo "📚 API Docs:   http://$HOST:$PORT/docs"
echo "📖 ReDoc:      http://$HOST:$PORT/redoc"
echo "🔄 Reload:     $RELOAD"
echo "🐍 Python:     $(python3 --version)"
echo "📦 Venv:       $VENV_DIR"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the server
exec python3 -m uvicorn music_brain.api:app \
    --host "$HOST" \
    --port "$PORT" \
    $RELOAD_FLAG
