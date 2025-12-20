#!/bin/bash
# Start the Music Brain API Server

cd "$(dirname "$0")/.." || exit 1

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Please create one first."
    exit 1
fi

# Start the server
echo "Starting Music Brain API Server..."
echo "API will be available at: http://127.0.0.1:8000"
echo "Interactive docs: http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn music_brain.api:app --host 127.0.0.1 --port 8000 --reload
