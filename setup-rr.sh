#!/bin/bash

# RR CLI Setup Script for Claude Code Integration
# This script installs and configures the RR CLI tool

set -e

echo "=================================================="
echo "RR CLI Tool Setup for Claude Code"
echo "=================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Install RR CLI Tool
echo "Installing RR CLI Tool..."
cd "$(dirname "$0")/rr_cli"
pip install -e .

if [ $? -eq 0 ]; then
    echo "✓ RR CLI installed successfully"
else
    echo "✗ Failed to install RR CLI"
    exit 1
fi

echo ""

# Check for OpenAI API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠ OpenAI API Key not found in environment variables"
    echo ""
    echo "To use AI features, set your API key:"
    echo ""
    echo "  Option 1: Set environment variable"
    echo "  export OPENAI_API_KEY=your_api_key_here"
    echo ""
    echo "  Option 2: Create .env file"
    echo "  cd $(dirname "$0")/rr_cli"
    echo "  cp .env.example .env"
    echo "  # Edit .env and add your API key"
    echo ""
else
    echo "✓ OpenAI API Key found"
fi

echo ""

# Verify installation
echo "Verifying installation..."
if rr --help > /dev/null 2>&1; then
    echo "✓ RR CLI is ready to use"
    echo ""
    echo "Quick start:"
    echo "  rr --help              Show help"
    echo "  rr git status          Check git status"
    echo "  rr ai commit-msg       Generate AI commit message"
    echo "  rr ai explain TOPIC    Learn about a concept"
else
    echo "✗ RR CLI verification failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "In Claude Code, you can now use:"
echo "  /rr-commit    Generate commit message"
echo "  /rr-status    Check git status"
echo "  /rr-analyze   Analyze code"
echo "  /rr-suggest   Get suggestions"
echo "  /rr-explain   Learn concepts"
echo "  /rr-ask       Ask questions"
echo ""
echo "Or use the full RR CLI directly:"
echo "  rr git status"
echo "  rr ai commit-msg --staged"
echo ""
