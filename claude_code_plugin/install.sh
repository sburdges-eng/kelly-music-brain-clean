#!/bin/bash

# RR CLI Claude Code Plugin Installation Script

set -e

echo "=================================================="
echo "RR CLI Claude Code Plugin Installation"
echo "=================================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    CLAUDE_CODE_DIR="$HOME/.claude"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CLAUDE_CODE_DIR="$HOME/.claude"
else
    CLAUDE_CODE_DIR="$HOME/.claude"
fi

PLUGINS_DIR="$CLAUDE_CODE_DIR/plugins"
PLUGIN_DEST="$PLUGINS_DIR/rr-cli"

echo "Detecting environment..."
echo "Claude Code directory: $CLAUDE_CODE_DIR"
echo "Plugins directory: $PLUGINS_DIR"
echo ""

# Create plugins directory if it doesn't exist
if [ ! -d "$PLUGINS_DIR" ]; then
    echo "Creating plugins directory..."
    mkdir -p "$PLUGINS_DIR"
fi

# Copy plugin
echo "Installing RR CLI plugin..."
if [ -d "$PLUGIN_DEST" ]; then
    echo "Plugin already exists. Updating..."
    rm -rf "$PLUGIN_DEST"
fi

cp -r "$(dirname "$0")" "$PLUGIN_DEST"

# Make scripts executable
echo "Making scripts executable..."
chmod +x "$PLUGIN_DEST"/commands/*.sh

# Verify RR CLI installation
echo ""
echo "Verifying RR CLI installation..."
if ! command -v rr &> /dev/null; then
    echo "⚠️  RR CLI not found in PATH"
    echo ""
    echo "Install it with:"
    echo "  pip install -e /Volumes/Extreme\ SSD/kelly-project/rr_cli"
    echo ""
else
    echo "✓ RR CLI found: $(which rr)"
fi

# Check OpenAI API key
echo ""
echo "Checking OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set in environment"
    echo ""
    echo "Set it with:"
    echo "  export OPENAI_API_KEY=sk-your-api-key"
    echo ""
    echo "Or create .env file in project root:"
    echo "  OPENAI_API_KEY=sk-your-api-key"
    echo ""
else
    echo "✓ OpenAI API key found"
fi

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Restart Claude Code"
echo "2. Try a command: /rr-help"
echo "3. Or: /rr-status"
echo ""
echo "Available commands:"
echo "  /rr                 - Main RR command"
echo "  /rr-commit          - Generate AI commit"
echo "  /rr-analyze FILE    - Analyze code"
echo "  /rr-suggest FILE    - Get suggestions"
echo "  /rr-explain TOPIC   - Learn concepts"
echo "  /rr-ask QUESTION    - Ask questions"
echo "  /rr-status          - Check git status"
echo "  /rr-help            - Show help"
echo ""
