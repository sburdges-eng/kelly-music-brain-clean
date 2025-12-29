#!/bin/bash
# Main RR CLI command handler for Claude Code plugin

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get command and subcommand
COMMAND="${1:-help}"
SUBCOMMAND="${2:-}"

# Show help if no arguments
if [ "$COMMAND" = "help" ] || [ "$COMMAND" = "--help" ] || [ -z "$COMMAND" ]; then
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}           ${BLUE}RR CLI - Git + OpenAI Integration${NC}           ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Usage: rr [COMMAND] [SUBCOMMAND] [OPTIONS]"
    echo ""
    echo -e "${GREEN}Git Commands:${NC}"
    echo "  rr git status          Show repository status"
    echo "  rr git log             View recent commits"
    echo "  rr git diff            Show changes"
    echo ""
    echo -e "${GREEN}AI Commands:${NC}"
    echo "  rr ai commit-msg       Generate AI commit message"
    echo "  rr ai analyze FILE     Analyze code with AI"
    echo "  rr ai suggest FILE     Get improvement suggestions"
    echo "  rr ai explain TOPIC    Learn about a concept"
    echo "  rr ai ask QUESTION     Ask a question"
    echo ""
    echo -e "${GREEN}Quick Commands:${NC}"
    echo "  rr-commit              Generate and commit with AI message"
    echo "  rr-analyze FILE        Analyze FILE"
    echo "  rr-suggest FILE        Suggest improvements for FILE"
    echo "  rr-explain TOPIC       Explain TOPIC"
    echo "  rr-ask QUESTION        Ask a question"
    echo "  rr-status              Check git status"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  rr git status"
    echo "  rr ai commit-msg --staged"
    echo "  rr ai analyze src/main.py"
    echo "  rr ai explain 'multi-task learning'"
    echo ""
    exit 0
fi

# Pass all arguments to RR CLI
if ! command -v rr &> /dev/null; then
    echo -e "${YELLOW}Error: RR CLI not found${NC}"
    echo "Install it with: pip install -e /path/to/rr_cli"
    exit 1
fi

# Execute the RR CLI command
rr "$@"
