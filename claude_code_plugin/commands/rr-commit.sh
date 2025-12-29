#!/bin/bash
# Generate AI commit message and optionally commit

set -e

STYLE="${1:-conventional}"

echo "Generating AI commit message (style: $STYLE)..."
echo ""

rr ai commit-msg --staged --style "$STYLE"
