#!/bin/bash
# Analyze code with AI

set -e

FILE="${1:-.}"
TYPE="${2:-general}"

if [ ! -f "$FILE" ] && [ "$FILE" != "." ]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

echo "Analyzing $FILE (type: $TYPE)..."
echo ""

rr ai analyze "$FILE" --analysis-type "$TYPE"
