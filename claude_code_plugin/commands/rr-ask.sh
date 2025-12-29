#!/bin/bash
# Ask a question to the AI

set -e

QUESTION="${1:-}"

if [ -z "$QUESTION" ]; then
    echo "Usage: rr-ask QUESTION"
    echo "Example: rr-ask 'How do I implement custom encoders?'"
    exit 1
fi

echo "Question: $QUESTION"
echo ""

rr ai ask "$QUESTION"
