#!/bin/bash
# Explain a programming concept

set -e

TOPIC="${1:-}"

if [ -z "$TOPIC" ]; then
    echo "Usage: rr-explain TOPIC"
    echo "Example: rr-explain 'multi-task learning'"
    exit 1
fi

echo "Explaining: $TOPIC"
echo ""

rr ai explain "$TOPIC"
