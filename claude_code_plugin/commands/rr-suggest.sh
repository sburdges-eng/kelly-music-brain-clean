#!/bin/bash
# Get improvement suggestions for code

set -e

FILE="${1:-.}"

if [ ! -f "$FILE" ] && [ "$FILE" != "." ]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

echo "Getting improvement suggestions for $FILE..."
echo ""

rr ai suggest "$FILE"
