#!/bin/bash
# RR Teach - Detailed microscopic explanations of programming concepts

set -e

TOPIC="${1:-}"

if [ -z "$TOPIC" ]; then
    echo "Usage: rr-teach TOPIC"
    echo ""
    echo "Get detailed, microscopic-level explanations of programming concepts."
    echo ""
    echo "Examples:"
    echo "  rr-teach 'encoder generalization'"
    echo "  rr-teach 'multi-task learning'"
    echo "  rr-teach 'task head independence'"
    echo "  rr-teach 'loss balancing'"
    echo "  rr-teach 'backwards compatibility'"
    echo ""
    echo "The teach command provides:"
    echo "  • Step-by-step breakdown of concepts"
    echo "  • Code examples and patterns"
    echo "  • Real-world applications"
    echo "  • Common pitfalls and how to avoid them"
    echo "  • Deep-dive technical details"
    echo ""
    exit 1
fi

# Create detailed teaching prompt
TEACHING_PROMPT=$(cat <<EOF
You are an expert programming educator. Provide a DETAILED, MICROSCOPIC-LEVEL explanation of: $TOPIC

Structure your response EXACTLY as follows:

## 📚 CONCEPT OVERVIEW
Brief 1-2 line summary of what this concept is.

## 🔍 MICROSCOPIC BREAKDOWN
Break down the concept into its smallest components:
- List each atomic piece
- Explain how each piece works individually
- Show how pieces interconnect
- Include mathematical notation if relevant

## 🧬 MECHANISM DETAILS
Go deep into HOW it actually works:
- Step-by-step process
- Memory/computation flow
- Data transformations at each stage
- Why each step matters
- What happens if you skip a step

## 💡 CONCRETE CODE EXAMPLE
Provide working Python code showing:
- Minimal but complete implementation
- Detailed comments explaining EVERY line
- Variable naming that shows meaning
- Input → Process → Output flow
- Common variations

## 🏗️ ARCHITECTURE
Show how this fits into larger systems:
- Where it's used
- What depends on it
- What it depends on
- Typical patterns
- When to use vs. alternatives

## ⚠️ CRITICAL DETAILS
Explain the tricky parts:
- Edge cases and how to handle them
- Common mistakes and why they happen
- Performance implications
- Numerical stability issues
- Thread safety considerations (if applicable)

## 🎯 PRACTICAL APPLICATIONS
Real examples of this concept:
- How it's used in production systems
- Industry examples
- Problem it solves
- When NOT to use it
- Alternative approaches and why they differ

## 🧠 MENTAL MODEL
Help visualize and remember:
- Analogies to physical world
- Visual diagrams (ASCII if helpful)
- Mnemonic devices
- Memory aids
- Relationship to other concepts

## 📖 FURTHER LEARNING
Point to deeper understanding:
- Related concepts to study next
- Research papers or resources
- Variations and extensions
- Advanced techniques
- Where experts disagree

Keep explanations PRECISE and DETAILED. Use technical terms correctly. Include specific numbers, formulas, and exact behaviors. Assume intermediate programming knowledge but explain everything thoroughly.

BE MICROSCOPIC: Don't just explain the concept, explain HOW it works at the most fundamental level.
EOF
)

echo "🎓 Teaching: $TOPIC"
echo ""
echo "Generating detailed, microscopic-level explanation..."
echo ""
echo "═══════════════════════════════════════════════════════"
echo ""

rr ai ask "$TEACHING_PROMPT"

echo ""
echo "═══════════════════════════════════════════════════════"
echo ""
echo "💾 To save this explanation: pipe to a file"
echo "   Example: /rr-teach '$TOPIC' > teaching_notes.txt"
