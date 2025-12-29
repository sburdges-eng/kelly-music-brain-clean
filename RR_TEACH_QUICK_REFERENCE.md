# /rr-teach Command - Quick Reference

## What It Does

The `/rr-teach` command provides **detailed, microscopic-level explanations** of programming concepts with structured, in-depth learning materials.

## Usage

```bash
/rr-teach "topic"
```

## What You Get

Each teaching session includes 9 comprehensive sections:

### 1. 📚 CONCEPT OVERVIEW
One-line summary of the concept.

### 2. 🔍 MICROSCOPIC BREAKDOWN
Atomic components:
- What makes up the concept
- How pieces connect
- Mathematical notation
- Fundamental principles

### 3. 🧬 MECHANISM DETAILS
How it actually works:
- Step-by-step process
- Memory/computation flow
- Data transformations
- Why each step matters
- Failure modes

### 4. 💡 CONCRETE CODE EXAMPLE
Working, commented Python code showing:
- Minimal complete example
- Line-by-line explanation
- Input → Process → Output
- Common variations

### 5. 🏗️ ARCHITECTURE
System integration:
- Where it's used
- Dependencies
- Typical patterns
- Alternatives

### 6. ⚠️ CRITICAL DETAILS
The tricky parts:
- Edge cases
- Common mistakes
- Performance implications
- Numerical stability
- Boundary conditions

### 7. 🎯 PRACTICAL APPLICATIONS
Real-world examples:
- Production systems
- Industry use
- Problems solved
- When NOT to use it

### 8. 🧠 MENTAL MODEL
Understanding aids:
- Physical analogies
- Diagrams
- Mnemonics
- Visualization

### 9. 📖 FURTHER LEARNING
Deeper resources:
- Related concepts
- Papers
- Extensions
- Advanced techniques

## Quick Examples

### Framework Concepts
```bash
/rr-teach "encoder generalization"
/rr-teach "multi-task learning"
/rr-teach "task head independence"
/rr-teach "loss balancing"
/rr-teach "backwards compatibility"
```

### Implementation Concepts
```bash
/rr-teach "factory pattern for models"
/rr-teach "plugin architecture"
/rr-teach "dynamic task management"
```

### Python/ML Concepts
```bash
/rr-teach "attention mechanisms"
/rr-teach "neural network layers"
/rr-teach "gradient descent"
```

## Save to File

```bash
/rr-teach "topic" > learning_notes.txt
```

## Compare with Other Commands

```bash
/rr-explain "topic"    # Quick overview
/rr-teach "topic"      # Deep, microscopic learning
/rr-ask "question"     # Specific answer
```

## Typical Learning Flow

1. Quick overview:
   ```bash
   /rr-explain "encoder generalization"
   ```

2. Deep understanding:
   ```bash
   /rr-teach "encoder generalization"
   ```

3. See code:
   ```bash
   /rr-analyze multi_task_framework/encoders.py
   ```

4. Get suggestions:
   ```bash
   /rr-suggest multi_task_framework/encoders.py
   ```

## Key Features

✅ **Microscopic Detail** - Breaks down to atomic concepts
✅ **Code Examples** - Complete, commented Python
✅ **Architecture** - Shows system integration
✅ **Edge Cases** - Covers critical details
✅ **Real Examples** - Practical applications
✅ **Mental Models** - Helps understanding
✅ **Resources** - Points to further learning

## Best For

- 📖 Learning programming concepts deeply
- 🔬 Understanding complex systems
- 💻 Code review preparation
- 📚 Interview preparation
- 🎓 Teaching others
- 🔍 Research understanding
- 🏗️ Architecture design

## Time Required

Typically 5-15 minutes to read through complete teaching.

## Tips

1. **Save important teachings** - Make study documents
2. **Read overview first** - Get context
3. **Study code carefully** - Line by line
4. **Review edge cases** - Understand failure modes
5. **Build mental models** - Internalize concepts
6. **Explore resources** - Go deeper

## Installation Note

The teach command is included in the Claude Code plugin. Install with:

```bash
cd claude_code_plugin
bash install.sh
```

Then restart Claude Code to use `/rr-teach`.

## Examples in Action

### Learn Encoder Generalization
```bash
/rr-teach "encoder generalization"

# Returns:
# - What it is (overview)
# - Components (microscopic breakdown)
# - How it works (mechanism)
# - Example code (implementation)
# - Where it's used (architecture)
# - What can go wrong (critical details)
# - Real examples (applications)
# - How to think about it (mental models)
# - Learn more about (resources)
```

### Learn Multi-Task Learning
```bash
/rr-teach "multi-task learning"

# Returns comprehensive explanation with:
# - Why it matters
# - What makes it work
# - How to implement
# - Common pitfalls
# - Real ML systems
# - When to use
```

### Save for Later
```bash
/rr-teach "loss balancing" > loss_balancing_notes.md

# Can then:
# - Review later
# - Share with team
# - Reference while coding
# - Study offline
```

## Quick Commands Summary

| Command | Use For |
|---------|---------|
| `/rr-help` | Show command list |
| `/rr-explain TOPIC` | Quick overview |
| `/rr-teach TOPIC` | Deep learning |
| `/rr-analyze FILE` | Code analysis |
| `/rr-ask QUESTION` | Specific answers |
| `/rr-commit` | AI commit messages |

## Next Steps

1. Install the plugin: `bash claude_code_plugin/install.sh`
2. Restart Claude Code
3. Try a teaching: `/rr-teach "encoder generalization"`
4. Save the output: `/rr-teach "topic" > notes.txt`
5. Use while learning!

---

**The /rr-teach command is your personal AI instructor for deep programming concept understanding!** 🎓
