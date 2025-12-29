# RR CLI Integration with Claude Code

Complete guide to using the RR CLI tool within Claude Code sessions.

## Overview

RR is a Git + OpenAI integrated CLI tool that allows you to:
- Generate smart commit messages using AI
- Analyze code with intelligent suggestions
- Learn programming concepts on the fly
- Manage git operations with enhanced AI insights

## Quick Setup

### 1. Run Setup Script

```bash
cd /Volumes/Extreme\ SSD/kelly-project
bash setup-rr.sh
```

This will:
- Install the RR CLI tool
- Verify Python installation
- Check for OpenAI API key
- Confirm everything is working

### 2. Set OpenAI API Key

If you haven't already set your OpenAI API key:

**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

**Option B: .env File**
```bash
cd rr_cli
cp .env.example .env
# Edit .env and add your key
```

### 3. Verify Installation

```bash
rr --help
```

Should show the help menu with available commands.

## Using with Claude Code

### In Claude Code Sessions

You now have instant access to powerful AI-assisted development:

#### Generate Commit Messages
When you've made changes:
```bash
git add .
rr ai commit-msg --staged
```

#### Analyze Code
To understand or improve code:
```bash
rr ai analyze multi_task_framework/base.py
rr ai analyze rr_cli/rr/cli.py --analysis-type security
```

#### Get Suggestions
For improvement recommendations:
```bash
rr ai suggest multi_task_framework/heads.py
```

#### Learn Concepts
Ask about programming concepts while coding:
```bash
rr ai explain "multi-task learning"
rr ai explain "backwards compatibility"
rr ai ask "How do I design extensible systems?"
```

#### Check Git Status
Quick repository checks:
```bash
rr git status
rr git log --count 10
rr git diff
```

## Command Reference

### Git Commands

```bash
# Repository status
rr git status [--repo PATH]

# View recent commits
rr git log [--count N]

# Show changes
rr git diff [--staged]
```

### AI Commands

```bash
# Generate commit message from changes
rr ai commit-msg [--staged] [--style conventional]

# Analyze code
rr ai analyze FILE [--analysis-type TYPE]

# Get improvement suggestions
rr ai suggest FILE

# Learn about a concept
rr ai explain "TOPIC"

# Ask a general question
rr ai ask "QUESTION"
```

## Workflow Examples

### Example 1: Smart Commit Workflow

```bash
# Make changes to your code
# ... edit files ...

# Stage your changes
git add .

# Generate AI commit message
rr ai commit-msg --staged

# System will show generated message and ask to confirm
# Type 'y' to commit with the message
```

### Example 2: Code Review Workflow

```bash
# Analyze a new module
rr ai analyze multi_task_framework/encoders.py --analysis-type architecture

# Get specific suggestions for improvements
rr ai suggest multi_task_framework/encoders.py

# Learn about patterns you see
rr ai explain "what is attention fusion in neural networks"
```

### Example 3: Feature Implementation Workflow

```bash
# While implementing a new feature, learn as you go
rr ai explain "task head independence"
rr ai explain "loss balancing in multi-task learning"

# Ask about best practices
rr ai ask "What are best practices for extensible APIs"

# Analyze your implementation
rr ai analyze new_feature.py --analysis-type best-practices

# Generate commit when done
rr ai commit-msg --staged
```

### Example 4: Quick Status Checks

```bash
# Check what changed
rr git status

# See recent work
rr git log --count 5

# View specific changes
rr git diff
rr git diff --staged
```

## Integration with Claude Code Slash Commands

The RR CLI is configured to work seamlessly with Claude Code. In your Claude sessions, you can reference:

- `/rr` - This command guide
- Use `rr` commands directly in terminal

## Tips & Best Practices

### 1. Stage Before Committing
Only stage the files you want to commit:
```bash
git add specific_file.py
rr ai commit-msg --staged
```

### 2. Use Analysis Types
Different analysis types provide different insights:
```bash
rr ai analyze file.py --analysis-type security
rr ai analyze file.py --analysis-type performance
rr ai analyze file.py --analysis-type architecture
```

### 3. Ask Specific Questions
More specific questions get better answers:
```bash
# Good
rr ai ask "What are the principles of dependency injection"

# Better
rr ai ask "How can I implement dependency injection in a Python plugin system"
```

### 4. Learn Concepts First
Before implementing, learn the theory:
```bash
rr ai explain "multi-task learning"
rr ai explain "encoder-decoder architecture"
rr ai ask "How do decoders differ from encoders"
```

### 5. Analyze Before Commit
Check code quality before committing:
```bash
rr ai analyze new_module.py
# Fix any suggestions
git add .
rr ai commit-msg --staged
```

## Troubleshooting

### "Command not found: rr"
**Solution**: Reinstall the CLI tool
```bash
pip install -e /Volumes/Extreme\ SSD/kelly-project/rr_cli
```

### "OPENAI_API_KEY not set"
**Solution**: Set your API key
```bash
export OPENAI_API_KEY=sk-your-key
# Or create .env file in rr_cli directory
```

### "Invalid git repository"
**Solution**: Use --repo flag to specify repository
```bash
rr git status --repo /Volumes/Extreme\ SSD/kelly-project
```

### Slow API responses
**Solution**: This is normal for first requests. Subsequent calls are faster.
- Check your OpenAI account for rate limits
- Ensure good internet connection
- Try simpler queries first

## Advanced Usage

### Custom Commit Styles

```bash
# Conventional commits (default)
rr ai commit-msg --style conventional

# Can be extended to support other styles
```

### Batch Analysis

Analyze multiple files one by one:
```bash
for file in multi_task_framework/*.py; do
    echo "Analyzing $file..."
    rr ai analyze "$file"
done
```

### Integration with Git Hooks

Create a pre-commit hook:
```bash
#!/bin/bash
rr ai analyze $(git diff --cached --name-only --diff-filter=d)
```

## What RR Can Help With

✓ Writing better git commit messages
✓ Understanding code architecture
✓ Finding potential issues before they occur
✓ Learning programming concepts
✓ Getting implementation suggestions
✓ Code review and analysis
✓ Best practices and patterns

## Project Integration

RR is configured for the Kelly Project with knowledge of:
- Multi-task learning framework
- RR CLI tool itself
- Git-based workflow
- Multi-modal encoders
- Task heads and loss balancing

## Support & Resources

- **RR CLI README**: `/Volumes/Extreme SSD/kelly-project/rr_cli/README.md`
- **Multi-task Framework**: `/Volumes/Extreme SSD/kelly-project/multi_task_framework/README.md`
- **Command Help**: `rr --help` or `rr [GROUP] --help`

## Next Steps

1. Run `setup-rr.sh` if you haven't already
2. Try your first commit: `git add . && rr ai commit-msg --staged`
3. Analyze a file: `rr ai analyze multi_task_framework/base.py`
4. Learn a concept: `rr ai explain "multi-task learning"`
5. Ask a question: `rr ai ask "How do I extend the framework"`

Happy coding with AI assistance! 🚀
