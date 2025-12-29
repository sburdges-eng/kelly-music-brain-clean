# Quick Start: Using RR with Claude Code

## 1-Minute Setup

```bash
# From project root
bash setup-rr.sh
export OPENAI_API_KEY=sk-your-api-key
```

## Instant Commands

```bash
# Generate AI commit message
rr ai commit-msg --staged

# Analyze code
rr ai analyze file.py

# Get suggestions
rr ai suggest file.py

# Learn concepts
rr ai explain "topic"

# Ask questions
rr ai ask "question"

# Check git
rr git status
```

## Common Workflows

### Smart Commits
```bash
git add .
rr ai commit-msg --staged
# Confirm to commit
```

### Code Review
```bash
rr ai analyze multi_task_framework/base.py
rr ai suggest multi_task_framework/heads.py
```

### Learning
```bash
rr ai explain "multi-task learning"
rr ai ask "How do I implement X"
```

## In Claude Code

All `rr` commands work directly in Claude Code terminal:

```bash
rr --help                    # Show help
rr git status               # Git operations
rr ai commit-msg --staged   # AI commit
rr ai analyze FILE          # Code analysis
rr ai suggest FILE          # Suggestions
rr ai explain TOPIC         # Learn
rr ai ask QUESTION          # Ask questions
```

## Documentation

- **Full Guide**: `RR_CLAUDE_INTEGRATION.md`
- **CLI Tool**: `rr_cli/README.md`
- **Framework**: `multi_task_framework/README.md`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `rr: command not found` | Run `pip install -e rr_cli` |
| `OPENAI_API_KEY not set` | Run `export OPENAI_API_KEY=sk-...` |
| `Invalid git repository` | Use `--repo /path` flag |
| Slow responses | Normal for first requests |

## Next Steps

1. ✅ Run `bash setup-rr.sh`
2. ✅ Try `rr git status`
3. ✅ Try `rr ai commit-msg --staged`
4. ✅ Try `rr ai explain "encoder"`
5. ✅ Read `RR_CLAUDE_INTEGRATION.md` for full guide

Enjoy AI-powered development! 🚀
