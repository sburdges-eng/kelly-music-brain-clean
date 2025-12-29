# 🚀 START HERE - Complete Installation & Setup Guide

Welcome! You have a fully installed, production-ready system. Here's what to do next.

## Current Status

```
✅ RR CLI Plugin           - Installed to ~/.claude/plugins/rr-cli/
✅ RR CLI Tool            - Installed globally
✅ 9 Slash Commands       - Ready to use
✅ Multi-Task Framework   - Available
⏳ OpenAI API Key        - YOUR TURN (30 seconds)
⏳ Claude Code Restart    - YOUR TURN
```

## Step 1: Get OpenAI API Key (2 minutes)

1. Go to: https://platform.openai.com/api-keys
2. Sign in to your OpenAI account
3. Click "Create new secret key"
4. Copy the key (you'll only see it once!)
5. Keep it safe for the next step

## Step 2: Set API Key (Choose ONE method)

### Method A: Quick Terminal (Temporary)
```bash
export OPENAI_API_KEY=sk-paste-your-key-here
```

### Method B: Permanent (Recommended)
```bash
echo 'export OPENAI_API_KEY=sk-paste-your-key-here' >> ~/.zshrc
source ~/.zshrc
```

Or for Bash:
```bash
echo 'export OPENAI_API_KEY=sk-paste-your-key-here' >> ~/.bashrc
source ~/.bashrc
```

### Method C: .env File
```bash
cd /Volumes/Extreme\ SSD/kelly-project
echo 'OPENAI_API_KEY=sk-paste-your-key-here' > .env
```

## Step 3: Restart Claude Code

1. **Close Claude Code completely** (not just the window)
2. **Wait 3 seconds**
3. **Reopen Claude Code**

## Step 4: Test Your Commands

In Claude Code, try these:

```bash
# No API key needed (test these first):
/rr-help                           # Show all commands
/rr-status                         # Check git status

# Need API key (test after restart):
/rr-teach "hello world"            # Get detailed teaching
/rr-explain "encoder"              # Quick explanation
/rr-commit                         # Generate AI commit
```

## What You Have

### 9 Slash Commands

| Command | Purpose | API Key Required |
|---------|---------|------------------|
| `/rr` | Run any RR CLI command | No |
| `/rr-teach TOPIC` | **Detailed microscopic teaching** ⭐ | Yes |
| `/rr-explain TOPIC` | Quick explanation | Yes |
| `/rr-commit` | Generate AI commit | Yes |
| `/rr-analyze FILE` | Analyze code | Yes |
| `/rr-suggest FILE` | Improvement suggestions | Yes |
| `/rr-ask QUESTION` | Ask questions | Yes |
| `/rr-status` | Git status | No |
| `/rr-help` | Show help | No |

### RR CLI Tool

Use directly in terminal:
```bash
rr git status          # Check repo
rr git log             # View commits
rr git diff            # Show changes
rr ai commit-msg       # Generate commit
```

### Multi-Task Framework

Use in Python:
```python
from multi_task_framework import MultiTaskModelFactory

model = MultiTaskModelFactory.build_model(
    task_configs=configs,
    encoder_kwargs=encoder_settings
)
```

## Documentation Files

- **`TEACH_COMMAND_GUIDE.md`** - Deep guide to /rr-teach (2000+ lines)
- **`RR_TEACH_QUICK_REFERENCE.md`** - Quick reference card
- **`QUICK_START_RR.md`** - Getting started
- **`INSTALLATION_COMPLETE.md`** - Installation details
- **`RR_SUMMARY.md`** - Complete overview
- **`PLUGIN_INSTALLATION_GUIDE.md`** - Plugin guide
- **`RR_CLAUDE_INTEGRATION.md`** - Integration guide

## Example: Using /rr-teach

After setup, try:

```bash
/rr-teach "encoder generalization"
```

You'll get a comprehensive explanation with:
- Concept overview
- Microscopic breakdown
- Mechanism details with code
- Architecture integration
- Critical edge cases
- Real-world applications
- Mental models
- Further learning resources

## Troubleshooting

### Issue: "Unknown slash command"
**Solution**: Restart Claude Code after setting API key

### Issue: "OPENAI_API_KEY not set"
**Solution**: Set the key using one of the methods above

### Issue: Git commands not working
**Solution**: Make sure you're in a git repository

### Issue: Plugin commands disappeared
**Solution**: Restart Claude Code

## Project Structure

```
/Volumes/Extreme\ SSD/kelly-project/
├── rr_cli/                           # RR CLI Tool (installed globally)
├── multi_task_framework/             # Deep learning framework
├── claude_code_plugin/               # Plugin for Claude Code
├── 00_START_HERE.md                 # ← You are here
├── TEACH_COMMAND_GUIDE.md           # Learn about /rr-teach
├── INSTALLATION_COMPLETE.md         # Installation details
└── [other documentation files]
```

## Next: Try /rr-teach

Once you've set your API key and restarted Claude Code, try:

```bash
/rr-teach "multi-task learning"
/rr-teach "encoder generalization"
/rr-teach "loss balancing"
```

These provide **detailed, microscopic-level explanations** perfect for:
- 📚 Deep learning of concepts
- 💻 Understanding architecture
- 🔬 Interview preparation
- 🎓 Teaching others
- 📖 Research understanding

## Quick Command Examples

### Generate Smart Commits
```bash
git add .
/rr-commit              # AI generates commit message
# Review and confirm
```

### Analyze Code
```bash
/rr-analyze multi_task_framework/base.py
/rr-suggest multi_task_framework/heads.py
```

### Learn While Coding
```bash
/rr-teach "backward compatibility"
/rr-ask "how do I implement this?"
```

### Check Status
```bash
/rr-status              # Git status
```

## Summary

**You're all set!** Just need to:

1. ✅ Get OpenAI API key (2 min)
2. ✅ Set environment variable (30 sec)
3. ✅ Restart Claude Code (10 sec)
4. 🎉 Start using commands!

## Support

- **Full docs**: Read the markdown files in project root
- **Teach command**: See `TEACH_COMMAND_GUIDE.md`
- **Quick ref**: See `RR_TEACH_QUICK_REFERENCE.md`
- **Installation**: See `INSTALLATION_COMPLETE.md`

## Verification Checklist

After API key setup and restart, verify:

- [ ] `/rr-help` shows command list
- [ ] `/rr-status` shows git status
- [ ] `/rr-teach "test"` returns detailed explanation
- [ ] `/rr-commit` can generate messages
- [ ] `/rr-analyze file.py` analyzes code

Once all checked, you're ready to go! 🚀

---

**Questions?** See the documentation files - everything is explained in detail.

**Ready?** Set your API key and restart Claude Code!

**Go!** Try `/rr-teach "hello"` in Claude Code! 🎓
