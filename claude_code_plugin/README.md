# RR CLI Claude Code Plugin

A Claude Code plugin that integrates the RR CLI tool (Git + OpenAI) directly into your Claude Code environment.

## Overview

This plugin provides seamless access to RR CLI commands within Claude Code sessions through custom slash commands:

- `/rr` - Run any RR CLI command
- `/rr-commit` - Generate and commit with AI message
- `/rr-analyze` - Analyze code with AI
- `/rr-suggest` - Get improvement suggestions
- `/rr-explain` - Learn programming concepts
- `/rr-teach` - Detailed microscopic-level explanations
- `/rr-ask` - Ask questions
- `/rr-status` - Check git status
- `/rr-help` - Show help

## Installation

### 1. Install RR CLI Tool

First, ensure the RR CLI tool is installed:

```bash
cd /path/to/rr_cli
pip install -e .
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

Or create a `.env` file in the project root.

### 3. Make Scripts Executable

```bash
chmod +x /path/to/claude_code_plugin/commands/*.sh
```

### 4. Register Plugin with Claude Code

There are several ways to register this plugin with Claude Code:

#### Option A: Local Plugin (Recommended for Development)

1. Copy the plugin to your Claude Code plugins directory:
```bash
cp -r /path/to/claude_code_plugin ~/.claude/plugins/rr-cli
```

2. Restart Claude Code

#### Option B: Via Claude Code Settings

1. Open Claude Code settings
2. Navigate to Plugins section
3. Add local plugin path: `/path/to/claude_code_plugin`
4. Enable the plugin

#### Option C: From Repository

If you have the plugin hosted on GitHub:

```
https://github.com/your-org/rr-claude-plugin
```

## Usage

Once installed, you can use the slash commands in Claude Code:

### Git Operations

```bash
/rr git status
/rr git log --count 5
/rr git diff
```

Or use the quick command:

```bash
/rr-status
```

### AI-Powered Commits

```bash
/rr-commit
# Or with style option
/rr ai commit-msg --staged --style conventional
```

### Code Analysis

```bash
/rr-analyze multi_task_framework/base.py
# Or with analysis type
/rr ai analyze file.py --analysis-type security
```

### Get Suggestions

```bash
/rr-suggest multi_task_framework/heads.py
# Or directly
/rr ai suggest file.py
```

### Learn Concepts (Simple Explanation)

```bash
/rr-explain "multi-task learning"
# Or directly
/rr ai explain "encoder generalization"
```

### Detailed Teaching (Microscopic Level)

For in-depth understanding with code examples, architecture details, and deep dives:

```bash
/rr-teach "encoder generalization"
/rr-teach "multi-task learning"
/rr-teach "task head independence"
/rr-teach "loss balancing"
```

The teach command provides:
- Step-by-step concept breakdown
- Detailed code examples with comments
- Architecture and system integration
- Critical edge cases and pitfalls
- Real-world applications
- Mental models for understanding
- Further learning resources

### Ask Questions

```bash
/rr-ask "How do I implement custom task heads?"
# Or directly
/rr ai ask "What is backwards compatibility?"
```

### Help

```bash
/rr-help
# Or
/rr --help
```

## Commands Reference

### Main Command: `/rr`

Runs any RR CLI command directly.

```bash
/rr [COMMAND] [SUBCOMMAND] [OPTIONS]
```

**Examples:**
- `/rr git status`
- `/rr ai commit-msg --staged`
- `/rr ai explain "topic"`

### Quick Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `/rr-commit` | Generate AI commit message | `/rr-commit [style]` |
| `/rr-analyze` | Analyze code | `/rr-analyze FILE [type]` |
| `/rr-suggest` | Get suggestions | `/rr-suggest FILE` |
| `/rr-explain` | Learn concept (simple) | `/rr-explain TOPIC` |
| `/rr-teach` | Learn concept (detailed) | `/rr-teach TOPIC` |
| `/rr-ask` | Ask question | `/rr-ask QUESTION` |
| `/rr-status` | Check git status | `/rr-status` |
| `/rr-help` | Show help | `/rr-help` |

## Common Workflows

### Smart Commit Workflow

1. Make your changes
2. Stage changes: `git add .`
3. Run: `/rr-commit`
4. Review generated message
5. Confirm to commit

### Code Review Workflow

```bash
/rr-analyze multi_task_framework/base.py
# Read suggestions
/rr-suggest multi_task_framework/base.py
# Make improvements
/rr-commit
```

### Learning While Coding

```bash
/rr-explain "task head independence"
/rr-ask "How do I implement custom encoders?"
/rr-analyze my_new_module.py --analysis-type architecture
```

### Quick Status Check

```bash
/rr-status
```

## Configuration

### Environment Variables

The plugin uses these environment variables:

- `OPENAI_API_KEY` - OpenAI API key (required for AI features)
- `PATH` - Must include directory where `rr` command is located

### .env File

Create `.env` in your project root:

```
OPENAI_API_KEY=sk-your-key-here
```

### Custom Paths

If RR CLI is in a non-standard location, update your PATH:

```bash
export PATH="/custom/path/to/rr/bin:$PATH"
```

## Troubleshooting

### "Command not found: rr"

**Solution:** Ensure RR CLI is installed and in PATH

```bash
pip install -e /path/to/rr_cli
which rr  # Verify installation
```

### "OPENAI_API_KEY not set"

**Solution:** Set the environment variable

```bash
export OPENAI_API_KEY=sk-your-key
# Or create .env file
```

### Plugin not showing up in Claude Code

**Solution:**

1. Check plugin is in correct directory
2. Restart Claude Code
3. Verify manifest.json is valid JSON
4. Check Claude Code logs for errors

### Scripts not executable

**Solution:** Make scripts executable

```bash
chmod +x claude_code_plugin/commands/*.sh
```

## Architecture

```
claude_code_plugin/
├── manifest.json           # Plugin manifest
├── README.md               # This file
└── commands/
    ├── rr.sh              # Main command handler
    ├── rr-commit.sh       # Commit command
    ├── rr-analyze.sh      # Analysis command
    ├── rr-suggest.sh      # Suggestions command
    ├── rr-explain.sh      # Learning command
    ├── rr-ask.sh          # Questions command
    ├── rr-status.sh       # Git status command
    └── rr-help.sh         # Help command
```

## Development

### Adding New Commands

1. Create new script in `commands/` directory
2. Add entry to `manifest.json`
3. Make script executable: `chmod +x commands/script.sh`
4. Restart Claude Code

### Example: New Command

Create `commands/rr-custom.sh`:

```bash
#!/bin/bash
set -e
echo "Custom command running..."
rr [your command here]
```

Add to `manifest.json`:

```json
{
  "name": "rr-custom",
  "description": "Custom RR command",
  "script": "./commands/rr-custom.sh",
  "arguments": ["arg1"]
}
```

## Requirements

- Python 3.8+
- Git 2.0+
- RR CLI tool installed
- OpenAI API key
- Claude Code (latest version)

## Support

For issues with:

- **RR CLI Tool**: See `rr_cli/README.md`
- **Multi-Task Framework**: See `multi_task_framework/README.md`
- **Claude Code Integration**: See `RR_CLAUDE_INTEGRATION.md`
- **This Plugin**: Check troubleshooting section above

## License

MIT License - Same as RR CLI Tool

## Related Documentation

- **RR CLI Tool**: `/rr_cli/README.md`
- **Integration Guide**: `RR_CLAUDE_INTEGRATION.md`
- **Quick Start**: `QUICK_START_RR.md`
- **Framework**: `multi_task_framework/README.md`
