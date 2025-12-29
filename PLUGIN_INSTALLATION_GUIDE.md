# RR CLI Claude Code Plugin - Installation Guide

Complete guide to installing and using the RR CLI plugin with Claude Code.

## What You're Installing

A Claude Code plugin that provides custom slash commands for the RR CLI tool:

```
/rr          - Run any RR CLI command
/rr-commit   - Generate AI commit message
/rr-analyze  - Analyze code with AI
/rr-suggest  - Get improvement suggestions
/rr-explain  - Learn programming concepts
/rr-ask      - Ask questions
/rr-status   - Check git status
/rr-help     - Show help
```

## Prerequisites

Before installing the plugin, ensure you have:

1. **Claude Code** - Latest version installed
2. **Python 3.8+** - For running RR CLI
3. **Git** - For version control operations
4. **OpenAI API Key** - For AI features (get from https://platform.openai.com/api-keys)

## Step-by-Step Installation

### Step 1: Verify RR CLI Installation

First, make sure the RR CLI tool is installed:

```bash
# Check if RR is installed
which rr

# If not found, install it
cd /Volumes/Extreme\ SSD/kelly-project/rr_cli
pip install -e .
```

Verify installation:
```bash
rr --help
```

### Step 2: Set OpenAI API Key

Choose one method:

**Method A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY=sk-your-api-key-here

# Add to ~/.bashrc or ~/.zshrc to make persistent
echo 'export OPENAI_API_KEY=sk-your-api-key-here' >> ~/.zshrc
source ~/.zshrc
```

**Method B: .env File**
```bash
cd /Volumes/Extreme\ SSD/kelly-project
cat > .env << EOF
OPENAI_API_KEY=sk-your-api-key-here
EOF
```

**Method C: In rr_cli directory**
```bash
cd /Volumes/Extreme\ SSD/kelly-project/rr_cli
cp .env.example .env
# Edit .env and add your API key
```

### Step 3: Install the Plugin

#### Option A: Automatic Installation (Recommended)

```bash
cd /Volumes/Extreme\ SSD/kelly-project/claude_code_plugin
bash install.sh
```

This will:
- Create the plugins directory if needed
- Copy the plugin to `~/.claude/plugins/rr-cli`
- Make scripts executable
- Verify dependencies

#### Option B: Manual Installation

```bash
# Create plugins directory
mkdir -p ~/.claude/plugins

# Copy plugin
cp -r /Volumes/Extreme\ SSD/kelly-project/claude_code_plugin \
      ~/.claude/plugins/rr-cli

# Make scripts executable
chmod +x ~/.claude/plugins/rr-cli/commands/*.sh
```

### Step 4: Restart Claude Code

Close and reopen Claude Code completely to load the new plugin.

### Step 5: Verify Installation

In Claude Code, try:

```bash
/rr-help
```

You should see the RR CLI help menu. If it works, installation is complete!

## Troubleshooting

### Issue: "Unknown slash command: rr"

**Cause**: Plugin wasn't loaded properly

**Solution**:
1. Check plugin is in `~/.claude/plugins/rr-cli/`
2. Verify `manifest.json` exists
3. Completely restart Claude Code
4. Check Claude Code logs for errors

### Issue: "Command not found: rr"

**Cause**: RR CLI not installed or not in PATH

**Solution**:
```bash
# Install RR CLI
pip install -e /Volumes/Extreme\ SSD/kelly-project/rr_cli

# Verify it's in PATH
which rr

# If not found, add to PATH
export PATH="/path/to/python/site-packages/bin:$PATH"
```

### Issue: "OPENAI_API_KEY not set"

**Cause**: API key not configured

**Solution**:
```bash
# Set environment variable
export OPENAI_API_KEY=sk-your-api-key

# Or create .env file
cd /Volumes/Extreme\ SSD/kelly-project
echo "OPENAI_API_KEY=sk-your-api-key" > .env
```

### Issue: Permission denied on scripts

**Cause**: Scripts aren't executable

**Solution**:
```bash
chmod +x ~/.claude/plugins/rr-cli/commands/*.sh
```

### Issue: Plugin commands aren't showing up

**Cause**: manifest.json not recognized

**Solution**:
1. Verify manifest.json syntax (valid JSON)
2. Check file is in plugin root directory
3. Restart Claude Code
4. Check folder permissions

## Verifying Installation

### Check Plugin Directory

```bash
ls -la ~/.claude/plugins/rr-cli/
```

Should show:
- `manifest.json` - Plugin configuration
- `README.md` - Documentation
- `install.sh` - Installation script
- `commands/` - Command scripts

### Check Executable Scripts

```bash
ls -la ~/.claude/plugins/rr-cli/commands/
```

All `.sh` files should have `x` (executable) permissions.

### Test Each Command

In Claude Code:

```bash
/rr-help              # Should show help
/rr-status            # Should show git status
/rr-ask "test"        # Should ask a question
```

## Usage Examples

Once installed, you can use these commands in Claude Code:

### Git Operations

```bash
# Check repository status
/rr-status

# View git status
/rr git status

# View recent commits
/rr git log --count 5
```

### Generate Commits

```bash
# Stage changes first
git add .

# Generate AI commit message
/rr-commit

# Or specify style
/rr ai commit-msg --staged --style conventional
```

### Analyze Code

```bash
# Analyze a file
/rr-analyze multi_task_framework/base.py

# Analyze with specific type
/rr ai analyze file.py --analysis-type security
```

### Get Suggestions

```bash
# Get improvement suggestions
/rr-suggest multi_task_framework/heads.py

# Or directly
/rr ai suggest file.py
```

### Learn Concepts

```bash
# Explain a concept
/rr-explain "multi-task learning"

# Or directly
/rr ai explain "encoder generalization"
```

### Ask Questions

```bash
# Ask a question
/rr-ask "How do I implement custom heads?"

# Or directly
/rr ai ask "What is backwards compatibility?"
```

## Advanced Configuration

### Custom Command Paths

If RR CLI is in a non-standard location, update your shell configuration:

```bash
# In ~/.zshrc or ~/.bashrc
export PATH="/custom/path/to/rr:$PATH"
```

### Custom Plugin Location

To use plugin from a different location:

```bash
# Create symlink instead of copy
ln -s /Volumes/Extreme\ SSD/kelly-project/claude_code_plugin \
      ~/.claude/plugins/rr-cli
```

### Environment-Specific Configuration

Create different `.env` files for different environments:

```bash
# For development
cat > .env.development << EOF
OPENAI_API_KEY=sk-dev-key
EOF

# For production
cat > .env.production << EOF
OPENAI_API_KEY=sk-prod-key
EOF
```

## Uninstalling

To remove the plugin:

```bash
rm -rf ~/.claude/plugins/rr-cli
```

Then restart Claude Code.

## Support

### Documentation

- **Plugin README**: `claude_code_plugin/README.md`
- **RR CLI README**: `rr_cli/README.md`
- **Integration Guide**: `RR_CLAUDE_INTEGRATION.md`
- **Quick Start**: `QUICK_START_RR.md`

### Getting Help

1. Check troubleshooting section above
2. Review plugin README: `claude_code_plugin/README.md`
3. Review RR CLI README: `rr_cli/README.md`
4. Check Claude Code documentation

## Next Steps After Installation

1. **Verify it works**: `/rr-help`
2. **Check git status**: `/rr-status`
3. **Generate a commit**: `git add . && /rr-commit`
4. **Learn a concept**: `/rr-explain "multi-task learning"`
5. **Analyze code**: `/rr-analyze rr_cli/rr/cli.py`

## System Requirements

| Component | Requirement |
|-----------|-------------|
| Claude Code | Latest version |
| Python | 3.8+ |
| Git | 2.0+ |
| pip | Latest |
| OpenAI API Key | Required for AI features |

## FAQs

**Q: Do I need to reinstall the plugin after updates?**
A: If using symlink (custom install), no. If copied, yes. Easiest is to reinstall via `bash install.sh`.

**Q: Can I use the plugin on multiple machines?**
A: Yes, install on each machine. Each needs its own OpenAI API key configured.

**Q: Does the plugin work offline?**
A: Git operations work offline. AI features require internet and OpenAI API access.

**Q: Can I modify the commands?**
A: Yes! Edit the scripts in `claude_code_plugin/commands/` and they'll be used immediately.

**Q: How do I update the plugin?**
A: Run `bash install.sh` again, or manually copy new files.

## Success Indicator

Installation was successful if:

✅ `/rr-help` shows the RR CLI help menu
✅ `/rr-status` shows your git status
✅ `/rr-ask "test"` gets a response from AI
✅ All command scripts are in `~/.claude/plugins/rr-cli/commands/`

## Ready to Use!

Once installation is complete and verified, you're ready to use RR CLI with Claude Code. Enjoy AI-powered development! 🚀
