# Anthropic Claude API Setup

## Overview

The RR CLI tool now uses **Anthropic's Claude API** instead of OpenAI. This provides access to Claude's advanced reasoning and learning capabilities through the `/rr-teach` command and other features.

## Get Your Anthropic API Key

1. **Visit**: https://console.anthropic.com/
2. **Sign in** or create an Anthropic account
3. **Navigate** to API Keys section
4. **Create** a new API key
5. **Copy** the key (starts with `sk-ant-`)
6. **Keep it safe** - treat it like a password

## Configure the .env File

### File Location
```
/Volumes/Extreme SSD/kelly-project/.env
```

### Edit the File

Replace the placeholder with your actual API key:

**Before:**
```
ANTHROPIC_API_KEY=sk-ant-your-actual-api-key-here
```

**After:**
```
ANTHROPIC_API_KEY=sk-ant-abc123def456ghi789jkl...
```

### Methods to Set

#### Method 1: Direct File Edit (Simplest)
```bash
# Open with your editor
open -e "/Volumes/Extreme SSD/kelly-project/.env"

# Replace placeholder with your actual key, then save
```

#### Method 2: Terminal Command
```bash
cat > "/Volumes/Extreme SSD/kelly-project/.env" << 'EOFKEY'
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
EOFKEY
```

## Install Required Dependencies

The RR CLI tool needs the Anthropic Python SDK. Install it:

```bash
cd "/Volumes/Extreme SSD/kelly-project/rr_cli"
pip install -e .
```

This will install:
- `anthropic>=0.7.0` - Anthropic Python SDK
- `click>=8.0.0` - CLI framework
- `gitpython>=3.1.0` - Git integration
- `python-dotenv>=0.19.0` - Environment variable loading

## Verify Installation

Test that everything is working:

```bash
# Check the RR CLI is installed
which rr

# Verify Anthropic SDK is available
python -c "from anthropic import Anthropic; print('✅ Anthropic SDK installed')"
```

## Restart Claude Code

After setting your API key:

1. **Close** Claude Code completely
2. **Wait** 3 seconds
3. **Reopen** Claude Code

## Test Your Setup

In Claude Code, try these commands:

```bash
# Test without API key (should work):
/rr-help
/rr-status

# Test with API key (should work after restart):
/rr-teach "hello world"
/rr-explain "encoder generalization"
/rr-commit
```

## API Key Format

Your Anthropic API key should:
- Start with `sk-ant-`
- Be a long alphanumeric string (100+ characters)
- Not have quotes around it in the .env file

Example valid entry:
```
ANTHROPIC_API_KEY=sk-ant-fF7BvFU5Xn8nLz9pQ4tRwYzAbCdEfGhIjKlMnOpQrStUvWxYz1A2B3C4D5E6F7G8H
```

## Available Models

The RR CLI uses `claude-3-5-sonnet-20241022` which provides:
- Fast responses
- Excellent reasoning
- Great for code analysis and explanations
- Cost-effective

## What You Can Do

### Learning Features (Requires API Key)
```bash
/rr-teach "topic"          # Deep microscopic-level learning
/rr-explain "topic"        # Quick concept explanation
/rr-ask "question"         # Ask programming questions
```

### Code Features (Requires API Key)
```bash
/rr-analyze file.py        # Analyze code
/rr-suggest file.py        # Get improvement suggestions
/rr-commit                 # Generate smart commits
```

### Git Features (No API Key Needed)
```bash
/rr-status                 # Check git status
/rr git log                # View git log
/rr git diff               # Show changes
```

## Troubleshooting

### Issue: "ANTHROPIC_API_KEY not set"
**Solution**:
1. Check .env file exists: `cat /Volumes/Extreme\ SSD/kelly-project/.env`
2. Verify key is there (not placeholder)
3. Verify key starts with `sk-ant-`
4. Restart Claude Code

### Issue: "ModuleNotFoundError: No module named 'anthropic'"
**Solution**:
```bash
cd "/Volumes/Extreme SSD/kelly-project/rr_cli"
pip install -e .
```

### Issue: "Connection error" when using commands
**Solution**:
1. Verify API key is correct: `cat /Volumes/Extreme\ SSD/kelly-project/.env`
2. Check key format (should start with `sk-ant-`)
3. Verify internet connection
4. Check API key is active in Anthropic console

### Issue: Commands time out
**Solution**:
- This is normal for longer requests (especially `/rr-teach`)
- The command may take 30-60 seconds for detailed responses
- Check your internet connection

## Security Notes

⚠️ **Important**:
- Never commit .env to git (already protected in .gitignore)
- Never share your API key
- Treat API key like a password
- Store safely - only see it once when created

## Differences from OpenAI

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| Model Used | gpt-4 | claude-3-5-sonnet |
| Cost | Higher | Competitive |
| Reasoning | Good | Excellent |
| Code Analysis | Good | Excellent |
| Teaching | Good | Excellent |
| Speed | Moderate | Fast |

## Next Steps

1. **Get API Key** from https://console.anthropic.com/
2. **Add to .env** file
3. **Install dependencies**: `pip install -e rr_cli`
4. **Restart Claude Code**
5. **Test commands**: `/rr-teach "hello"`

## Support

For issues:
- Check this guide (ANTHROPIC_API_SETUP.md)
- See SETUP_STATUS.md for installation overview
- Read 00_START_HERE.md for quick reference
- Check troubleshooting section above

---

**Your system is ready to use Anthropic Claude for all AI features!** 🚀

