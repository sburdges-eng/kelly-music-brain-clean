# OpenAI API Key Setup

## What You Need

1. Your OpenAI API key (see "Get Your API Key" below)
2. A text editor
3. The .env file already created in the project root

## Get Your OpenAI API Key

1. **Visit**: https://platform.openai.com/api-keys
2. **Sign in** with your OpenAI account
3. **Create a new secret key** by clicking "Create new secret key"
4. **Copy the key immediately** - you'll only see it once!
5. **Keep it safe** - this key grants access to your OpenAI account

## Configure the .env File

### Method 1: Direct Edit (Simplest)

1. **Find the .env file**:
   ```
   /Volumes/Extreme SSD/kelly-project/.env
   ```

2. **Open with your editor** (VSCode, TextEdit, etc.):
   ```bash
   open -e "/Volumes/Extreme SSD/kelly-project/.env"
   ```

3. **Edit the line**:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

4. **Replace with your actual key**:
   ```
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxx
   ```

5. **Save the file** (Cmd+S)

### Method 2: Terminal Command

If you prefer command line:

```bash
cat > "/Volumes/Extreme SSD/kelly-project/.env" << 'EOFKEY'
OPENAI_API_KEY=sk-your-actual-api-key-here
EOFKEY
```

Then edit the file to add your actual key.

## Verify Setup

After editing the .env file, verify it's correct:

```bash
# Check file exists
cat "/Volumes/Extreme SSD/kelly-project/.env"

# Should output:
# OPENAI_API_KEY=sk-your-actual-api-key-here
# (with your actual key, not the placeholder)
```

## Next: Restart Claude Code

1. **Close Claude Code completely** (not just the window)
2. **Wait 3 seconds**
3. **Reopen Claude Code**

## Test Your Setup

Once restarted, try these commands in Claude Code:

```bash
# Test without API key (should work):
/rr-help
/rr-status

# Test with API key (should work after restart):
/rr-teach "hello"
/rr-explain "encoder"
/rr-commit
```

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"
**Solution**: 
- Check .env file exists: `ls /Volumes/Extreme\ SSD/kelly-project/.env`
- Check key is set: `cat /Volumes/Extreme\ SSD/kelly-project/.env`
- Restart Claude Code

### Issue: "Permission denied" on .env file
**Solution**:
```bash
chmod 644 "/Volumes/Extreme SSD/kelly-project/.env"
```

### Issue: Key doesn't work after restart
**Solution**:
- Verify key format starts with `sk-`
- Double-check no extra spaces in .env file
- Make sure .env file has no trailing newlines
- Try with a fresh API key from platform.openai.com

## Security Note

⚠️ **IMPORTANT**: Never commit your .env file to git!

Check `.gitignore` includes .env:
```bash
grep "\.env" /Volumes/Extreme\ SSD/kelly-project/.gitignore
```

If not present, add it:
```bash
echo ".env" >> /Volumes/Extreme\ SSD/kelly-project/.gitignore
```

## File Locations

- **Your .env file**: `/Volumes/Extreme SSD/kelly-project/.env`
- **Plugin location**: `~/.claude/plugins/rr-cli/`
- **RR CLI tool**: `/Library/Frameworks/Python.framework/Versions/3.13/bin/rr`

## Summary

1. ✅ .env file created at: `/Volumes/Extreme SSD/kelly-project/.env`
2. 📝 Edit file with your actual API key
3. 🔄 Restart Claude Code
4. 🧪 Test with `/rr-teach "test"`

---

**Once done, your RR CLI Plugin is fully functional with all AI features!** 🚀
