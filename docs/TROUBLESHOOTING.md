# miDiKompanion Troubleshooting Guide

This guide helps you resolve common issues with miDiKompanion.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Plugin Not Loading](#plugin-not-loading)
- [No MIDI Output](#no-midi-output)
- [Project Save/Load Issues](#project-saveload-issues)
- [MIDI Export Problems](#midi-export-problems)
- [Performance Issues](#performance-issues)
- [Cloud Generation Issues](#cloud-generation-issues)
- [Error Messages](#error-messages)

## Installation Issues

### Plugin Not Found After Installation

**Symptoms**: DAW doesn't show the plugin in the list

**Solutions**:

1. **Verify installation location**:
   - macOS AU: `/Library/Audio/Plug-Ins/Components/`
   - macOS VST3: `/Library/Audio/Plug-Ins/VST3/`
   - Windows VST3: `C:\Program Files\Common Files\VST3\`

2. **Rescan plugins**:
   - Most DAWs have a "Rescan" or "Refresh" option in preferences
   - In Ableton: Preferences > Plugins > Rescan
   - In Logic: Restart Logic to trigger automatic scan

3. **Check file permissions**:
   - Ensure the plugin file is readable
   - On macOS: Right-click > Get Info > check permissions

4. **Verify architecture match**:
   - Use ARM version on Apple Silicon Macs
   - Use Intel version on Intel Macs
   - Use x64 version on 64-bit Windows

### Installation Permission Denied

**macOS**:
```bash
sudo cp -R "Kelly Emotion Processor.component" /Library/Audio/Plug-Ins/Components/
```

**Windows**: Run installer as Administrator

## Plugin Not Loading

### "Plugin validation failed"

**Causes**:
- Corrupted plugin file
- Missing dependencies
- Architecture mismatch

**Solutions**:
1. Re-download and reinstall the plugin
2. Verify your DAW supports the plugin format
3. Check Console.app (macOS) or Event Viewer (Windows) for details

### "Code signing error" (macOS)

**Solution**:
1. Open System Preferences > Security & Privacy
2. Click "Allow Anyway" for the blocked plugin
3. Rescan plugins in your DAW

Alternatively, use Terminal:
```bash
xattr -cr "/Library/Audio/Plug-Ins/Components/Kelly Emotion Processor.component"
```

### Plugin crashes DAW on load

**Solutions**:
1. Update your DAW to the latest version
2. Check for conflicting plugins
3. Try loading in a new, empty project
4. Report the issue with crash logs

## No MIDI Output

### No notes playing

**Check**:

1. **Track setup**:
   - Ensure the track is a MIDI track
   - Route MIDI output to an instrument
   - Check track isn't muted

2. **Plugin state**:
   - Is the plugin bypassed?
   - Check the status label (should show "Ready" or "Local processing")

3. **Generation settings**:
   - Bars > 0
   - Generation Rate is reasonable (try 8)

4. **DAW playback**:
   - Press Play in your DAW
   - Check transport is running

### MIDI only works intermittently

**Possible causes**:
- Generation Rate too high (try lowering it)
- CPU overload
- Buffer size issues

**Solutions**:
1. Lower Generation Rate to 4-8
2. Increase audio buffer size
3. Check CPU usage

## Project Save/Load Issues

### "Failed to save project"

**Common causes**:
1. **Permission denied**: No write access to folder
2. **Disk full**: Not enough space
3. **Invalid path**: Special characters in filename

**Solutions**:
- Save to Documents folder instead
- Check available disk space
- Use simple filenames (letters, numbers, underscores)

### "Failed to load project"

**Causes**:
1. **File corrupted**: Project file damaged
2. **Version mismatch**: Project from newer version
3. **File not found**: Project was moved/deleted

**Solutions**:
1. Check if file exists in the specified location
2. Try loading backup file (projectname_backup_timestamp.mkp)
3. Update plugin to latest version
4. Verify the project extension is `.mkp`

### Project loads but settings are wrong

**Cause**: Version migration issue

**Solution**:
1. Check for error messages in DAW console
2. Re-create the project with current version
3. Report issue if persists

## MIDI Export Problems

### "No MIDI data to export"

**Cause**: No MIDI has been generated yet

**Solution**:
1. Play the session to generate MIDI
2. Wait for status to show "MIDI data ready for export"
3. Then export

### Exported MIDI is empty

**Solutions**:
1. Ensure MIDI was generated before export
2. Check export options
3. Try exporting to different location
4. If vocals are missing, enable **Include Vocals** in export options

### MIDI file won't open in other software

**Possible causes**:
- File permissions
- Corrupted export
- Unsupported MIDI features

**Solutions**:
1. Re-export with "SMF Type 0" format (most compatible)
2. Try opening in different software
3. Verify file size is > 0 bytes

### Wrong tempo in exported MIDI

**Solution**:
- Ensure "Include Tempo" option is checked
- Check that DAW tempo was set before export

## Performance Issues

### High CPU usage

**Solutions**:
1. Increase audio buffer size
2. Lower Generation Rate
3. Disable cloud generation if not needed
4. Close other resource-intensive applications

### Audio dropouts/glitches

**Solutions**:
1. Increase buffer size (try 512 or 1024 samples)
2. Reduce number of plugin instances
3. Check for driver issues
4. Disable Wi-Fi/Bluetooth if not needed

### Plugin UI is slow/unresponsive

**Solutions**:
1. Check GPU drivers are up to date
2. Reduce screen resolution
3. Close other graphical applications
4. Restart DAW

## Cloud Generation Issues

### Cloud not connecting

**Check**:
1. Internet connection is working
2. Cloud endpoint is configured:
   ```bash
   echo $KELLY_AI_ENDPOINT
   ```
3. Firewall isn't blocking connection

### Cloud returns errors

**Solutions**:
1. Check API credentials
2. Verify endpoint URL is correct
3. Check server status
4. Try again later (may be temporary outage)

### Cloud is slow

**Causes**:
- Network latency
- Server load
- Large generation requests

**Solutions**:
1. Use local ONNX model for faster results
2. Reduce Bars setting
3. Try off-peak hours

## Error Messages

### "AI generation error"

**Causes**:
- ONNX model not found
- Model incompatible
- Memory issue

**Solutions**:
1. Set ONNX model path:
   ```bash
   export KELLY_ONNX_MODEL=/path/to/emotion_model.onnx
   ```
2. Verify model file exists and is valid
3. Check available memory

### "Invalid project file format"

**Cause**: File is not a valid `.mkp` file

**Solutions**:
1. Check you're opening the right file
2. Don't rename other files to `.mkp`
3. Check if file was corrupted during transfer

### "Track has invalid MIDI channel"

**Cause**: MIDI channel out of range (must be 0-15)

**Solution**: This indicates a bug; please report it

## Getting More Help

### Collecting Debug Information

When reporting issues, include:

1. **System info**:
   - OS version
   - DAW name and version
   - Plugin version

2. **Steps to reproduce**:
   - What you did
   - What you expected
   - What happened instead

3. **Log files**:
   - DAW console output
   - System logs (Console.app on macOS)

### Reporting Bugs

1. Check if issue already reported on GitHub
2. Create new issue with debug information
3. Attach project file if relevant (remove sensitive data)

### Contact

- **GitHub Issues**: Report bugs and feature requests
- **Discord**: Community support and discussion
- **Email**: support@midikompanion.dev

---

If your issue isn't listed here, please report it on [GitHub Issues](https://github.com/midikompanion/issues).
