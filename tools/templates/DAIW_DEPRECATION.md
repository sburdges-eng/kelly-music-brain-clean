# ⚠️ DEPRECATED: DAiW-Music-Brain

**This repository has been merged into [Kelly Music Brain 2.0](https://github.com/sburdges-eng/kelly-music-brain-clean).**

**All future development happens in Kelly 2.0.**

---

## What Changed?

DAiW-Music-Brain has evolved into **Kelly Music Brain 2.0**, a complete therapeutic iDAW platform with:

### 🎵 Enhanced Core Features
- **216-node emotional taxonomy** (evolved from 7-node system)
- **Real-time audio processing** with C++ DSP engine
- **VST3/CLAP plugin support** for use in any DAW
- **Desktop application** with Qt6 GUI
- **Mobile companion apps** (iOS/Android)
- **Web interface** for remote collaboration

### 🔧 What You Get
All DAiW 1.0 functionality **plus**:
- Faster, more accurate emotion → music mapping
- Real-time audio rendering and playback
- Professional plugin integration
- Multi-platform support (macOS, Windows, Linux, iOS, Android)
- Enhanced CLI with backwards compatibility

---

## Migration Guide

### Quick Migration (5 minutes)

```bash
# Uninstall DAiW 1.0
pip uninstall daiw-music-brain

# Install Kelly 2.0
pip install kelly-music-brain

# Your existing data files work as-is!
# CLI commands work with both 'kelly' and 'daiw' (legacy alias)
```

### Package Imports

```python
# Old (DAiW 1.0)
from music_brain.session.intent_schema import CompleteSongIntent
from music_brain.groove.applicator import GrooveApplicator

# New (Kelly 2.0)
from kelly_core.session.intent_schema import CompleteSongIntent
from kelly_core.groove.applicator import GrooveApplicator
```

### CLI Commands

```bash
# Both work! 'daiw' is an alias for 'kelly'
kelly intent new --title "My Song"
daiw intent new --title "My Song"  # Legacy alias for backwards compatibility
```

### Data Files

Your existing intent JSON/YAML files are **100% compatible**. No changes needed.

---

## Full Migration Documentation

For detailed migration instructions, see:
- **[Migration Guide](https://github.com/sburdges-eng/kelly-music-brain-clean/blob/main/docs/MIGRATION.md)**
- **[API Changes](https://github.com/sburdges-eng/kelly-music-brain-clean/blob/main/docs/api/changes.md)**
- **[New Features Guide](https://github.com/sburdges-eng/kelly-music-brain-clean/blob/main/docs/guides/new-features.md)**

---

## Legacy Support

### Maintenance Branch
Critical bug fixes **only** will be provided on the `1.x-maintenance` branch.

### Timeline
- **Now - Dec 2025:** DAiW 1.x maintenance branch supported
- **Jan 2026 onwards:** Kelly 2.0 only

### Security Updates
Security patches will be backported to `1.x-maintenance` until Dec 2025.

---

## Why the Change?

DAiW was always about **emotional authenticity in music creation**. Kelly 2.0 takes that philosophy to the next level:

1. **More nuanced emotions:** 216 nodes vs. 7 = capture exactly what you feel
2. **Real-time feedback:** Hear your emotions as you work, not after export
3. **Professional integration:** Use Kelly as a plugin in Logic, Ableton, FL Studio, etc.
4. **Cross-platform:** Create on desktop, refine on mobile, collaborate on web

The name "Kelly" honors the vision of therapeutic music creation while signaling a new era of capability.

---

## Questions or Issues?

- **Migration help:** Open an issue in [Kelly Music Brain](https://github.com/sburdges-eng/kelly-music-brain-clean/issues)
- **Bug reports (1.x):** Open an issue here with `[1.x-maintenance]` tag
- **General discussion:** [Kelly Discussions](https://github.com/sburdges-eng/kelly-music-brain-clean/discussions)

---

## Thank You

To everyone who contributed to DAiW 1.0 - your work lives on in Kelly 2.0. This isn't an ending, it's an evolution. 🎹

**The philosophy stays the same:**
> *"Interrogate Before Generate"*
> 
> *"The tool shouldn't finish art for people. It should make them braver."*

We're just giving you better tools to express that bravery.

---

**Ready to upgrade?** → [Get started with Kelly 2.0](https://github.com/sburdges-eng/kelly-music-brain-clean)
