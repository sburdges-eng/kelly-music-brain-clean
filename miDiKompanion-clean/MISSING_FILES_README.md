# Missing Files Reference

This is a **cleaned** version of the miDiKompanion project, containing only the active working files.

## If Something Seems Missing

The original project location contains the full history and archive:

**Original Location:** `/Users/seanburdges/Desktop/final kel/`

**Size:** 346GB (611,480 files)
**Clean Version:** 2.1GB (~6,000 files)

## What's Included Here

### Core Source Code
- `src/bridge/` - Python/C++ communication layer
- `src/engine/` - KellyBrain, QuantumEmotionalField, emotion mapping
- `src/engines/` - 13 specialized music engines (Bass, Melody, Rhythm, etc.)
- `src/midi/` - MIDI generation and chord systems
- `src/ml/` - ML processors (DDSP, RTNeural, ONNX inference)
- `src/plugin/` - VST plugin processor and editor
- `src/ui/` - EmotionWorkstation and all UI components
- `src/voice/` - Vocal synthesis and lyric generation
- `src/audio/`, `src/dsp/`, `src/common/` - Support systems

### ML & Training
- `ml_training/` - Training scripts and trained models (1.2GB)
  - `trained_models/` - All .pt, .onnx, .json model files
  - `trained_models/checkpoints/` - Training checkpoints
  - Config files and deployment scripts
- `ml_framework/` - ML framework code (769MB)
- `training_pipe/` - Training pipeline (14MB)

### Data Files
- `Data_Files/` - JSON databases (100KB)
  - Chord progression families
  - Genre pocket maps
  - Mix fingerprints
- `data/` - Application data (5MB)
  - Emotional mapping data
  - Scale databases
  - Vernacular databases
- `datasets/` - Training datasets (12KB)

### Build System
- `CMakeLists.txt` (root and src)
- `external/` - JUCE, oscpack, pybind11, readerwriterqueue

## What's NOT Included (Check Original Location)

### Documentation Archive (~344GB)
- `vault/` - Obsidian knowledge base
- `docs/` - Extensive documentation
- `Obsidian_Documentation/`
- `Production_Workflows/`
- `Songwriting_Guides/`
- `Theory_Reference/`

### Legacy Repositories
- `DAiW-Music-Brain/`
- `penta_core_music-brain/`
- `src_penta-core/`
- `music_brain/`
- `reference/`

### Archive Directories
- `kell ml ai/`, `kell ml ai 2/`
- `kell ml final/`
- `ml model training/`
- `training pipe/`
- `iDAW_Core/` (if not actively used)
- `miDiKompanion/` (if duplicate)

### Python Tools (if needed)
- `python/` - Full Python tooling suite
- `mcp_workstation/` - Multi-AI orchestration
- `mcp_todo/` - TODO server

### Build Artifacts
- `build/` directories
- `.vscode/`, `.idea/` IDE configs
- `node_modules/`
- Virtual environments

## Quick Recovery

If you need something that's not here:

```bash
# Check original location
cd "/Users/seanburdges/Desktop/final kel"

# Search for a file
find . -name "filename.ext"

# Search for content
grep -r "search term" .

# Copy specific directory back
cp -r "/Users/seanburdges/Desktop/final kel/path/to/dir" .
```

## Git Repository

The `.git` directory is included with full history on the `miDiKompanion` branch.

```bash
git status
git log --oneline
git branch -a
```

## Notes

- This cleaned version was created: 2025-12-18
- Original project preserved at: `/Users/seanburdges/Desktop/final kel/`
- All active development files are present
- All trained models and training data included
- Build system is complete and functional

**When in doubt, check the original location first.**
