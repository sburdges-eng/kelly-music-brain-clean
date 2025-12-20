# Quick Start Guide: K2 Research Agent

## 🚀 Start Your First Research Session (5 minutes)

### Step 1: Choose a Research Domain

Pick one of these ready-to-use prompts:
- **Audio Emotion Detection** → `prompts/01-audio-emotion-detection.md`
- **JUCE Plugin Architecture** → `prompts/02-juce-plugin-architecture.md`
- **DAW Integration** → `prompts/03-daw-integration-patterns.md`

### Step 2: Open Cursor Composer

1. Press **Cmd+I** (Mac) or **Ctrl+I** (Windows/Linux)
2. This opens the Composer panel

### Step 3: Paste Your Research Prompt

1. Open the prompt file you chose (e.g., `research/prompts/01-audio-emotion-detection.md`)
2. Copy the entire contents
3. Paste into Cursor Composer
4. Press Enter

### Step 4: Review K2's Recommendations

K2 will provide:
- ✅ Top 5-10 relevant GitHub repositories
- ✅ Code patterns and examples
- ✅ Comparison with your current implementation
- ✅ Specific improvement recommendations

### Step 5: Document Findings

1. Save K2's response: `research/findings/[domain]-[date].md`
2. Update `research-index.md` with new findings
3. Extract reusable patterns to `research/patterns/`

---

## 📋 Generate Custom Research Prompts

```bash
cd research
python k2_research_automation.py
```

This generates prompts for all domains. Edit the script to customize.

---

## 🎯 Research Domains Available

| Domain | Prompt File | Focus Area |
|--------|------------|------------|
| Audio Analysis | `01-audio-emotion-detection.md` | Feature extraction, emotion detection |
| JUCE Plugins | `02-juce-plugin-architecture.md` | VST3/AU architecture, DSP patterns |
| DAW Integration | `03-daw-integration-patterns.md` | OSC/MIDI, Logic Pro, Ableton |
| Music Theory | Generate with script | Chord detection, harmony analysis |
| Emotion Mapping | Generate with script | Audio → emotion classification |
| Python ↔ C++ | Generate with script | Real-time audio bridges |
| Sample Management | Generate with script | Cataloging, tagging, search |

---

## 💡 Pro Tips

1. **Be Specific**: Include your current code context using `@filename` references
2. **Ask Follow-ups**: After initial research, ask K2 to show integration examples
3. **Compare Approaches**: Ask K2 to compare 2-3 different solutions
4. **Extract Patterns**: Ask K2 to show how to adapt patterns to your codebase

---

## 📝 Example Follow-up Prompt

After initial research, use this to get integration help:

```
Based on your recommendations, show me how to integrate [chosen library] into my existing code:

CURRENT CODE:
@music_brain/emotional_mapping.py

SHOW ME: 
1. Installation steps
2. Code changes needed (with diffs)
3. Migration path (can I keep my current API?)
4. Performance implications
```

---

## 🔄 Weekly Research Schedule

**Monday**: Audio Analysis & Feature Extraction  
**Tuesday**: Music Theory & Harmony  
**Wednesday**: DAW Integration  
**Thursday**: JUCE/C++ Patterns  
**Friday**: Review & Integration Planning  

---

## 📚 Next Steps

1. ✅ Infrastructure is ready
2. 🎯 Start your first research session now
3. 📊 Document findings as you go
4. 🔧 Implement improvements iteratively

**Ready to start?** Open `research/prompts/01-audio-emotion-detection.md` and paste it into Cursor Composer!
