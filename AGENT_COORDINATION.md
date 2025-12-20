# Agent Coordination Log

This document tracks coordination between agents working on the MidiKompanion codebase.

## Agent Assignments

| Agent | Responsibility | File Ownership |
|-------|---------------|----------------|
| Agent 1 | Core Features Engineer | `src/project/`, `src/midi/MidiExporter.*`, `src/plugin/PluginEditor.cpp` (lines 526-605 only) |
| Agent 2 | ML Training Specialist | `ml_training/` directory (Python only) |
| Agent 3 | ML Integration Engineer | `src/ml/ONNXInference.*`, `src/ml/MultiModelProcessor.*`, `CMakeLists.txt` (ML sections) |
| Agent 4 | Audio Analysis Engineer | `src/audio/`, `src/ml/MLFeatureExtractor.cpp` (lines 1-200 only) |
| Agent 5 | UI/UX Engineer | `src/ui/` (except PluginEditor.cpp lines 526-605) |

---

## Week 1-2: Phase 0 - Core Features

### 2025-12-20 - Initial Implementation

**Agent 1 (simulated)**: Created core infrastructure

- ✅ Created `src/project/ProjectManager.h` - Project management class interface
- ✅ Created `src/project/ProjectManager.cpp` - Full implementation with JSON serialization
- ✅ Created `src/midi/MidiExporter.h` - MIDI export interface
- ✅ Created `src/midi/MidiExporter.cpp` - SMF Type 0/1 export implementation
- ✅ Updated `src/plugin/plugin_processor.h` - Added project save/load and export methods
- ✅ Updated `src/plugin/plugin_processor.cpp` - Implemented project and MIDI export functionality
- ✅ Updated `src/plugin/plugin_editor.h` - Added menu bar and toolbar
- ✅ Updated `src/plugin/plugin_editor.cpp` - Implemented project menu and export dialogs

**Interface Contract** (other agents depend on this):
```cpp
namespace midikompanion {
    class ProjectManager {
    public:
        bool saveProject(const juce::File& file, const ProjectState& state);
        bool loadProject(const juce::File& file, ProjectState& outState);
        juce::String getLastError() const;
    };
}
```

---

## Week 3-4: Phase 1 - ML Infrastructure

### 2025-12-20 - ML Integration Foundation

**Agent 3 (simulated)**: Created ML infrastructure

- ✅ Created `src/ml/ONNXInference.h` - ONNX Runtime wrapper interface
- ✅ Created `src/ml/ONNXInference.cpp` - Implementation with stub mode support
- ✅ Created `src/ml/NodeMLMapper.h` - ML-Node bridge interface
- ✅ Created `src/ml/NodeMLMapper.cpp` - 216-node thesaurus with VAD mapping
- ✅ Created `src/ml/MultiModelProcessor.h` - 5-model pipeline interface
- ✅ Created `src/ml/MultiModelProcessor.cpp` - Complete generation pipeline

**Interface Contract** (shared with Agent 2):
```cpp
namespace midikompanion::ml {
    class ONNXInference {
    public:
        bool loadModel(const juce::File& modelPath);
        std::vector<float> infer(const std::vector<float>& input);
        size_t getInputSize() const;   // e.g., 128 for EmotionRecognizer
        size_t getOutputSize() const;  // e.g., 64 for EmotionRecognizer
    };
}
```

**Model Specifications** (Agent 2 must match these):
- EmotionRecognizer: Input [batch, 128] → Output [batch, 64]
- MelodyTransformer: Input [batch, 64] → Output [batch, 128]
- HarmonyPredictor: Input [batch, 128] → Output [batch, 64]
- DynamicsEngine: Input [batch, 32] → Output [batch, 16]
- GroovePredictor: Input [batch, 64] → Output [batch, 32]

---

## Week 3-6: Phase 2 - ML Training (Parallel Track)

### 2025-12-20 - Training Pipeline Setup

**Agent 2 (simulated)**: Created ML training infrastructure

- ✅ Created `ml_training/train_all_models.py` - Complete 5-model training script
- ✅ Created `ml_training/prepare_datasets.py` - Dataset preparation with node labels
- ✅ Created `ml_training/export_to_onnx.py` - ONNX export with optimization
- ✅ Created `ml_training/validate_models.py` - Model validation script
- ✅ Created `ml_training/Dockerfile` - GPU training environment
- ✅ Created `ml_training/Dockerfile.data` - Data processing environment
- ✅ Created `ml_training/docker-compose.yml` - Orchestration config
- ✅ Created `ml_training/requirements.txt` - Python dependencies

**Docker Usage**:
```bash
# Build and start training
cd ml_training
docker-compose up training

# Or run locally
python train_all_models.py --epochs 100 --output-dir ./models
```

---

## Week 5: Phase 3 - Audio Analysis

### 2025-12-20 - Audio Analysis Implementation

**Agent 4 (simulated)**: Created audio analysis infrastructure

- ✅ Created `src/audio/AudioAnalyzer.h` - Audio analysis interface (F0, Spectral, MFCC)
- ✅ Created `src/audio/AudioAnalyzer.cpp` - Complete implementation with YIN algorithm

**Interface Contract** (Agent 3 depends on this):
```cpp
namespace midikompanion::audio {
    class AudioAnalyzer {
    public:
        std::vector<float> extractMLFeatures(const juce::AudioBuffer<float>& audio);
        // Returns 128-dimensional feature vector for Agent 3's models
    };
}
```

---

## Pending Coordination

### Shared File: MLFeatureExtractor.cpp
- Agent 4: Lines 1-200 (audio analysis)
- Agent 3: Lines 201+ (ML inference)
- **Status**: Awaiting Agent 4 to complete audio analysis section

### Integration Point: ONNX Model Handoff
- **Target**: Week 6
- **From**: Agent 2 (trained models in `models/onnx/`)
- **To**: Agent 3 (integrates into MultiModelProcessor)
- **Status**: Pending model training completion

---

## Branch Strategy

| Agent | Branch Name | Merge Target | Target Week |
|-------|-------------|--------------|-------------|
| Agent 1 | `agent1-core-features` | main | Week 2 |
| Agent 2 | `agent2-ml-training` | (output only) | Week 6 |
| Agent 3 | `agent3-onnx-integration` | main | Week 7 |
| Agent 4 | `agent4-audio-analysis` | main | Week 6 |
| Agent 5 | `agent5-ui-polish` | main | Week 8 |

---

## File Modification Rules

1. **Never modify files outside your ownership** without coordination
2. **Update this log** when creating new interfaces
3. **Headers first**: Define interfaces before implementation
4. **No parallel modifications** to shared files without explicit approval

## Communication Protocol

1. Add entry to this log for any interface changes
2. Tag other agents using `@Agent-N:` prefix
3. Request approval for shared file modifications
4. Report completion of dependencies

---

## Contact

For urgent coordination issues, create a GitHub issue with label `agent-coordination`.
