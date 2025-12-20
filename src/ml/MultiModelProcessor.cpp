/**
 * @file MultiModelProcessor.cpp
 * @brief Implementation of the 5-model ML pipeline
 */

#include "MultiModelProcessor.h"
#include <chrono>
#include <algorithm>
#include <cmath>

namespace midikompanion::ml {

//==============================================================================
// Scale definitions
//==============================================================================

static const std::map<juce::String, std::vector<int>> SCALE_INTERVALS = {
    {"major",      {0, 2, 4, 5, 7, 9, 11}},
    {"minor",      {0, 2, 3, 5, 7, 8, 10}},
    {"dorian",     {0, 2, 3, 5, 7, 9, 10}},
    {"phrygian",   {0, 1, 3, 5, 7, 8, 10}},
    {"lydian",     {0, 2, 4, 6, 7, 9, 11}},
    {"mixolydian", {0, 2, 4, 5, 7, 9, 10}},
    {"locrian",    {0, 1, 3, 5, 6, 8, 10}},
    {"pentatonic", {0, 2, 4, 7, 9}},
    {"blues",      {0, 3, 5, 6, 7, 10}},
    {"harmonic_minor", {0, 2, 3, 5, 7, 8, 11}},
    {"melodic_minor",  {0, 2, 3, 5, 7, 9, 11}}
};

static const std::map<juce::String, std::vector<std::vector<int>>> CHORD_PROGRESSIONS = {
    {"major", {{0, 4, 7}, {5, 9, 0}, {7, 11, 2}, {0, 4, 7}}},  // I-IV-V-I
    {"minor", {{0, 3, 7}, {5, 8, 0}, {7, 10, 2}, {0, 3, 7}}}   // i-iv-v-i
};

//==============================================================================
// Constructor
//==============================================================================

MultiModelProcessor::MultiModelProcessor() {
    initialize();
}

//==============================================================================
// Initialization
//==============================================================================

bool MultiModelProcessor::initialize() {
    // Initialize node mapper (creates 216-node thesaurus)
    nodeMapper_.initialize();
    
    // Create stub models for now (will be replaced with real models when loaded)
    emotionRecognizer_ = ONNXModelFactory::createStubModel(
        ModelSpecs::EMOTION_RECOGNIZER_INPUT, 
        ModelSpecs::EMOTION_RECOGNIZER_OUTPUT);
    
    melodyTransformer_ = ONNXModelFactory::createStubModel(
        ModelSpecs::MELODY_TRANSFORMER_INPUT, 
        ModelSpecs::MELODY_TRANSFORMER_OUTPUT);
    
    harmonyPredictor_ = ONNXModelFactory::createStubModel(
        ModelSpecs::HARMONY_PREDICTOR_INPUT, 
        ModelSpecs::HARMONY_PREDICTOR_OUTPUT);
    
    dynamicsEngine_ = ONNXModelFactory::createStubModel(
        ModelSpecs::DYNAMICS_ENGINE_INPUT, 
        ModelSpecs::DYNAMICS_ENGINE_OUTPUT);
    
    groovePredictor_ = ONNXModelFactory::createStubModel(
        ModelSpecs::GROOVE_PREDICTOR_INPUT, 
        ModelSpecs::GROOVE_PREDICTOR_OUTPUT);
    
    initialized_ = true;
    return true;
}

bool MultiModelProcessor::loadModels(const juce::File& modelsDir) {
    if (!modelsDir.isDirectory()) {
        juce::Logger::writeToLog("Models directory not found: " + modelsDir.getFullPathName());
        return false;
    }
    
    bool allLoaded = true;
    
    // Try to load each model
    juce::File emotionFile = modelsDir.getChildFile("emotion_recognizer.onnx");
    if (emotionFile.existsAsFile()) {
        emotionRecognizer_ = ONNXModelFactory::createEmotionRecognizer(emotionFile);
    } else {
        allLoaded = false;
    }
    
    juce::File melodyFile = modelsDir.getChildFile("melody_transformer.onnx");
    if (melodyFile.existsAsFile()) {
        melodyTransformer_ = ONNXModelFactory::createMelodyTransformer(melodyFile);
    } else {
        allLoaded = false;
    }
    
    juce::File harmonyFile = modelsDir.getChildFile("harmony_predictor.onnx");
    if (harmonyFile.existsAsFile()) {
        harmonyPredictor_ = ONNXModelFactory::createHarmonyPredictor(harmonyFile);
    } else {
        allLoaded = false;
    }
    
    juce::File dynamicsFile = modelsDir.getChildFile("dynamics_engine.onnx");
    if (dynamicsFile.existsAsFile()) {
        dynamicsEngine_ = ONNXModelFactory::createDynamicsEngine(dynamicsFile);
    } else {
        allLoaded = false;
    }
    
    juce::File grooveFile = modelsDir.getChildFile("groove_predictor.onnx");
    if (grooveFile.existsAsFile()) {
        groovePredictor_ = ONNXModelFactory::createGroovePredictor(grooveFile);
    } else {
        allLoaded = false;
    }
    
    mlModelsLoaded_ = allLoaded;
    
    if (!allLoaded) {
        juce::Logger::writeToLog("Some models not found - using stub mode for missing models");
    }
    
    return allLoaded;
}

MultiModelProcessor::ModelStatus MultiModelProcessor::getModelStatus() const {
    ModelStatus status;
    
    if (emotionRecognizer_) {
        status.emotionRecognizer = emotionRecognizer_->isModelLoaded();
    }
    if (melodyTransformer_) {
        status.melodyTransformer = melodyTransformer_->isModelLoaded();
    }
    if (harmonyPredictor_) {
        status.harmonyPredictor = harmonyPredictor_->isModelLoaded();
    }
    if (dynamicsEngine_) {
        status.dynamicsEngine = dynamicsEngine_->isModelLoaded();
    }
    if (groovePredictor_) {
        status.groovePredictor = groovePredictor_->isModelLoaded();
    }
    
    return status;
}

//==============================================================================
// Generation - Main Interface
//==============================================================================

GenerationResult MultiModelProcessor::generateFromNode(int nodeId, const GenerationConfig& config) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    GenerationResult result;
    
    // Get node from thesaurus
    auto nodeOpt = nodeMapper_.getNode(nodeId);
    if (!nodeOpt.has_value()) {
        result.success = false;
        result.errorMessage = "Invalid node ID: " + juce::String(nodeId);
        return result;
    }
    
    EmotionNodeML node = *nodeOpt;
    result.sourceNodeId = nodeId;
    
    // Get node context for ML models
    NodeContext context = nodeMapper_.getNodeContext(nodeId);
    
    // Generate each component
    if (config.generateChords) {
        if (mlEnabled_ && config.useMLEnhancement) {
            // ML-enhanced chord generation
            auto mlChords = runHarmonyPredictor(context.contextVector, config);
            auto ruleChords = generateChordsRuleBased(node, config);
            result.chords = mlChords.empty() ? ruleChords : mlChords;
        } else {
            result.chords = generateChordsRuleBased(node, config);
        }
    }
    
    if (config.generateMelody) {
        if (mlEnabled_ && config.useMLEnhancement) {
            std::vector<float> embedding = nodeMapper_.nodeToMLInput(node);
            auto mlMelody = runMelodyTransformer(embedding, config);
            auto ruleMelody = generateMelodyRuleBased(node, config);
            
            // Blend ML and rule-based outputs
            result.melody = mlMelody.empty() ? ruleMelody : mlMelody;
        } else {
            result.melody = generateMelodyRuleBased(node, config);
        }
        applyConstraints(result.melody, config);
    }
    
    if (config.generateBass) {
        result.bassline = generateBassRuleBased(node, config, result.chords);
        applyConstraints(result.bassline, config);
    }
    
    // Generate dynamics
    if (mlEnabled_ && config.useMLEnhancement) {
        std::vector<float> intensityInput(32, node.vad.intensity);
        result.dynamics = runDynamicsEngine(intensityInput);
    }
    if (result.dynamics.empty()) {
        result.dynamics = generateDynamicsRuleBased(node, config);
    }
    
    // Apply groove
    if (mlEnabled_ && config.useMLEnhancement && groovePredictor_) {
        std::vector<float> arousalInput(64, node.vad.arousal);
        auto grooveResult = groovePredictor_->infer(arousalInput);
        if (grooveResult.success) {
            applyGroove(result, grooveResult.output);
        }
    }
    
    // Calculate processing time
    auto endTime = std::chrono::high_resolution_clock::now();
    result.processingTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    result.success = true;
    result.mlConfidence = mlEnabled_ ? config.mlBlendRatio : 0.0f;
    
    return result;
}

GenerationResult MultiModelProcessor::generateFromVAD(const VADCoordinates& vad, 
                                                      const GenerationConfig& config) {
    // Find nearest node
    auto [nearestNode, distance] = nodeMapper_.findNearestNode(vad);
    
    // Generate from that node
    GenerationResult result = generateFromNode(nearestNode.id, config);
    
    // Adjust confidence based on distance
    result.mlConfidence *= (1.0f / (1.0f + distance));
    
    return result;
}

GenerationResult MultiModelProcessor::generateFromEmbedding(const std::vector<float>& embedding,
                                                            const GenerationConfig& config) {
    // Convert embedding to node
    EmotionNodeML node = nodeMapper_.embeddingToNode(embedding);
    
    // Store embedding in node for ML processing
    node.mlEmbedding = embedding;
    
    return generateFromNode(node.id, config);
}

GenerationResult MultiModelProcessor::generateFromAudio(const std::vector<float>& audioFeatures,
                                                        const GenerationConfig& config) {
    // Run emotion recognizer
    std::vector<float> embedding = runEmotionRecognizer(audioFeatures);
    
    // Generate from embedding
    return generateFromEmbedding(embedding, config);
}

//==============================================================================
// Pipeline Stages
//==============================================================================

std::vector<float> MultiModelProcessor::runEmotionRecognizer(const std::vector<float>& audioFeatures) {
    if (!emotionRecognizer_) {
        return {};
    }
    
    auto result = emotionRecognizer_->infer(audioFeatures);
    return result.success ? result.output : std::vector<float>();
}

std::vector<GeneratedNote> MultiModelProcessor::runMelodyTransformer(
    const std::vector<float>& embedding,
    const GenerationConfig& config) {
    
    if (!melodyTransformer_) {
        return {};
    }
    
    auto result = melodyTransformer_->infer(embedding);
    if (!result.success) {
        return {};
    }
    
    // Convert probabilities to notes
    std::vector<GeneratedNote> notes;
    
    int numBeats = config.numBars * config.beatsPerBar;
    int notesPerBeat = 2; // Eighth notes
    
    for (int i = 0; i < numBeats * notesPerBeat; ++i) {
        // Use output probabilities to determine if note plays
        float noteProb = (i < static_cast<int>(result.output.size())) ? 
                         result.output[i % result.output.size()] : 0.5f;
        
        if (noteProb > 0.3f || (random_.nextFloat() < noteProb * config.temperature)) {
            GeneratedNote note;
            
            // Use output to determine pitch (map to MIDI range)
            int pitchIdx = i % std::min<int>(result.output.size(), 12);
            float pitchProb = result.output[pitchIdx];
            
            // Map to MIDI note (60-84 range by default)
            note.pitch = 60 + static_cast<int>(pitchProb * 24);
            note.pitch = std::clamp(note.pitch, config.lowestNote, config.highestNote);
            
            note.velocity = 80 + static_cast<int>(noteProb * 40);
            note.startBeat = i * 0.5; // Eighth notes
            note.durationBeats = 0.4;
            note.channel = 1;
            
            notes.push_back(note);
        }
    }
    
    return notes;
}

std::vector<GeneratedChord> MultiModelProcessor::runHarmonyPredictor(
    const std::vector<float>& context,
    const GenerationConfig& config) {
    
    if (!harmonyPredictor_) {
        return {};
    }
    
    auto result = harmonyPredictor_->infer(context);
    if (!result.success) {
        return {};
    }
    
    std::vector<GeneratedChord> chords;
    
    // Generate one chord per bar
    for (int bar = 0; bar < config.numBars; ++bar) {
        GeneratedChord chord;
        chord.startBeat = bar * config.beatsPerBar;
        chord.durationBeats = config.beatsPerBar;
        
        // Use output probabilities to determine chord
        int chordIdx = bar % std::min<int>(result.output.size() / 4, 4);
        float chordProb = result.output[chordIdx * 4];
        
        // Build chord based on probability
        int root = config.keyRoot + static_cast<int>(chordProb * 12) % 12;
        
        // Major or minor based on next probability
        float qualityProb = (chordIdx * 4 + 1 < static_cast<int>(result.output.size())) ?
                           result.output[chordIdx * 4 + 1] : 0.5f;
        
        if (qualityProb > 0.5f) {
            // Major chord
            chord.pitches = {root + 48, root + 52, root + 55}; // Root, M3, P5
            chord.chordName = juce::String::charToString(static_cast<juce::juce_wchar>('C' + root)) + " Major";
        } else {
            // Minor chord
            chord.pitches = {root + 48, root + 51, root + 55}; // Root, m3, P5
            chord.chordName = juce::String::charToString(static_cast<juce::juce_wchar>('C' + root)) + " Minor";
        }
        
        chords.push_back(chord);
    }
    
    return chords;
}

std::vector<float> MultiModelProcessor::runDynamicsEngine(const std::vector<float>& intensityFeatures) {
    if (!dynamicsEngine_) {
        return {};
    }
    
    auto result = dynamicsEngine_->infer(intensityFeatures);
    return result.success ? result.output : std::vector<float>();
}

void MultiModelProcessor::applyGroove(GenerationResult& result, const std::vector<float>& grooveParams) {
    if (grooveParams.empty()) {
        return;
    }
    
    // Extract groove parameters
    result.swing = grooveParams.size() > 0 ? grooveParams[0] : 0.0f;
    result.humanize = grooveParams.size() > 1 ? grooveParams[1] : 0.0f;
    result.grooveIntensity = grooveParams.size() > 2 ? grooveParams[2] : 0.5f;
    
    // Apply swing to melody notes
    for (auto& note : result.melody) {
        // Swing: delay off-beat notes
        double beatPos = std::fmod(note.startBeat, 1.0);
        if (beatPos > 0.4 && beatPos < 0.6) {
            note.timingOffset = result.swing * 0.1; // Up to 10% of beat
        }
        
        // Humanize: random timing variation
        note.timingOffset += (random_.nextFloat() - 0.5f) * result.humanize * 0.05;
        
        // Velocity variation
        note.velocityVariation = (random_.nextFloat() - 0.5f) * result.humanize * 20.0f;
    }
    
    // Apply to bass
    for (auto& note : result.bassline) {
        note.timingOffset += (random_.nextFloat() - 0.5f) * result.humanize * 0.03;
    }
}

//==============================================================================
// Rule-Based Fallbacks
//==============================================================================

std::vector<GeneratedNote> MultiModelProcessor::generateMelodyRuleBased(
    const EmotionNodeML& node,
    const GenerationConfig& config) {
    
    std::vector<GeneratedNote> notes;
    
    // Get scale notes
    std::vector<int> scaleNotes = getScaleNotes(config.keyRoot, node.musical.mode);
    
    int numBeats = config.numBars * config.beatsPerBar;
    double noteDensity = node.musical.rhythmicDensity;
    
    // Generate notes based on emotion
    int currentPitch = 60 + config.keyRoot;
    
    for (int beat = 0; beat < numBeats; ++beat) {
        // Determine if we play a note this beat
        float playChance = 0.3f + noteDensity * 0.5f;
        
        if (random_.nextFloat() < playChance) {
            GeneratedNote note;
            note.startBeat = beat;
            
            // Duration based on density
            if (random_.nextFloat() < noteDensity) {
                note.durationBeats = 0.5; // Eighth note
            } else {
                note.durationBeats = 1.0; // Quarter note
            }
            
            // Pitch movement based on melodic range
            int movement = static_cast<int>((random_.nextFloat() - 0.5f) * 
                                           node.musical.melodicRange * 12);
            currentPitch += movement;
            currentPitch = std::clamp(currentPitch, config.lowestNote, config.highestNote);
            
            // Quantize to scale
            note.pitch = quantizeToScale(currentPitch, scaleNotes);
            
            // Velocity based on dynamics
            int baseVelocity = 60 + static_cast<int>(node.musical.dynamicRange * 60);
            note.velocity = std::clamp(baseVelocity + random_.nextInt(20) - 10, 1, 127);
            
            note.channel = 1;
            notes.push_back(note);
        }
    }
    
    return notes;
}

std::vector<GeneratedNote> MultiModelProcessor::generateBassRuleBased(
    const EmotionNodeML& node,
    const GenerationConfig& config,
    const std::vector<GeneratedChord>& chords) {
    
    std::vector<GeneratedNote> bassNotes;
    
    // Generate bass following chord roots
    for (const auto& chord : chords) {
        if (chord.pitches.empty()) continue;
        
        GeneratedNote note;
        note.pitch = chord.pitches[0] - 12; // Bass is one octave below chord
        note.pitch = std::clamp(note.pitch, 28, 55); // Bass range
        note.startBeat = chord.startBeat;
        note.durationBeats = chord.durationBeats * 0.9;
        note.velocity = 90 + static_cast<int>(node.musical.dynamicRange * 30);
        note.channel = 2;
        
        bassNotes.push_back(note);
        
        // Add passing note if density is high
        if (node.musical.rhythmicDensity > 0.5f && chord.durationBeats >= 2.0) {
            GeneratedNote passing;
            passing.pitch = note.pitch + 5; // Fifth above
            passing.startBeat = chord.startBeat + chord.durationBeats / 2;
            passing.durationBeats = chord.durationBeats / 2 * 0.9;
            passing.velocity = note.velocity - 10;
            passing.channel = 2;
            bassNotes.push_back(passing);
        }
    }
    
    return bassNotes;
}

std::vector<GeneratedChord> MultiModelProcessor::generateChordsRuleBased(
    const EmotionNodeML& node,
    const GenerationConfig& config) {
    
    std::vector<GeneratedChord> chords;
    
    // Get appropriate chord progression based on mode
    bool isMajor = (node.musical.mode == "major" || 
                    node.musical.mode == "lydian" || 
                    node.musical.mode == "mixolydian");
    
    // Common progressions
    std::vector<std::vector<int>> progression;
    if (isMajor) {
        if (node.vad.valence > 0.3f) {
            progression = {{0, 4, 7}, {5, 9, 0}, {7, 11, 2}, {0, 4, 7}}; // I-IV-V-I
        } else {
            progression = {{0, 4, 7}, {9, 0, 4}, {5, 9, 0}, {0, 4, 7}}; // I-vi-IV-I
        }
    } else {
        progression = {{0, 3, 7}, {5, 8, 0}, {7, 10, 2}, {0, 3, 7}}; // i-iv-v-i (minor)
    }
    
    // Generate chords for each bar
    for (int bar = 0; bar < config.numBars; ++bar) {
        GeneratedChord chord;
        chord.startBeat = bar * config.beatsPerBar;
        chord.durationBeats = config.beatsPerBar;
        
        // Get chord from progression (cycle through)
        auto& intervals = progression[bar % progression.size()];
        
        int root = config.keyRoot + 48; // Middle C octave
        for (int interval : intervals) {
            chord.pitches.push_back(root + interval);
        }
        
        // Add chord name
        chord.chordName = juce::String::charToString(static_cast<juce::juce_wchar>('C' + config.keyRoot));
        chord.chordName += isMajor ? " Major" : " Minor";
        
        // Voicing spread based on harmonic complexity
        chord.voicingSpread = node.musical.harmonicComplexity;
        
        chords.push_back(chord);
    }
    
    return chords;
}

std::vector<float> MultiModelProcessor::generateDynamicsRuleBased(
    const EmotionNodeML& node,
    const GenerationConfig& config) {
    
    std::vector<float> dynamics;
    int numBeats = config.numBars * config.beatsPerBar;
    dynamics.reserve(numBeats);
    
    float baseLevel = node.musical.dynamicRange;
    
    for (int beat = 0; beat < numBeats; ++beat) {
        // Create dynamic curve based on position in phrase
        float phrasePos = static_cast<float>(beat % config.beatsPerBar) / config.beatsPerBar;
        float barPos = static_cast<float>(beat / config.beatsPerBar) / config.numBars;
        
        // Crescendo within bar, with overall arc
        float dynamic = baseLevel;
        dynamic += phrasePos * 0.1f; // Slight crescendo within beat
        dynamic += std::sin(barPos * 3.14159f) * 0.2f; // Arc over piece
        
        dynamic = std::clamp(dynamic, 0.1f, 1.0f);
        dynamics.push_back(dynamic);
    }
    
    return dynamics;
}

//==============================================================================
// Utility
//==============================================================================

void MultiModelProcessor::applyConstraints(std::vector<GeneratedNote>& notes, 
                                           const GenerationConfig& config) {
    std::vector<int> scaleNotes = getScaleNotes(config.keyRoot, config.scale);
    
    for (auto& note : notes) {
        // Clamp to range
        note.pitch = std::clamp(note.pitch, config.lowestNote, config.highestNote);
        
        // Quantize to scale
        note.pitch = quantizeToScale(note.pitch, scaleNotes);
        
        // Clamp velocity
        note.velocity = std::clamp(note.velocity, 1, 127);
    }
}

std::vector<int> MultiModelProcessor::getScaleNotes(int keyRoot, const juce::String& scale) {
    std::vector<int> notes;
    
    auto it = SCALE_INTERVALS.find(scale);
    if (it == SCALE_INTERVALS.end()) {
        // Default to major scale
        it = SCALE_INTERVALS.find("major");
    }
    
    const auto& intervals = it->second;
    
    // Generate all scale notes in MIDI range
    for (int octave = 0; octave < 11; ++octave) {
        for (int interval : intervals) {
            int note = keyRoot + (octave * 12) + interval;
            if (note >= 0 && note <= 127) {
                notes.push_back(note);
            }
        }
    }
    
    return notes;
}

int MultiModelProcessor::quantizeToScale(int pitch, const std::vector<int>& scaleNotes) {
    if (scaleNotes.empty()) {
        return pitch;
    }
    
    // Find nearest scale note
    int nearest = scaleNotes[0];
    int minDist = std::abs(pitch - nearest);
    
    for (int scaleNote : scaleNotes) {
        int dist = std::abs(pitch - scaleNote);
        if (dist < minDist) {
            minDist = dist;
            nearest = scaleNote;
        }
    }
    
    return nearest;
}

} // namespace midikompanion::ml
