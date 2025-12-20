#include "ml/MLBridge.h"
// MLBridge.h includes KellyTypes.h, so we have KellyTypes types available
// Create aliases before including EmotionThesaurus.h (which includes Types.h
// via IntentPipeline.h)
namespace kelly {
// Alias KellyTypes versions before Types.h redefines them
using KellyTypesEmotionNode = EmotionNode;
using KellyTypesWound = Wound;
using KellyTypesIntentResult = IntentResult;
} // namespace kelly

// Now include IntentPipeline.h and EmotionThesaurus.h - this brings in Types.h
// Note: We need Types.h types for EmotionThesaurus, so we include it after
// creating aliases
#include "common/Types.h" // Explicit include for Types.h types
#include "engine/EmotionThesaurus.h"
#include "engine/IntentPipeline.h" // Full definition needed
#include <algorithm>
#include <cmath>
#include <numeric>

namespace kelly {

// Conversion helper: Convert Types.h EmotionNode to KellyTypes.h EmotionNode
namespace {
KellyTypesEmotionNode convertToUnifiedEmotionNode(const EmotionNode &legacy) {
  KellyTypesEmotionNode unified;
  unified.id = legacy.id;
  unified.name = legacy.name;
  unified.categoryEnum = legacy.categoryEnum; // Use categoryEnum, not category
  // Convert enum to string
  const char *catNames[] = {"Joy",      "Sadness", "Anger", "Fear",
                            "Surprise", "Disgust", "Trust", "Anticipation"};
  int catIdx = static_cast<int>(legacy.categoryEnum);
  if (catIdx >= 0 && catIdx < 8) {
    unified.category = catNames[catIdx];
  } else {
    unified.category = "Joy";
  }
  unified.intensity = legacy.intensity;
  unified.valence = legacy.valence;
  unified.arousal = legacy.arousal;
  unified.dominance = legacy.dominance;
  unified.relatedEmotions = legacy.relatedEmotions;
  // Set defaults for unified-only fields
  unified.synonyms.clear();
  unified.layerIndex = 0;
  unified.subIndex = 0;
  unified.subSubIndex = 0;
  // Initialize musical attributes
  unified.musicalAttributes.tempoModifier = 0.8f + legacy.arousal * 0.4f;
  unified.musicalAttributes.mode = (legacy.valence < 0) ? "minor" : "major";
  unified.musicalAttributes.dynamics = 0.3f + legacy.arousal * 0.5f;
  unified.musicalAttributes.articulation = 0.5f - legacy.arousal * 0.3f;
  unified.musicalAttributes.density = 0.5f;
  unified.musicalAttributes.dissonance = std::max(0.0f, -legacy.valence * 0.5f);
  unified.musicalAttributes.suggestedRuleBreaks.clear();
  return unified;
}
} // namespace

// =============================================================================
// EmotionEmbeddingMapper Implementation
// =============================================================================

EmotionEmbeddingMapper::EmotionEmbeddingMapper() : thesaurus_(nullptr) {
  initializeDefaultWeights();
}

EmotionEmbeddingMapper::EmotionEmbeddingMapper(EmotionThesaurus *thesaurus)
    : thesaurus_(thesaurus) {
  initializeDefaultWeights();
}

void EmotionEmbeddingMapper::initializeDefaultWeights() {
  // Initialize category detection weights
  // Each category has a 64-dim weight vector
  // These would ideally be learned from data

  // Joy: High valence (dims 0-15), moderate arousal (dims 16-31)
  for (int i = 0; i < 16; ++i) {
    categoryWeights_[0][i] = 0.8f;      // Valence
    categoryWeights_[0][i + 16] = 0.4f; // Arousal
  }

  // Sadness: Low valence, low arousal
  for (int i = 0; i < 16; ++i) {
    categoryWeights_[1][i] = -0.7f;
    categoryWeights_[1][i + 16] = -0.3f;
  }

  // Anger: Low valence, high arousal
  for (int i = 0; i < 16; ++i) {
    categoryWeights_[2][i] = -0.5f;
    categoryWeights_[2][i + 16] = 0.8f;
  }

  // Fear: Low valence, high arousal, low dominance
  for (int i = 0; i < 16; ++i) {
    categoryWeights_[3][i] = -0.6f;
    categoryWeights_[3][i + 16] = 0.7f;
    categoryWeights_[3][i + 32] = -0.5f;
  }

  // Surprise: Neutral valence, high arousal
  for (int i = 0; i < 16; ++i) {
    categoryWeights_[4][i] = 0.1f;
    categoryWeights_[4][i + 16] = 0.8f;
  }

  // Disgust: Low valence, moderate arousal
  for (int i = 0; i < 16; ++i) {
    categoryWeights_[5][i] = -0.6f;
    categoryWeights_[5][i + 16] = 0.4f;
  }

  // Trust: High valence, low arousal
  for (int i = 0; i < 16; ++i) {
    categoryWeights_[6][i] = 0.6f;
    categoryWeights_[6][i + 16] = -0.2f;
  }

  // Anticipation: Moderate valence, moderate arousal
  for (int i = 0; i < 16; ++i) {
    categoryWeights_[7][i] = 0.3f;
    categoryWeights_[7][i + 16] = 0.5f;
  }
}

EmotionEmbeddingMapper::EmbeddingInterpretation
EmotionEmbeddingMapper::interpret(const std::array<float, 64> &embedding) {
  EmbeddingInterpretation result;

  // Extract dimensional values
  // Dims 0-15: Valence
  float valenceSum = 0.0f;
  for (int i = 0; i < 16; ++i)
    valenceSum += embedding[i];
  result.valence = std::tanh(valenceSum / 8.0f);

  // Dims 16-31: Arousal
  float arousalSum = 0.0f;
  for (int i = 16; i < 32; ++i)
    arousalSum += embedding[i];
  result.arousal = (std::tanh(arousalSum / 8.0f) + 1.0f) / 2.0f;

  // Dims 32-47: Dominance
  float dominanceSum = 0.0f;
  for (int i = 32; i < 48; ++i)
    dominanceSum += embedding[i];
  result.dominance = (std::tanh(dominanceSum / 8.0f) + 1.0f) / 2.0f;

  // Dims 48-63: Complexity/Specificity
  float complexitySum = 0.0f;
  for (int i = 48; i < 64; ++i)
    complexitySum += std::abs(embedding[i]);
  result.complexity = complexitySum / 16.0f;

  // Determine primary category by dot product with category weights
  float maxScore = -1e9f;
  result.primaryCategory = EmotionCategory::Joy;

  for (int cat = 0; cat < 8; ++cat) {
    float score = 0.0f;
    for (int i = 0; i < 64; ++i) {
      score += embedding[i] * categoryWeights_[cat][i];
    }
    if (score > maxScore) {
      maxScore = score;
      result.primaryCategory = static_cast<EmotionCategory>(cat);
    }
  }

  // Confidence based on how strong the max score is
  result.confidence = std::min(1.0f, std::max(0.0f, maxScore / 10.0f + 0.5f));

  return result;
}

KellyTypesEmotionNode EmotionEmbeddingMapper::embeddingToEmotion(
    const std::array<float, 64> &embedding) {
  auto interp = interpret(embedding);

  KellyTypesEmotionNode node;
  node.valence = interp.valence;
  node.arousal = interp.arousal;
  node.dominance = interp.dominance;
  node.categoryEnum = interp.primaryCategory;
  node.category = categoryToString(interp.primaryCategory);
  node.intensity = interp.complexity;

  // Try to find matching node in thesaurus
  // Note: thesaurus_->findByCategory() returns Types.h EmotionNode
  if (thesaurus_) {
    auto candidates = thesaurus_->findByCategory(interp.primaryCategory);

    float minDist = 1e9f;
    EmotionNode bestMatchLegacy; // Types.h EmotionNode
    bool found = false;

    // Compare with legacy types (Types.h EmotionNode)
    for (const auto &candidate : candidates) {
      float dv = candidate.valence - node.valence;
      float da = candidate.arousal - node.arousal;
      float dd = candidate.dominance - node.dominance;
      float dist = std::sqrt(dv * dv + da * da + dd * dd);

      if (dist < minDist) {
        minDist = dist;
        bestMatchLegacy = candidate; // Store Types.h version
        found = true;
      }
    }

    if (found && minDist < 0.5f) {
      // Convert Types.h EmotionNode to KellyTypes.h EmotionNode
      node = convertToUnifiedEmotionNode(bestMatchLegacy);
      node.intensity = interp.complexity;
    } else {
      // Use category name as fallback
      node.name = "detected_emotion";
    }
  } else {
    node.name = "detected_emotion";
  }

  return node;
}

std::array<float, 64>
EmotionEmbeddingMapper::emotionToEmbedding(const KellyTypesEmotionNode &node) {
  std::array<float, 64> embedding{};

  // Encode valence in dims 0-15
  for (int i = 0; i < 16; ++i) {
    embedding[i] = node.valence * (1.0f - static_cast<float>(i) / 32.0f);
  }

  // Encode arousal in dims 16-31
  for (int i = 16; i < 32; ++i) {
    embedding[i] = (node.arousal * 2.0f - 1.0f) *
                   (1.0f - static_cast<float>(i - 16) / 32.0f);
  }

  // Encode dominance in dims 32-47
  for (int i = 32; i < 48; ++i) {
    embedding[i] = (node.dominance * 2.0f - 1.0f) *
                   (1.0f - static_cast<float>(i - 32) / 32.0f);
  }

  // Encode intensity/complexity in dims 48-63
  int catIdx = static_cast<int>(node.categoryEnum);
  if (catIdx >= 0 && catIdx < 8) {
    for (int i = 48; i < 64; ++i) {
      embedding[i] = node.intensity * categoryWeights_[catIdx][i];
    }
  }

  return embedding;
}

std::vector<KellyTypesEmotionNode> EmotionEmbeddingMapper::embeddingsToEmotions(
    const std::vector<std::array<float, 64>> &embeddings) {
  std::vector<KellyTypesEmotionNode> result;
  result.reserve(embeddings.size());

  for (const auto &emb : embeddings) {
    result.push_back(embeddingToEmotion(emb));
  }

  return result;
}

// =============================================================================
// MLIntentPipeline Implementation
// =============================================================================

MLIntentPipeline::MLIntentPipeline() {
  brain_ = std::make_unique<KellyBrain>();
  mlProcessor_ = std::make_unique<Kelly::ML::MultiModelProcessor>();
  featureExtractor_ = std::make_unique<MLFeatureExtractor>();
  mapper_ = std::make_unique<EmotionEmbeddingMapper>(&brain_->thesaurus());
}

bool MLIntentPipeline::initialize(const std::string &modelsPath,
                                  const std::string &dataPath) {
  // Initialize KellyBrain
  if (!brain_->initialize(dataPath)) {
    return false;
  }

// Initialize ML models
// Note: MultiModelProcessor uses juce::File
#ifdef JUCE_VERSION
  juce::File modelsDir(modelsPath);
  if (!mlProcessor_->initialize(modelsDir)) {
    // Continue with fallback mode
  }
#endif

  initialized_ = true;
  return true;
}

KellyTypesIntentResult MLIntentPipeline::processAudio(const float *audioData,
                                                      size_t numSamples) {
  if (!mlEnabled_ || !initialized_) {
    // Fallback to text-based processing
    return brain_->fromText("audio input");
  }

  // Extract features using existing MLFeatureExtractor
  // Note: MLFeatureExtractor expects juce::AudioBuffer, so we need to adapt
  // For now, create a simple feature extraction
  std::array<float, 128> features{};

  // Simple feature extraction (RMS, spectral centroid, etc.)
  // In a full implementation, this would use MLFeatureExtractor
  float rms = 0.0f;
  for (size_t i = 0; i < numSamples && i < 2048; ++i) {
    rms += audioData[i] * audioData[i];
  }
  rms = std::sqrt(rms / numSamples);

  // Fill features array (simplified)
  for (size_t i = 0; i < 128; ++i) {
    features[i] = rms * (1.0f + 0.1f * std::sin(static_cast<float>(i)));
  }

  // Run full ML pipeline
  auto mlResult = mlProcessor_->runFullPipeline(features);

  // Convert embedding to emotion
  KellyTypesEmotionNode emotion =
      mapper_->embeddingToEmotion(mlResult.emotionEmbedding);

  // Create wound from detected emotion
  KellyTypesWound wound;
  wound.description = "Detected from audio";
  wound.urgency = emotion.intensity;
  wound.intensity = emotion.intensity;
  wound.source = "ml_audio";
  wound.primaryEmotion = emotion;

  // Process through intent pipeline
  KellyTypesIntentResult result = brain_->fromWound(wound);

  // Override emotion in result - use sourceWound.primaryEmotion
  result.sourceWound.primaryEmotion = emotion;

  return result;
}

void MLIntentPipeline::submitAudio(const float *audioData, size_t numSamples) {
  // For async processing, we would use AsyncMLPipeline
  // For now, this is a placeholder
  (void)audioData;
  (void)numSamples;
}

bool MLIntentPipeline::hasResult() const {
  return false; // Placeholder for async implementation
}

KellyTypesIntentResult MLIntentPipeline::getResult() {
  return KellyTypesIntentResult{}; // Placeholder
}

KellyTypesIntentResult
MLIntentPipeline::processHybrid(const float *audioData, size_t numSamples,
                                const std::string &textDescription) {

  // Get audio-based result
  KellyTypesIntentResult audioResult = processAudio(audioData, numSamples);

  // Get text-based result
  KellyTypesIntentResult textResult = brain_->fromText(textDescription);

  // Blend results (favor text for explicit intent, audio for implicit emotion)
  KellyTypesIntentResult blended = textResult;

  // Use audio emotion intensity (KellyTypes::IntentResult uses
  // sourceWound.primaryEmotion)
  blended.sourceWound.primaryEmotion.intensity =
      (audioResult.sourceWound.primaryEmotion.intensity +
       textResult.sourceWound.primaryEmotion.intensity) /
      2.0f;

  // Combine rule breaks
  for (const auto &rb : audioResult.ruleBreaks) {
    bool found = false;
    for (const auto &existing : blended.ruleBreaks) {
      if (existing.type == rb.type) {
        found = true;
        break;
      }
    }
    if (!found) {
      blended.ruleBreaks.push_back(rb);
    }
  }

  return blended;
}

std::vector<MidiNote> MLIntentPipeline::generateMelodyFromProbabilities(
    const std::array<float, 128> &noteProbabilities,
    const KellyTypesIntentResult &intent, int bars) {

  std::vector<MidiNote> notes;

  int ticksPerBar = 480 * 4; // Assuming 4/4 time
  int baseNote = 60;         // C4

  // Find top notes by probability
  std::vector<std::pair<float, int>> sortedNotes;
  for (int i = 0; i < 128; ++i) {
    sortedNotes.push_back({noteProbabilities[i], i});
  }
  std::sort(sortedNotes.begin(), sortedNotes.end(), std::greater<>());

  // Generate notes based on probabilities
  for (int bar = 0; bar < bars; ++bar) {
    int notesPerBar = 4; // Quarter notes

    for (int beat = 0; beat < notesPerBar; ++beat) {
      // Select note based on probability
      int noteIdx = sortedNotes[beat % 12].second; // Use top 12 notes

      // Constrain to reasonable range
      while (noteIdx < baseNote - 12)
        noteIdx += 12;
      while (noteIdx > baseNote + 12)
        noteIdx -= 12;

      MidiNote note;
      note.pitch = noteIdx;
      note.startTick = bar * ticksPerBar + beat * 480;
      note.durationTicks = 480;
      note.velocity = static_cast<int>(intent.dynamicRange * 127);
      note.channel = 0;

      notes.push_back(note);
    }
  }

  return notes;
}

std::vector<Chord> MLIntentPipeline::generateHarmonyFromPrediction(
    const std::array<float, 64> &harmonyPrediction,
    const KellyTypesIntentResult &intent) {

  // Simplified chord generation
  std::vector<Chord> chords;

  // Generate 4 chords based on prediction
  for (int i = 0; i < 4; ++i) {
    Chord chord;
    chord.symbol = "C";
    chord.quality = "major";
    chord.root = "C";
    // KellyTypes::Chord doesn't have rootNote, startTick, durationTicks -
    // adjust structure For now, just set the pitches and symbol
    chord.pitches = {60, 64, 67}; // C major triad
    chord.intervals = {0, 4, 7};  // Root, major third, perfect fifth

    chords.push_back(chord);
  }

  return chords;
}

void MLIntentPipeline::applyDynamics(
    std::vector<MidiNote> &notes, const std::array<float, 16> &dynamicsOutput) {

  // Apply dynamics modulation
  float baseVelocityMod = dynamicsOutput[0];

  for (size_t i = 0; i < notes.size(); ++i) {
    float positionInPhrase = static_cast<float>(i % 4) / 4.0f;
    float velocityCurve =
        baseVelocityMod + std::sin(positionInPhrase * 3.14159f) * 0.3f;
    notes[i].velocity = std::clamp(
        static_cast<int>(notes[i].velocity * (0.7f + velocityCurve * 0.6f)), 1,
        127);
  }
}

void MLIntentPipeline::applyGroove(std::vector<MidiNote> &notes,
                                   const std::array<float, 32> &grooveParams) {

  // Apply groove timing and velocity
  for (auto &note : notes) {
    int sixteenthPos = (note.startTick / 120) % 16; // 120 ticks per 16th note

    // Apply timing offset
    float timingOffset = grooveParams[sixteenthPos];
    note.startTick += static_cast<int>(timingOffset * 60);

    // Apply velocity accent
    float velocityMod = grooveParams[16 + sixteenthPos];
    note.velocity = std::clamp(
        static_cast<int>(note.velocity * (0.8f + velocityMod * 0.4f)), 1, 127);
  }
}

GeneratedMidi MLIntentPipeline::generateFromAudio(const float *audioData,
                                                  size_t numSamples, int bars) {

  KellyTypesIntentResult intent = processAudio(audioData, numSamples);

  GeneratedMidi result;
  result.tempoBpm =
      intent.tempoBpm; // KellyTypes::IntentResult has tempoBpm (int)
  result.bars = bars;
  result.key = intent.key;
  result.mode = intent.mode;
  // KellyTypes::GeneratedMidi doesn't have bpm or lengthInBeats fields -
  // they're in Types::GeneratedMidi Store tempo in metadata if needed
  result.metadata["bpm"] = std::to_string(intent.tempoBpm);

  if (mlEnabled_) {
    // Get ML results
    std::array<float, 128> features{};
    // Extract features (simplified)
    float rms = 0.0f;
    for (size_t i = 0; i < numSamples && i < 2048; ++i) {
      rms += audioData[i] * audioData[i];
    }
    rms = std::sqrt(rms / numSamples);
    for (size_t i = 0; i < 128; ++i) {
      features[i] = rms;
    }

    auto mlResult = mlProcessor_->runFullPipeline(features);

    // Generate melody from ML probabilities
    result.notes = generateMelodyFromProbabilities(mlResult.melodyProbabilities,
                                                   intent, bars);

    // Generate harmony from ML prediction
    result.chords =
        generateHarmonyFromPrediction(mlResult.harmonyPrediction, intent);

    // Apply dynamics
    applyDynamics(result.notes, mlResult.dynamicsOutput);

    // Apply groove
    applyGroove(result.notes, mlResult.grooveParameters);
  } else {
    // Fallback: use KellyBrain
    result = brain_->generateMidi(intent, bars);
  }

  return result;
}

void MLIntentPipeline::setModelEnabled(Kelly::ML::ModelType type,
                                       bool enabled) {
  mlProcessor_->setModelEnabled(type, enabled);
}

} // namespace kelly
