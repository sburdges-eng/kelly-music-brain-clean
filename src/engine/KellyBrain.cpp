#include "engine/KellyBrain.h"
// KellyBrain.h includes KellyTypes.h, so Wound, EmotionNode, etc. are
// KellyTypes versions Now we create aliases for the KellyTypes versions before
// Types.h redefines them
namespace kelly {
// Alias KellyTypes versions before Types.h redefines them
using KellyTypesWound = Wound;
using KellyTypesEmotionNode = EmotionNode;
using KellyTypesIntentResult = IntentResult;
using KellyTypesRuleBreak = RuleBreak;
using KellyTypesRuleBreakType = RuleBreakType;
} // namespace kelly

// Now include IntentPipeline.h - this brings in Types.h which redefines the
// types. Must include before using IntentPipeline as complete type.
#include "common/Types.h" // Explicit include - this redefines Wound, EmotionNode, etc.
#include "engine/IntentPipeline.h" // Full definition needed for std::unique_ptr<IntentPipeline>
#include <algorithm>
#include <cctype>

namespace kelly {

// Helper function to convert EmotionCategory enum to string
static std::string categoryEnumToString(EmotionCategory cat) {
  const char *catNames[] = {"Joy",      "Sadness", "Anger", "Fear",
                            "Surprise", "Disgust", "Trust", "Anticipation"};
  int catIdx = static_cast<int>(cat);
  if (catIdx >= 0 && catIdx < 8) {
    return std::string(catNames[catIdx]);
  }
  return "Joy";
}

// Conversion helpers between KellyTypes.h and Types.h structures
// These work by manually copying fields between compatible structures
namespace {
// Convert KellyTypes::Wound to Types::Wound
Wound convertToLegacyWound(const KellyTypesWound &unified) {
  Wound legacy; // This is Types::Wound now
  legacy.description = unified.description;
  legacy.intensity = unified.intensity; // Use compatibility field
  legacy.source = unified.source;       // Use compatibility field
  return legacy;
}

// Convert Types::IntentResult to KellyTypes::IntentResult
KellyTypesIntentResult
convertFromLegacyIntentResult(const IntentResult &legacy) {
  KellyTypesIntentResult unified; // This is KellyTypes::IntentResult

  // Map wound to sourceWound
  unified.sourceWound.description = legacy.sourceWound.description;
  unified.sourceWound.intensity = legacy.sourceWound.intensity;
  unified.sourceWound.urgency =
      legacy.sourceWound.intensity; // urgency = intensity
  unified.sourceWound.source = legacy.sourceWound.source;
  unified.sourceWound.desire = legacy.sourceWound.source;

  // Map emotion to sourceWound.primaryEmotion and also set emotion
  // compatibility field
  unified.sourceWound.primaryEmotion.id = legacy.emotion.id;
  unified.sourceWound.primaryEmotion.name = legacy.emotion.name;
  unified.sourceWound.primaryEmotion.categoryEnum = legacy.emotion.categoryEnum;
  unified.sourceWound.primaryEmotion.category =
      categoryEnumToString(legacy.emotion.categoryEnum);
  unified.sourceWound.primaryEmotion.valence = legacy.emotion.valence;
  unified.sourceWound.primaryEmotion.arousal = legacy.emotion.arousal;
  unified.sourceWound.primaryEmotion.dominance = legacy.emotion.dominance;
  unified.sourceWound.primaryEmotion.intensity = legacy.emotion.intensity;

  // Also set emotion compatibility field (should match primaryEmotion)
  unified.emotion.id = legacy.emotion.id;
  unified.emotion.name = legacy.emotion.name;
  unified.emotion.categoryEnum = legacy.emotion.categoryEnum;
  unified.emotion.category = categoryEnumToString(legacy.emotion.categoryEnum);
  unified.emotion.valence = legacy.emotion.valence;
  unified.emotion.arousal = legacy.emotion.arousal;
  unified.emotion.dominance = legacy.emotion.dominance;
  unified.emotion.intensity = legacy.emotion.intensity;

  // Set tempo from tempoBpm (convert BPM to modifier)
  unified.tempo = static_cast<float>(unified.tempoBpm) / 120.0f;

  // Map musical parameters
  unified.mode = legacy.mode;
  unified.tempoBpm =
      static_cast<int>(120 * legacy.tempo); // tempo is a multiplier
  unified.syncopationLevel = legacy.syncopationLevel;
  unified.humanization = legacy.humanization;
  unified.dynamicRange = legacy.dynamicRange;
  unified.allowChromaticism = legacy.allowDissonance;

  // Convert rule breaks
  unified.ruleBreaks.clear();
  for (const auto &rb : legacy.ruleBreaks) {
    KellyTypesRuleBreak unifiedRb; // KellyTypes::RuleBreak
    // Map RuleBreakType enum values
    // legacy.ruleBreaks uses Types.h RuleBreakType (Harmony, Rhythm, etc.)
    // unified uses KellyTypes.h RuleBreakType (ModalMixture, CrossRhythm, etc.)
    // At this point, RuleBreakType refers to Types.h version (included last)
    // So we use integer values to map to KellyTypes version
    switch (rb.type) {
    case RuleBreakType::ModalMixture: // Types.h version
      unifiedRb.type = static_cast<KellyTypesRuleBreakType>(1); // ModalMixture
      break;
    case RuleBreakType::CrossRhythm:
      unifiedRb.type = static_cast<KellyTypesRuleBreakType>(4); // CrossRhythm
      break;
    case RuleBreakType::DynamicContrast:
      unifiedRb.type =
          static_cast<KellyTypesRuleBreakType>(6); // DynamicContrast
      break;
    case RuleBreakType::RegisterShift:
      unifiedRb.type = static_cast<KellyTypesRuleBreakType>(5); // RegisterShift
      break;
    case RuleBreakType::HarmonicAmbiguity:
      unifiedRb.type =
          static_cast<KellyTypesRuleBreakType>(7); // HarmonicAmbiguity
      break;
    default:
      unifiedRb.type = static_cast<KellyTypesRuleBreakType>(0); // None
    }
    unifiedRb.description = rb.description;
    unifiedRb.justification = rb.justification;
    unifiedRb.intensity = rb.intensity;
    unified.ruleBreaks.push_back(unifiedRb);
  }

  // Set defaults for unified-only fields
  unified.key = "C";
  unified.timeSignature = {4, 4};
  unified.chordProgression.clear();
  unified.melodicRange = 0.6f;
  unified.leapProbability = 0.3f;
  unified.baseVelocity = 0.6f;
  unified.productionNotes.clear();
  unified.confidence = 0.8f;

  return unified;
}
} // namespace

KellyBrain::KellyBrain() : pipeline_(std::make_unique<IntentPipeline>()) {
  // IntentPipeline is initialized
}

bool KellyBrain::initialize(const std::string &dataPath) {
  // The existing IntentPipeline already initializes EmotionThesaurus
  // This could load additional data if needed
  initialized_ = true;
  return true;
}

KellyTypesIntentResult KellyBrain::fromWound(const KellyTypesWound &wound) {
  // Wound parameter is KellyTypes::Wound (from header via alias)
  // Convert to Types::Wound for IntentPipeline
  Wound legacyWound = convertToLegacyWound(wound); // Wound here is Types::Wound

  // Call IntentPipeline with legacy types
  IntentResult legacyResult =
      pipeline_->process(legacyWound); // IntentResult is Types::IntentResult

  // Convert result back to unified types (KellyTypes::IntentResult)
  return convertFromLegacyIntentResult(legacyResult);
}

KellyTypesIntentResult KellyBrain::fromJourney(const SideA &current,
                                               const SideB &desired) {
  // SideA/SideB parameters are KellyTypes versions (from header)
  // Types.h has SideA and SideB with same structure, so we can use them
  // directly Both have: description, intensity, emotionId
  SideA legacyCurrent; // Types::SideA
  legacyCurrent.description = current.description;
  legacyCurrent.intensity = current.intensity;
  legacyCurrent.emotionId = current.emotionId;

  SideB legacyDesired; // Types::SideB
  legacyDesired.description = desired.description;
  legacyDesired.intensity = desired.intensity;
  legacyDesired.emotionId = desired.emotionId;

  // Call IntentPipeline
  IntentResult legacyResult = pipeline_->processJourney(
      legacyCurrent, legacyDesired); // Types::IntentResult

  // Convert result back to unified types
  return convertFromLegacyIntentResult(legacyResult);
}

KellyTypesIntentResult KellyBrain::fromText(const std::string &description) {
  // Create a wound from text description
  Wound wound = descriptionToWound(description);
  return fromWound(wound);
}

KellyTypesIntentResult KellyBrain::fromEmotion(const std::string &emotionName,
                                               float intensity) {
  // Look up emotion in thesaurus (returns Types.h EmotionNode)
  auto emotionOpt = pipeline_->thesaurus().findByName(emotionName);
  if (emotionOpt) {
    // Create wound from emotion
    KellyTypesWound wound;
    wound.description = "Feeling " + emotionName;
    wound.urgency = intensity;
    wound.intensity = intensity;
    wound.source = "emotion_selection";
    wound.expression = "Emotion: " + emotionName;

    // Set primary emotion from thesaurus result
    wound.primaryEmotion.id = emotionOpt->id;
    wound.primaryEmotion.name = emotionOpt->name;
    wound.primaryEmotion.categoryEnum =
        emotionOpt->categoryEnum; // Use categoryEnum, not category
    wound.primaryEmotion.category =
        categoryEnumToString(emotionOpt->categoryEnum);
    wound.primaryEmotion.valence = emotionOpt->valence;
    wound.primaryEmotion.arousal = emotionOpt->arousal;
    wound.primaryEmotion.dominance = emotionOpt->dominance;
    wound.primaryEmotion.intensity = emotionOpt->intensity;

    return fromWound(wound);
  }

  // Fallback: create basic wound
  return fromText("Feeling " + emotionName);
}

GeneratedMidi KellyBrain::generateMidi(const KellyTypesIntentResult &intent,
                                       int bars) {
  // This is a placeholder - actual MIDI generation should use MidiGenerator
  GeneratedMidi result;
  result.tempoBpm = intent.tempoBpm;
  result.bars = bars;
  result.key = intent.key;
  result.mode = intent.mode;
  // KellyTypes::GeneratedMidi doesn't have lengthInBeats or bpm fields
  // Store in metadata if needed
  result.metadata["lengthInBeats"] = std::to_string(bars * 4.0);
  result.metadata["bpm"] = std::to_string(intent.tempoBpm);

  // Generate basic chord progression
  for (const auto &chordSymbol : intent.chordProgression) {
    Chord chord;
    chord.symbol = chordSymbol;
    chord.root = chordSymbol.substr(0, 1); // Extract root note
    result.chords.push_back(chord);
  }

  // TODO: Use MidiGenerator to generate actual MIDI notes
  // For now, return empty result (notes vector is empty)

  return result;
}

GeneratedMidi KellyBrain::generateMidiFromWound(const KellyTypesWound &wound,
                                                int bars) {
  KellyTypesIntentResult result = fromWound(wound);
  return generateMidi(result, bars);
}

KellyTypesEmotionNode
KellyBrain::resolveEmotionByName(const std::string &emotionName) {
  // Try to find emotion in thesaurus (returns Types.h EmotionNode)
  auto emotionOpt = pipeline_->thesaurus().findByName(emotionName);
  if (emotionOpt) {
    // Convert Types.h EmotionNode to KellyTypes.h EmotionNode
    KellyTypesEmotionNode unified; // KellyTypes::EmotionNode
    unified.id = emotionOpt->id;
    unified.name = emotionOpt->name;
    unified.categoryEnum =
        emotionOpt->categoryEnum; // Use categoryEnum, not category
    unified.category = categoryEnumToString(emotionOpt->categoryEnum);
    unified.valence = emotionOpt->valence;
    unified.arousal = emotionOpt->arousal;
    unified.dominance = emotionOpt->dominance;
    unified.intensity = emotionOpt->intensity;
    unified.relatedEmotions = emotionOpt->relatedEmotions;
    // Set defaults for unified-only fields
    unified.synonyms.clear();
    unified.layerIndex = 0;
    unified.subIndex = 0;
    unified.subSubIndex = 0;
    return unified;
  }

  // Fallback: create a basic emotion node (KellyTypes::EmotionNode)
  KellyTypesEmotionNode fallback;
  fallback.name = emotionName;
  fallback.intensity = 0.5f;
  fallback.valence = 0.0f;
  fallback.arousal = 0.5f;
  fallback.dominance = 0.5f;
  fallback.categoryEnum = static_cast<EmotionCategory>(0); // Joy = 0
  fallback.category = "Joy";

  return fallback;
}

std::string KellyBrain::woundToDescription(const KellyTypesWound &wound) {
  if (!wound.expression.empty()) {
    return wound.description + " - " + wound.expression;
  }
  return wound.description;
}

KellyTypesWound KellyBrain::descriptionToWound(const std::string &description,
                                               float intensity) {
  KellyTypesWound wound;
  wound.description = description;
  wound.urgency = intensity;
  wound.intensity = intensity;
  wound.source = "text_input";
  wound.expression = description;
  return wound;
}

// Implement accessor methods that require IntentPipeline definition
IntentPipeline &KellyBrain::pipeline() { return *pipeline_; }

const IntentPipeline &KellyBrain::pipeline() const { return *pipeline_; }

IntentPipeline &KellyBrain::getIntentPipeline() { return *pipeline_; }

const IntentPipeline &KellyBrain::getIntentPipeline() const {
  return *pipeline_;
}

EmotionThesaurus &KellyBrain::thesaurus() { return pipeline_->thesaurus(); }

const EmotionThesaurus &KellyBrain::thesaurus() const {
  return pipeline_->thesaurus();
}

} // namespace kelly
