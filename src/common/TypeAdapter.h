#pragma once
/**
 * TypeAdapter.h - Conversion utilities between Types.h and KellyTypes.h
 *
 * IMPORTANT USAGE NOTE:
 * This header uses a technique to handle type name conflicts. The conversion
 * functions work by having the IMPLEMENTATION file include both headers in
 * a controlled way. This header only provides declarations.
 *
 * To use these adapters:
 * 1. Include TypeAdapter.h in your .cpp file
 * 2. The implementation (TypeAdapter.cpp) handles including both type systems
 * 3. Call the conversion functions as needed
 *
 * NOTE: You cannot include both Types.h and KellyTypes.h in the same translation
 * unit due to type name conflicts. The adapter implementation handles this carefully.
 */

#include <string>
#include <vector>

// Forward declarations - actual types come from the respective headers
// We'll use a namespace alias technique in the implementation
namespace kelly {

// Forward declare the types from both systems
// The implementation will handle the actual conversions

// Conversion function declarations
// These will be implemented in TypeAdapter.cpp which can handle both type systems

// EmotionNode conversions
struct EmotionNode;  // Forward declare (actual definition depends on which header is included)
struct EmotionNode convertToUnifiedEmotionNode(const struct EmotionNode& legacy);
struct EmotionNode convertToLegacyEmotionNode(const struct EmotionNode& unified);

// Wound conversions
struct Wound;
struct Wound convertToUnifiedWound(const struct Wound& legacy);
struct Wound convertToLegacyWound(const struct Wound& unified);

// IntentResult conversions
struct IntentResult;
struct IntentResult convertToUnifiedIntentResult(const struct IntentResult& legacy, int tempoBpm = 120);
struct IntentResult convertToLegacyIntentResult(const struct IntentResult& unified);

// MidiNote conversions
struct MidiNote;
struct MidiNote convertToUnifiedMidiNote(const struct MidiNote& legacy, int tempoBpm = 120);
struct MidiNote convertToLegacyMidiNote(const struct MidiNote& unified);

// RuleBreak conversions
struct RuleBreak;
struct RuleBreak convertToUnifiedRuleBreak(const struct RuleBreak& legacy);
struct RuleBreak convertToLegacyRuleBreak(const struct RuleBreak& unified);

} // namespace kelly
