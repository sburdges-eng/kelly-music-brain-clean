#pragma once
/*
 * MidiKompanionBrain.h - High-Level Intent Processing API
 * ===============================================
 * Wraps existing IntentPipeline with a simplified high-level API
 *
 * This provides a convenience layer over the existing IntentPipeline,
 * allowing easier access to wound-to-music processing.
 *
 * CONNECTIONS (for Cursor Graph):
 * - Core Engine: IntentPipeline (wound → emotion → music)
 * - Emotion System: EmotionThesaurus (via IntentPipeline)
 * - Type System: KellyTypes.h (unified types)
 * - Used By: PluginProcessor, MLBridge, Python bridge (midikompanion_bridge.cpp)
 *
 * Data Flow:
 *   Wound/Text/Emotion → MidiKompanionBrain → IntentPipeline → IntentResult → GeneratedMidi
 */

// Include KellyTypes.h for the unified type system
#include "common/KellyTypes.h"
// Forward declare IntentPipeline to avoid including Types.h here
// We'll include IntentPipeline.h only in the .cpp file where we can handle conversions
#include <memory>
#include <string>

// Forward declare kelly types
namespace kelly {
    class IntentPipeline;
    class EmotionThesaurus;
}

namespace midikompanion {

// Import kelly types for use in this namespace
using kelly::IntentPipeline;
using kelly::EmotionThesaurus;
using kelly::Wound;
using kelly::SideA;
using kelly::SideB;
using kelly::IntentResult;
using kelly::GeneratedMidi;
using kelly::EmotionNode;

/**
 * MidiKompanionBrain - High-level interface for intent processing
 *
 * This class wraps the existing IntentPipeline to provide:
 * - Simplified initialization
 * - Convenient text-to-intent conversion
 * - Direct emotion-to-intent mapping
 *
 * Usage:
 *   MidiKompanionBrain brain;
 *   brain.initialize("./data");
 *   IntentResult result = brain.fromText("I feel lost and alone");
 */
class MidiKompanionBrain {
public:
    MidiKompanionBrain();
    ~MidiKompanionBrain();  // Defined in .cpp to handle incomplete IntentPipeline type

    /**
     * Initialize with data directory path
     * @param dataPath Path to data directory (for emotion thesaurus data)
     * @return true if initialization successful
     */
    bool initialize(const std::string& dataPath);

    /**
     * Process a wound and generate intent result
     * Uses existing IntentPipeline::process()
     */
    IntentResult fromWound(const Wound& wound);

    /**
     * Process a journey from Side A (current) to Side B (desired).
     * Creates a musical journey between two emotional states.
     */
    IntentResult fromJourney(const SideA& current, const SideB& desired);

    /**
     * Process text description and generate intent result
     * Creates a Wound from text and processes it
     */
    IntentResult fromText(const std::string& description);

    /**
     * Process emotion name and generate intent result
     * Looks up emotion in thesaurus and processes it
     */
    IntentResult fromEmotion(const std::string& emotionName, float intensity = 0.7f);

    /**
     * Generate MIDI from intent result
     */
    GeneratedMidi generateMidi(const IntentResult& intent, int bars = 8);

    /**
     * Generate MIDI directly from a wound (convenience method).
     * Combines fromWound() and generateMidi() in one call.
     */
    GeneratedMidi generateMidiFromWound(const Wound& wound, int bars = 8);

    /**
     * Get direct access to underlying IntentPipeline
     * Note: Returns a reference to the internal IntentPipeline which uses Types.h types
     */
    IntentPipeline& pipeline();
    const IntentPipeline& pipeline() const;

    /**
     * Get direct access to underlying IntentPipeline (alias for pipeline())
     */
    IntentPipeline& getIntentPipeline();
    const IntentPipeline& getIntentPipeline() const;

    /**
     * Get direct access to EmotionThesaurus
     */
    EmotionThesaurus& thesaurus();
    const EmotionThesaurus& thesaurus() const;

    /**
     * Check if initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * Convert wound to description string
     */
    static std::string woundToDescription(const Wound& wound);

    /**
     * Create wound from description string
     */
    static Wound descriptionToWound(const std::string& description, float intensity = 0.5f);

private:
    // Use pointer to avoid including IntentPipeline.h in header (which would include Types.h)
    // This allows us to keep KellyTypes.h types in the interface
    std::unique_ptr<IntentPipeline> pipeline_;
    bool initialized_ = false;

    /**
     * Resolve emotion name to EmotionNode (helper for fromEmotion)
     */
    EmotionNode resolveEmotionByName(const std::string& emotionName);
};

} // namespace kelly
