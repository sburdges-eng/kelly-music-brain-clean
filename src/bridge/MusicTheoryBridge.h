#pragma once
/*
 * MusicTheoryBridge.h - Python Music Theory Teacher Bridge
 * =========================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Python Layer: AdaptiveMusicTeacher, ExerciseGenerator, ExplanationEngine
 * - Engine Layer: MusicTheoryBrain (C++ music theory engine)
 * - UI Layer: MusicTheoryWorkstation (learning interface)
 * - Used By: MusicTheoryWorkstation, LearningPanel
 *
 * Purpose: Bridge between Python adaptive teacher and C++ MusicTheoryBrain
 *          Provides adaptive learning, exercise generation, and explanations
 */

#include "../music_theory/MusicTheoryBrain.h"
#include "../music_theory/Types.h"
#include <string>
#include <memory>
#include <map>
#include <chrono>

namespace kelly {

/**
 * MusicTheoryBridge - C++ interface to Python AdaptiveMusicTeacher
 *
 * Provides methods for adaptive learning, exercise generation, and explanations
 * that leverage Python ML capabilities while using C++ MusicTheoryBrain for core operations.
 *
 * Thread Safety:
 * - All methods are thread-safe
 * - Python calls are made from worker thread, not audio thread
 * - Results are cached to avoid repeated processing
 */
class MusicTheoryBridge {
public:
    MusicTheoryBridge();
    ~MusicTheoryBridge();

    /**
     * Explanation struct for bridge return type
     */
    struct Explanation {
        std::string text;
        midikompanion::theory::ExplanationType style;
        midikompanion::theory::ExplanationDepth depth;
    };

    /**
     * Explain a concept with specified style.
     *
     * @param concept Concept name (e.g., "Perfect Fifth")
     * @param style Explanation style ("intuitive", "mathematical", "historical", "acoustic")
     * @param userLevel User's current level (0-4: Beginner to Expert)
     * @return Explanation object with text and examples
     */
    Explanation explainConcept(
        const std::string& conceptName,
        const std::string& style,
        int userLevel = 1
    );

    /**
     * Generate exercise for a concept.
     *
     * @param conceptName Concept to practice
     * @param userProfileJson User profile JSON:
     *   {
     *     "userName": "User",
     *     "currentLevel": 1,
     *     "masteredConcepts": ["Interval", "Major Scale"],
     *     "strugglingConcepts": ["Voice Leading"]
     *   }
     * @return Exercise object with question, answer, and hints
     */
    midikompanion::theory::Exercise generateExercise(
        const std::string& conceptName,
        const std::string& userProfileJson
    );

    /**
     * Provide feedback on exercise attempt.
     *
     * @param exerciseJson Exercise JSON
     * @param attemptJson User attempt JSON:
     *   {
     *     "answer": "...",
     *     "timeSpent": 45,
     *     "hintsUsed": 1
     *   }
     * @return Feedback object with correctness, explanation, and suggestions
     */
    struct Feedback {
        bool isCorrect;
        std::string explanation;
        std::string hint;
        std::vector<std::string> suggestedReview;
    };

    Feedback provideFeedback(
        const std::string& exerciseJson,
        const std::string& attemptJson
    );

    /**
     * Create personalized lesson plan.
     *
     * @param conceptName Target concept
     * @param userProfileJson User profile JSON
     * @return Lesson plan JSON:
     *   {
     *     "concept": "...",
     *     "lessons": [
     *       {"step": 1, "concept": "...", "rationale": "...", "estimatedMinutes": 10},
     *       ...
     *     ],
     *     "totalEstimatedHours": 2.5
     *   }
     */
    std::string createLessonPlan(
        const std::string& conceptName,
        const std::string& userProfileJson
    );

    /**
     * Check if Python bridge is available.
     */
    bool isAvailable() const { return available_; }

    /**
     * Set MusicTheoryBrain instance (for direct engine access).
     */
    void setMusicTheoryBrain(midikompanion::theory::MusicTheoryBrain* brain) {
        brain_ = brain;
    }

private:
    bool available_;
    midikompanion::theory::MusicTheoryBrain* brain_ = nullptr;

    // Python function pointers
    void* explainConceptFunc_;
    void* generateExerciseFunc_;
    void* provideFeedbackFunc_;
    void* createLessonPlanFunc_;

    // Result cache
    struct CachedExplanation {
        Explanation explanation;
        std::string cacheKey;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::map<std::string, CachedExplanation> explanationCache_;
    static constexpr int CACHE_TTL_MS = 10000;  // Cache for 10 seconds

    bool initializePython();
    void shutdownPython();
    Explanation getCachedExplanation(const std::string& cacheKey);
    void cacheExplanation(const std::string& cacheKey, const Explanation& explanation);
    std::string hashRequest(const std::string& conceptName, const std::string& style, int userLevel);
};

} // namespace kelly
