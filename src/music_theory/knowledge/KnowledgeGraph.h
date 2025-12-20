#pragma once
/*
 * KnowledgeGraph.h - Music Theory Concept Network and Learning Paths
 * ===================================================================
 *
 * Connects all music theory concepts into an intelligent knowledge graph:
 * - Concept nodes with prerequisites and relationships
 * - Multiple explanation styles (Intuitive, Acoustic, Mathematical, Historical)
 * - Adaptive learning paths based on user progress
 * - Musical examples with timestamps
 * - "Why does this work?" explanations at all depth levels
 *
 * CONNECTIONS:
 * - Uses: All theory engines (Core, Harmony, Rhythm)
 * - Used by: MusicTheoryBrain (main interface)
 * - Used by: Python Adaptive Teacher (ML-driven learning)
 * - UI: Interactive concept browser, learning roadmap
 */

#include "../Types.h"
#include <memory>
#include <set>
#include <queue>

namespace midikompanion::theory {

class KnowledgeGraph {
public:
    KnowledgeGraph();
    ~KnowledgeGraph() = default;

    //==========================================================================
    // Concept Management
    //==========================================================================

    /**
     * Add concept to knowledge graph
     *
     * @param node Complete concept definition
     */
    void addConcept(const KnowledgeNode& node);

    /**
     * Get concept by name
     *
     * @param conceptName Name of concept (e.g., "Perfect Fifth")
     * @return Concept node with all metadata
     */
    std::optional<KnowledgeNode> getConcept(const std::string& conceptName) const;

    /**
     * Get all concepts in a category
     *
     * @param category Category name ("Interval", "Chord", "Scale", "Rhythm")
     * @return All concepts in that category
     */
    std::vector<KnowledgeNode> getConceptsByCategory(const std::string& category) const;

    /**
     * Search concepts by keyword
     *
     * @param keyword Search term
     * @return Matching concepts sorted by relevance
     */
    std::vector<KnowledgeNode> searchConcepts(const std::string& keyword) const;

    /**
     * Get concept statistics
     * @return Total concepts, categories, connections
     */
    struct GraphStatistics {
        int totalConcepts;
        int totalCategories;
        int totalConnections;
        std::map<std::string, int> conceptsPerCategory;
        float avgConnectionsPerConcept;
    };

    GraphStatistics getStatistics() const;

    //==========================================================================
    // Concept Relationships
    //==========================================================================

    /**
     * Get prerequisites for a concept
     *
     * @param conceptName Concept to check
     * @return Ordered list of prerequisite concepts
     *
     * Example:
     *   auto prereqs = getPrerequisites("Secondary Dominant");
     *   // Returns: ["Interval", "Scale Degree", "Dominant Chord"]
     */
    std::vector<std::string> getPrerequisites(const std::string& conceptName) const;

    /**
     * Get related concepts
     *
     * @param conceptName Starting concept
     * @param maxDepth How many relationship hops (default: 1)
     * @return Related concepts with relationship type
     *
     * Example:
     *   auto related = getRelatedConcepts("Major Scale");
     *   // Returns: [{"Minor Scale", "Parallel mode"},
     *   //           {"Ionian Mode", "Synonymous"},
     *   //           {"Key Signature", "Notational"}]
     */
    struct ConceptRelationship {
        std::string conceptName;
        std::string relationshipType;  // "Prerequisite", "Related", "Opposite", "Example"
        float strength;                // 0-1 (how strong the relationship)
    };

    std::vector<ConceptRelationship> getRelatedConcepts(
        const std::string& conceptName,
        int maxDepth = 1
    ) const;

    /**
     * Get application contexts for a concept
     *
     * "Where is this used in real music?"
     *
     * @param conceptName Concept to explore
     * @return List of practical applications
     */
    std::vector<std::string> getApplications(const std::string& conceptName) const;

    /**
     * Find concept dependencies (what needs to be learned first)
     *
     * @param conceptName Target concept
     * @return Topologically sorted list of prerequisites
     */
    std::vector<std::string> findDependencies(const std::string& conceptName) const;

    /**
     * Check if user has prerequisites for concept
     *
     * @param conceptName Concept to check
     * @param masteredConcepts Concepts user has mastered
     * @return true if ready to learn, false if missing prerequisites
     */
    bool hasPrerequisites(
        const std::string& conceptName,
        const std::set<std::string>& masteredConcepts
    ) const;

    /**
     * Get missing prerequisites
     *
     * @param conceptName Target concept
     * @param masteredConcepts What user knows
     * @return List of concepts to learn first
     */
    std::vector<std::string> getMissingPrerequisites(
        const std::string& conceptName,
        const std::set<std::string>& masteredConcepts
    ) const;

    //==========================================================================
    // Learning Path Generation
    //==========================================================================

    /**
     * Generate learning path from current concept to target
     *
     * @param fromConcept Starting point (what user knows)
     * @param toConcept Goal concept (what user wants to learn)
     * @return Ordered sequence of concepts to study
     *
     * Example:
     *   auto path = getLearningPath("Major Scale", "Jazz Reharmonization");
     *   // Returns: ["Chord Construction", "Chord Progressions",
     *   //           "Functional Harmony", "Secondary Dominants",
     *   //           "Tritone Substitution", "Jazz Reharmonization"]
     */
    struct LearningPathStep {
        std::string conceptName;
        std::string rationale;         // Why this step is necessary
        int estimatedMinutes;          // Time to master
        std::vector<std::string> exercises;
    };

    std::vector<LearningPathStep> getLearningPath(
        const std::string& fromConcept,
        const std::string& toConcept
    ) const;

    /**
     * Generate personalized curriculum
     *
     * @param userProfile Current user state
     * @param targetConcepts Goals
     * @param maxLessons Maximum lessons to generate
     * @return Complete curriculum with lessons
     */
    struct Curriculum {
        std::vector<LearningPathStep> lessons;
        int totalEstimatedHours;
        std::vector<std::string> milestones;
        std::string description;
    };

    Curriculum generateCurriculum(
        const UserProfile& userProfile,
        const std::vector<std::string>& targetConcepts,
        int maxLessons = 20
    ) const;

    /**
     * Suggest next concepts to learn
     *
     * @param masteredConcepts What user already knows
     * @param numSuggestions How many to return
     * @return Recommended next concepts with explanations
     */
    struct ConceptSuggestion {
        std::string conceptName;
        std::string category;
        std::string reason;            // Why learn this now
        float priority;                // 0-1 (higher = more important)
        bool prerequisitesMet;
    };

    std::vector<ConceptSuggestion> suggestNextConcepts(
        const std::set<std::string>& masteredConcepts,
        int numSuggestions = 5
    ) const;

    /**
     * Get recommended practice order for multiple concepts
     *
     * @param concepts Concepts to learn
     * @return Optimally ordered learning sequence
     */
    std::vector<std::string> getOptimalLearningOrder(
        const std::vector<std::string>& concepts
    ) const;

    //==========================================================================
    // Explanations
    //==========================================================================

    /**
     * Explain concept with specific style
     *
     * @param conceptName Concept to explain
     * @param type Explanation type (Intuitive, Acoustic, Mathematical, Historical)
     * @param depth Depth level (Simple, Intermediate, Advanced, Expert)
     * @return Tailored explanation
     *
     * Example:
     *   auto exp = explainConcept("Perfect Fifth", ExplanationType::Intuitive,
     *                            ExplanationDepth::Simple);
     *   // Returns: "Think of it like the distance from 'do' to 'sol' in
     *   //           'do-re-mi'. It sounds strong and stable."
     */
    std::string explainConcept(
        const std::string& conceptName,
        ExplanationType type,
        ExplanationDepth depth
    ) const;

    /**
     * Get all available explanation types for concept
     */
    std::vector<ExplanationType> getAvailableExplanationTypes(
        const std::string& conceptName
    ) const;

    /**
     * Get musical examples for concept
     *
     * @param conceptName Concept to demonstrate
     * @param maxExamples Maximum examples to return
     * @return List of songs with timestamps and descriptions
     */
    std::vector<KnowledgeNode::MusicalExample> getMusicalExamples(
        const std::string& conceptName,
        int maxExamples = 5
    ) const;

    /**
     * Explain relationship between two concepts
     *
     * "How does X relate to Y?"
     *
     * @param concept1 First concept
     * @param concept2 Second concept
     * @return Explanation of relationship
     */
    std::string explainRelationship(
        const std::string& concept1Name,
        const std::string& concept2Name
    ) const;

    //==========================================================================
    // Concept Discovery
    //==========================================================================

    /**
     * Find concept by example
     *
     * "What is this called?"
     *
     * @param description User description
     * @return Matching concepts with confidence scores
     */
    struct ConceptMatch {
        std::string conceptName;
        float confidence;              // 0-1 (how well it matches)
        std::string explanation;
    };

    std::vector<ConceptMatch> findConceptByDescription(
        const std::string& description
    ) const;

    /**
     * Identify concept from MIDI analysis
     *
     * Automatically identify what theory concepts are present in MIDI
     *
     * @param midiNotes MIDI note numbers
     * @param onsetTimes Note onset times
     * @return Identified concepts with evidence
     */
    struct IdentifiedConcept {
        std::string conceptName;
        std::string category;
        std::string evidence;          // Why we think it's this concept
        std::vector<int> relevantNotes; // Which notes demonstrate it
    };

    std::vector<IdentifiedConcept> identifyConceptsFromMIDI(
        const std::vector<int>& midiNotes,
        const std::vector<float>& onsetTimes
    ) const;

    /**
     * Get conceptual hierarchy
     *
     * Shows parent-child relationships
     *
     * Example:
     *   "Chord" → "Triad" → "Major Triad"
     *   "Chord" → "Seventh Chord" → "Dominant Seventh"
     */
    struct ConceptHierarchy {
        std::string conceptName;
        std::vector<ConceptHierarchy> children;
        int depth;
    };

    ConceptHierarchy getConceptHierarchy(const std::string& rootConcept) const;

    //==========================================================================
    // Interactive Learning
    //==========================================================================

    /**
     * Generate practice exercise for concept
     *
     * @param conceptName Concept to practice
     * @param difficultyLevel User's current level
     * @return Exercise tailored to concept and level
     */
    Exercise generateExercise(
        const std::string& conceptName,
        DifficultyLevel difficultyLevel
    ) const;

    /**
     * Validate user answer and provide feedback
     *
     * @param exercise The exercise
     * @param userAnswer User's response
     * @param userProfile User's learning profile
     * @return Feedback with hints and guidance
     */
    Feedback validateAnswer(
        const Exercise& exercise,
        const std::string& userAnswer,
        const UserProfile& userProfile
    ) const;

    /**
     * Update user mastery based on performance
     *
     * @param conceptName Concept being practiced
     * @param correct Was answer correct
     * @param timeSeconds Time taken to answer
     * @return Updated mastery score (0-1)
     */
    float updateMastery(
        const std::string& conceptName,
        bool correct,
        float timeSeconds,
        UserProfile& userProfile
    ) const;

    /**
     * Identify struggling concepts
     *
     * @param userProfile User's profile
     * @return Concepts that need review
     */
    std::vector<std::string> identifyStrugglingConcepts(
        const UserProfile& userProfile
    ) const;

    /**
     * Suggest review concepts (spaced repetition)
     *
     * @param userProfile User's state
     * @return Concepts due for review
     */
    std::vector<std::string> suggestReview(
        const UserProfile& userProfile
    ) const;

    //==========================================================================
    // Data Import/Export
    //==========================================================================

    /**
     * Load knowledge graph from JSON files
     *
     * @param dataDirectory Path to data/music_theory/ directory
     * @return true if successful
     */
    bool loadFromJSON(const std::string& dataDirectory);

    /**
     * Save knowledge graph to JSON files
     *
     * @param dataDirectory Output directory
     * @return true if successful
     */
    bool saveToJSON(const std::string& dataDirectory) const;

    /**
     * Load user profile
     */
    bool loadUserProfile(const std::string& filePath, UserProfile& profile) const;

    /**
     * Save user profile
     */
    bool saveUserProfile(const std::string& filePath, const UserProfile& profile) const;

    /**
     * Export learning progress report
     *
     * @param userProfile User's profile
     * @return Formatted progress report
     */
    std::string exportProgressReport(const UserProfile& userProfile) const;

    //==========================================================================
    // Concept Validation
    //==========================================================================

    /**
     * Verify graph integrity
     *
     * Checks for:
     * - Missing prerequisites
     * - Circular dependencies
     * - Orphaned concepts
     *
     * @return List of issues found
     */
    std::vector<std::string> validateGraph() const;

    /**
     * Check for circular dependencies
     */
    bool hasCircularDependencies() const;

    /**
     * Find orphaned concepts (no connections)
     */
    std::vector<std::string> findOrphanedConcepts() const;

private:
    //==========================================================================
    // Internal Data
    //==========================================================================

    // Main concept storage
    std::map<std::string, KnowledgeNode> concepts_;

    // Category index
    std::map<std::string, std::vector<std::string>> categoryIndex_;

    // Adjacency list for graph traversal
    std::map<std::string, std::vector<std::string>> prerequisiteGraph_;
    std::map<std::string, std::vector<std::string>> relatedConceptsGraph_;

    // Concept metadata
    struct ConceptMetadata {
        int avgTimeToMaster;           // Minutes
        float difficulty;              // 0-1
        std::vector<ExerciseType> recommendedExercises;
    };
    std::map<std::string, ConceptMetadata> metadata_;

    //==========================================================================
    // Internal Helpers
    //==========================================================================

    // Graph algorithms
    std::vector<std::string> topologicalSort(
        const std::string& startConcept
    ) const;

    std::vector<std::string> breadthFirstSearch(
        const std::string& start,
        const std::string& goal
    ) const;

    std::vector<std::string> depthFirstSearch(
        const std::string& start,
        std::set<std::string>& visited
    ) const;

    bool hasPath(
        const std::string& from,
        const std::string& to,
        std::set<std::string>& visited
    ) const;

    // Cycle detection
    bool hasCycleDFS(
        const std::string& node,
        std::set<std::string>& visited,
        std::set<std::string>& recursionStack
    ) const;

    // Path finding
    std::vector<std::string> findShortestPath(
        const std::string& from,
        const std::string& to
    ) const;

    // Relationship scoring
    float calculateRelationshipStrength(
        const std::string& concept1Name,
        const std::string& concept2Name
    ) const;

    // Concept matching
    float calculateConceptSimilarity(
        const std::string& description,
        const KnowledgeNode& conceptNode
    ) const;

    // Learning path optimization
    int estimateLearningTime(const std::string& conceptName) const;

    std::vector<std::string> optimizePathOrder(
        const std::vector<std::string>& concepts
    ) const;

    // Mastery calculations
    float calculateMasteryDecay(
        float currentMastery,
        float daysSinceLastPractice
    ) const;

    float calculateNewMastery(
        float currentMastery,
        bool correct,
        float responseTime,
        int attemptCount
    ) const;

    // Exercise generation
    Exercise generateIntervalExercise(DifficultyLevel level) const;
    Exercise generateChordExercise(DifficultyLevel level) const;
    Exercise generateProgressionExercise(DifficultyLevel level) const;
    Exercise generateRhythmExercise(DifficultyLevel level) const;

    // JSON parsing helpers
    bool parseConceptJSON(const std::string& jsonPath);
    bool parseRelationshipsJSON(const std::string& jsonPath);
    void parseConceptFromJSON(const std::string& jsonStr);

    // String matching
    std::vector<std::string> tokenize(const std::string& text) const;
    float fuzzyMatch(const std::string& str1, const std::string& str2) const;
};

} // namespace midikompanion::theory
