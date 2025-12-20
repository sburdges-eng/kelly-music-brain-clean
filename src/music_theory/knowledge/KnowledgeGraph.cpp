/*
 * KnowledgeGraph.cpp - Music Theory Concept Network Implementation
 * =================================================================
 */

#include "KnowledgeGraph.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <queue>
#include <stack>
#include <cmath>
#include <fstream>

namespace midikompanion::theory {

//==============================================================================
// Constructor
//==============================================================================

KnowledgeGraph::KnowledgeGraph() {
    // Initialize with fundamental concepts
    // In production, these would be loaded from JSON files
    initializeBasicConcepts();
}

//==============================================================================
// Concept Management
//==============================================================================

void KnowledgeGraph::addConcept(const KnowledgeNode& node) {
    concepts_[node.conceptName] = node;

    // Update category index
    categoryIndex_[node.category].push_back(node.conceptName);

    // Update prerequisite graph
    for (const auto& prereq : node.prerequisites) {
        prerequisiteGraph_[node.conceptName].push_back(prereq);
    }

    // Update related concepts graph
    for (const auto& related : node.relatedConcepts) {
        relatedConceptsGraph_[node.conceptName].push_back(related);
    }
}

std::optional<KnowledgeNode> KnowledgeGraph::getConcept(
    const std::string& conceptName) const
{
    auto it = concepts_.find(conceptName);
    if (it != concepts_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<KnowledgeNode> KnowledgeGraph::getConceptsByCategory(
    const std::string& category) const
{
    std::vector<KnowledgeNode> results;

    auto it = categoryIndex_.find(category);
    if (it != categoryIndex_.end()) {
        for (const auto& conceptName : it->second) {
            auto concept = getConcept(conceptName);
            if (concept) {
                results.push_back(*concept);
            }
        }
    }

    return results;
}

std::vector<KnowledgeNode> KnowledgeGraph::searchConcepts(
    const std::string& keyword) const
{
    std::vector<std::pair<KnowledgeNode, float>> matches;

    std::string lowerKeyword = keyword;
    std::transform(lowerKeyword.begin(), lowerKeyword.end(),
                  lowerKeyword.begin(), ::tolower);

    for (const auto& [name, node] : concepts_) {
        std::string lowerName = name;
        std::transform(lowerName.begin(), lowerName.end(),
                      lowerName.begin(), ::tolower);

        // Check if keyword appears in concept name
        if (lowerName.find(lowerKeyword) != std::string::npos) {
            float relevance = 1.0f;
            if (lowerName == lowerKeyword) {
                relevance = 2.0f; // Exact match
            }
            matches.push_back({node, relevance});
        }
    }

    // Sort by relevance
    std::sort(matches.begin(), matches.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<KnowledgeNode> results;
    for (const auto& [node, _] : matches) {
        results.push_back(node);
    }

    return results;
}

KnowledgeGraph::GraphStatistics KnowledgeGraph::getStatistics() const {
    GraphStatistics stats;

    stats.totalConcepts = concepts_.size();
    stats.totalCategories = categoryIndex_.size();

    int totalConnections = 0;
    for (const auto& [_, prereqs] : prerequisiteGraph_) {
        totalConnections += prereqs.size();
    }
    for (const auto& [_, related] : relatedConceptsGraph_) {
        totalConnections += related.size();
    }
    stats.totalConnections = totalConnections;

    // Concepts per category
    for (const auto& [category, concepts] : categoryIndex_) {
        stats.conceptsPerCategory[category] = concepts.size();
    }

    // Average connections
    if (stats.totalConcepts > 0) {
        stats.avgConnectionsPerConcept =
            static_cast<float>(totalConnections) / stats.totalConcepts;
    }

    return stats;
}

//==============================================================================
// Concept Relationships
//==============================================================================

std::vector<std::string> KnowledgeGraph::getPrerequisites(
    const std::string& conceptName) const
{
    auto concept = getConcept(conceptName);
    if (concept) {
        return concept->prerequisites;
    }
    return {};
}

std::vector<KnowledgeGraph::ConceptRelationship>
KnowledgeGraph::getRelatedConcepts(
    const std::string& conceptName,
    int maxDepth) const
{
    std::vector<ConceptRelationship> relationships;

    auto concept = getConcept(conceptName);
    if (!concept) return relationships;

    // Direct relationships
    for (const auto& related : concept->relatedConcepts) {
        ConceptRelationship rel;
        rel.conceptName = related;
        rel.relationshipType = "Related";
        rel.strength = calculateRelationshipStrength(conceptName, related);
        relationships.push_back(rel);
    }

    // Prerequisites (reverse relationship)
    for (const auto& prereq : concept->prerequisites) {
        ConceptRelationship rel;
        rel.conceptName = prereq;
        rel.relationshipType = "Prerequisite";
        rel.strength = 0.9f;
        relationships.push_back(rel);
    }

    // If depth > 1, explore second-level relationships
    if (maxDepth > 1) {
        for (const auto& related : concept->relatedConcepts) {
            auto secondLevel = getRelatedConcepts(related, maxDepth - 1);
            for (auto& rel : secondLevel) {
                rel.strength *= 0.5f; // Reduce strength for indirect relationships
                relationships.push_back(rel);
            }
        }
    }

    // Remove duplicates
    std::sort(relationships.begin(), relationships.end(),
             [](const auto& a, const auto& b) { return a.conceptName < b.conceptName; });
    relationships.erase(
        std::unique(relationships.begin(), relationships.end(),
                   [](const auto& a, const auto& b) { return a.conceptName == b.conceptName; }),
        relationships.end()
    );

    return relationships;
}

std::vector<std::string> KnowledgeGraph::getApplications(
    const std::string& conceptName) const
{
    auto concept = getConcept(conceptName);
    if (concept) {
        return concept->applications;
    }
    return {};
}

std::vector<std::string> KnowledgeGraph::findDependencies(
    const std::string& conceptName) const
{
    return topologicalSort(conceptName);
}

bool KnowledgeGraph::hasPrerequisites(
    const std::string& conceptName,
    const std::set<std::string>& masteredConcepts) const
{
    auto prereqs = getPrerequisites(conceptName);

    for (const auto& prereq : prereqs) {
        if (masteredConcepts.find(prereq) == masteredConcepts.end()) {
            return false;
        }
    }

    return true;
}

std::vector<std::string> KnowledgeGraph::getMissingPrerequisites(
    const std::string& conceptName,
    const std::set<std::string>& masteredConcepts) const
{
    std::vector<std::string> missing;

    auto prereqs = getPrerequisites(conceptName);

    for (const auto& prereq : prereqs) {
        if (masteredConcepts.find(prereq) == masteredConcepts.end()) {
            missing.push_back(prereq);
        }
    }

    return missing;
}

//==============================================================================
// Learning Path Generation
//==============================================================================

std::vector<KnowledgeGraph::LearningPathStep>
KnowledgeGraph::getLearningPath(
    const std::string& fromConcept,
    const std::string& toConcept) const
{
    std::vector<LearningPathStep> path;

    // Find shortest path through concept graph
    auto concepts = findShortestPath(fromConcept, toConcept);

    for (const auto& conceptName : concepts) {
        LearningPathStep step;
        step.conceptName = conceptName;
        step.estimatedMinutes = estimateLearningTime(conceptName);

        // Generate rationale
        auto concept = getConcept(conceptName);
        if (concept) {
            step.rationale = "Essential for understanding " + toConcept;

            // Add suggested exercises
            auto it = metadata_.find(conceptName);
            if (it != metadata_.end()) {
                for (auto exerciseType : it->second.recommendedExercises) {
                    // Convert exercise type to string
                    step.exercises.push_back("Practice exercise");
                }
            }
        }

        path.push_back(step);
    }

    return path;
}

KnowledgeGraph::Curriculum KnowledgeGraph::generateCurriculum(
    const UserProfile& userProfile,
    const std::vector<std::string>& targetConcepts,
    int maxLessons) const
{
    Curriculum curriculum;

    // Get mastered concepts
    std::set<std::string> mastered;
    for (const auto& [concept, masteryLevel] : userProfile.conceptMastery) {
        if (masteryLevel > 0.7f) {
            mastered.insert(concept);
        }
    }

    // Generate learning paths for each target
    std::vector<std::string> allConcepts;
    for (const auto& target : targetConcepts) {
        auto missing = getMissingPrerequisites(target, mastered);
        allConcepts.insert(allConcepts.end(), missing.begin(), missing.end());
        allConcepts.push_back(target);
    }

    // Remove duplicates and optimize order
    std::sort(allConcepts.begin(), allConcepts.end());
    allConcepts.erase(std::unique(allConcepts.begin(), allConcepts.end()),
                     allConcepts.end());

    auto optimized = getOptimalLearningOrder(allConcepts);

    // Limit to max lessons
    if (static_cast<int>(optimized.size()) > maxLessons) {
        optimized.resize(maxLessons);
    }

    // Convert to curriculum
    int totalMinutes = 0;
    for (const auto& conceptName : optimized) {
        LearningPathStep step;
        step.conceptName = conceptName;
        step.estimatedMinutes = estimateLearningTime(conceptName);
        totalMinutes += step.estimatedMinutes;

        auto concept = getConcept(conceptName);
        if (concept) {
            step.rationale = "Building towards: " + targetConcepts[0];
        }

        curriculum.lessons.push_back(step);
    }

    curriculum.totalEstimatedHours = totalMinutes / 60;
    curriculum.description = "Personalized learning path to master " +
                            std::to_string(targetConcepts.size()) + " concepts";

    // Add milestones every 5 lessons
    for (size_t i = 4; i < curriculum.lessons.size(); i += 5) {
        curriculum.milestones.push_back(curriculum.lessons[i].concept);
    }

    return curriculum;
}

std::vector<KnowledgeGraph::ConceptSuggestion>
KnowledgeGraph::suggestNextConcepts(
    const std::set<std::string>& masteredConcepts,
    int numSuggestions) const
{
    std::vector<ConceptSuggestion> suggestions;

    for (const auto& [conceptName, node] : concepts_) {
        // Skip if already mastered
        if (masteredConcepts.find(conceptName) != masteredConcepts.end()) {
            continue;
        }

        // Check prerequisites
        bool prereqsMet = hasPrerequisites(conceptName, masteredConcepts);

        ConceptSuggestion suggestion;
        suggestion.conceptName = conceptName;
        suggestion.category = node.category;
        suggestion.prerequisitesMet = prereqsMet;

        if (prereqsMet) {
            suggestion.priority = 0.9f;
            suggestion.reason = "Ready to learn (prerequisites met)";
        } else {
            suggestion.priority = 0.3f;
            suggestion.reason = "Missing prerequisites";
        }

        suggestions.push_back(suggestion);
    }

    // Sort by priority
    std::sort(suggestions.begin(), suggestions.end(),
             [](const auto& a, const auto& b) { return a.priority > b.priority; });

    // Return top N
    if (static_cast<int>(suggestions.size()) > numSuggestions) {
        suggestions.resize(numSuggestions);
    }

    return suggestions;
}

std::vector<std::string> KnowledgeGraph::getOptimalLearningOrder(
    const std::vector<std::string>& concepts) const
{
    return optimizePathOrder(concepts);
}

//==============================================================================
// Explanations
//==============================================================================

std::string KnowledgeGraph::explainConcept(
    const std::string& conceptName,
    ExplanationType type,
    ExplanationDepth depth) const
{
    auto concept = getConcept(conceptName);
    if (!concept) {
        return "Concept not found: " + conceptName;
    }

    auto it = concept->explanations.find(type);
    if (it != concept->explanations.end()) {
        return it->second;
    }

    // Default explanation
    return "The concept of " + conceptName + " in the " + concept->category + " category.";
}

std::vector<ExplanationType> KnowledgeGraph::getAvailableExplanationTypes(
    const std::string& conceptName) const
{
    std::vector<ExplanationType> types;

    auto concept = getConcept(conceptName);
    if (concept) {
        for (const auto& [type, _] : concept->explanations) {
            types.push_back(type);
        }
    }

    return types;
}

std::vector<KnowledgeNode::MusicalExample> KnowledgeGraph::getMusicalExamples(
    const std::string& conceptName,
    int maxExamples) const
{
    auto concept = getConcept(conceptName);
    if (!concept) return {};

    auto examples = concept->examples;

    if (static_cast<int>(examples.size()) > maxExamples) {
        examples.resize(maxExamples);
    }

    return examples;
}

std::string KnowledgeGraph::explainRelationship(
    const std::string& concept1,
    const std::string& concept2) const
{
    auto node1 = getConcept(concept1);
    auto node2 = getConcept(concept2);

    if (!node1 || !node2) {
        return "One or both concepts not found";
    }

    // Check if concept2 is a prerequisite of concept1
    auto prereqs1 = getPrerequisites(concept1);
    if (std::find(prereqs1.begin(), prereqs1.end(), concept2) != prereqs1.end()) {
        return concept2 + " is a prerequisite for " + concept1 +
               ". You need to understand " + concept2 + " before learning " + concept1 + ".";
    }

    // Check reverse
    auto prereqs2 = getPrerequisites(concept2);
    if (std::find(prereqs2.begin(), prereqs2.end(), concept1) != prereqs2.end()) {
        return concept1 + " is a prerequisite for " + concept2 +
               ". Understanding " + concept1 + " helps you learn " + concept2 + ".";
    }

    // Check if related
    if (std::find(node1->relatedConcepts.begin(), node1->relatedConcepts.end(), concept2)
        != node1->relatedConcepts.end()) {
        return concept1 + " and " + concept2 + " are related concepts. " +
               "They often appear together in music.";
    }

    return concept1 + " and " + concept2 + " are separate concepts in music theory.";
}

//==============================================================================
// Concept Discovery
//==============================================================================

std::vector<KnowledgeGraph::ConceptMatch>
KnowledgeGraph::findConceptByDescription(
    const std::string& description) const
{
    std::vector<ConceptMatch> matches;

    for (const auto& [name, node] : concepts_) {
        float similarity = calculateConceptSimilarity(description, node);

        if (similarity > 0.3f) {
            ConceptMatch match;
            match.conceptName = name;
            match.confidence = similarity;
            match.explanation = "Matches description: " + description;
            matches.push_back(match);
        }
    }

    // Sort by confidence
    std::sort(matches.begin(), matches.end(),
             [](const auto& a, const auto& b) { return a.confidence > b.confidence; });

    return matches;
}

std::vector<KnowledgeGraph::IdentifiedConcept>
KnowledgeGraph::identifyConceptsFromMIDI(
    const std::vector<int>& midiNotes,
    const std::vector<float>& onsetTimes) const
{
    std::vector<IdentifiedConcept> identified;

    // Identify intervals
    if (midiNotes.size() >= 2) {
        int interval = std::abs(midiNotes[1] - midiNotes[0]);

        IdentifiedConcept concept;
        concept.category = "Interval";
        concept.relevantNotes = {midiNotes[0], midiNotes[1]};

        if (interval == 7) {
            concept.conceptName = "Perfect Fifth";
            concept.evidence = "7 semitones between notes";
        } else if (interval == 4) {
            concept.conceptName = "Major Third";
            concept.evidence = "4 semitones between notes";
        } else if (interval == 3) {
            concept.conceptName = "Minor Third";
            concept.evidence = "3 semitones between notes";
        }

        if (!concept.conceptName.empty()) {
            identified.push_back(concept);
        }
    }

    // Identify chords (3+ simultaneous notes)
    if (midiNotes.size() >= 3) {
        IdentifiedConcept concept;
        concept.category = "Chord";
        concept.relevantNotes = {midiNotes[0], midiNotes[1], midiNotes[2]};

        // Simple chord detection
        std::vector<int> intervals;
        for (size_t i = 1; i < std::min(size_t(3), midiNotes.size()); ++i) {
            intervals.push_back((midiNotes[i] - midiNotes[0]) % 12);
        }

        if (intervals.size() >= 2) {
            if (intervals[0] == 4 && intervals[1] == 7) {
                concept.conceptName = "Major Triad";
                concept.evidence = "Major third + Perfect fifth structure";
            } else if (intervals[0] == 3 && intervals[1] == 7) {
                concept.conceptName = "Minor Triad";
                concept.evidence = "Minor third + Perfect fifth structure";
            }
        }

        if (!concept.conceptName.empty()) {
            identified.push_back(concept);
        }
    }

    return identified;
}

KnowledgeGraph::ConceptHierarchy KnowledgeGraph::getConceptHierarchy(
    const std::string& rootConcept) const
{
    ConceptHierarchy hierarchy;
    hierarchy.conceptName = rootConcept;
    hierarchy.depth = 0;

    // Find all concepts that have rootConcept as a prerequisite
    for (const auto& [name, node] : concepts_) {
        auto prereqs = node.prerequisites;
        if (std::find(prereqs.begin(), prereqs.end(), rootConcept) != prereqs.end()) {
            // This concept depends on rootConcept, so it's a child
            ConceptHierarchy child = getConceptHierarchy(name);
            child.depth = 1;
            hierarchy.children.push_back(child);
        }
    }

    return hierarchy;
}

//==============================================================================
// Interactive Learning
//==============================================================================

Exercise KnowledgeGraph::generateExercise(
    const std::string& conceptName,
    DifficultyLevel difficultyLevel) const
{
    auto concept = getConcept(conceptName);
    if (!concept) {
        return Exercise{};
    }

    // Route to specific generator based on category
    if (concept->category == "Interval") {
        return generateIntervalExercise(difficultyLevel);
    } else if (concept->category == "Chord") {
        return generateChordExercise(difficultyLevel);
    } else if (concept->category == "Progression") {
        return generateProgressionExercise(difficultyLevel);
    } else if (concept->category == "Rhythm") {
        return generateRhythmExercise(difficultyLevel);
    }

    // Default exercise
    Exercise exercise;
    exercise.type = ExerciseType::IntervalRecognition;
    exercise.level = difficultyLevel;
    exercise.concept = conceptName;
    exercise.instruction = "Identify the " + conceptName;

    return exercise;
}

Feedback KnowledgeGraph::validateAnswer(
    const Exercise& exercise,
    const std::string& userAnswer,
    const UserProfile& userProfile) const
{
    Feedback feedback;

    // Simple validation (case-insensitive comparison)
    std::string lowerAnswer = userAnswer;
    std::string lowerCorrect = exercise.correctAnswer;

    std::transform(lowerAnswer.begin(), lowerAnswer.end(),
                  lowerAnswer.begin(), ::tolower);
    std::transform(lowerCorrect.begin(), lowerCorrect.end(),
                  lowerCorrect.begin(), ::tolower);

    feedback.correct = (lowerAnswer == lowerCorrect);

    if (feedback.correct) {
        feedback.message = "Correct! Well done.";
        feedback.encourageRetry = false;
    } else {
        feedback.message = "Not quite. The correct answer is: " + exercise.correctAnswer;
        feedback.hints = exercise.hints;
        feedback.encourageRetry = true;

        // Suggest review
        feedback.suggestedReview.push_back(exercise.concept);

        // Check prerequisites
        auto prereqs = getPrerequisites(exercise.concept);
        for (const auto& prereq : prereqs) {
            auto it = userProfile.conceptMastery.find(prereq);
            if (it == userProfile.conceptMastery.end() || it->second < 0.5f) {
                feedback.suggestedReview.push_back(prereq);
            }
        }
    }

    return feedback;
}

float KnowledgeGraph::updateMastery(
    const std::string& conceptName,
    bool correct,
    float timeSeconds,
    UserProfile& userProfile) const
{
    float currentMastery = 0.0f;

    auto it = userProfile.conceptMastery.find(conceptName);
    if (it != userProfile.conceptMastery.end()) {
        currentMastery = it->second;
    }

    // Calculate new mastery
    float newMastery = calculateNewMastery(currentMastery, correct, timeSeconds, 1);

    userProfile.conceptMastery[conceptName] = newMastery;

    return newMastery;
}

std::vector<std::string> KnowledgeGraph::identifyStrugglingConcepts(
    const UserProfile& userProfile) const
{
    std::vector<std::string> struggling;

    for (const auto& [concept, mastery] : userProfile.conceptMastery) {
        if (mastery < 0.4f) {
            struggling.push_back(concept);
        }
    }

    return struggling;
}

std::vector<std::string> KnowledgeGraph::suggestReview(
    const UserProfile& userProfile) const
{
    std::vector<std::string> review;

    // Concepts with moderate mastery (need reinforcement)
    for (const auto& [concept, mastery] : userProfile.conceptMastery) {
        if (mastery >= 0.4f && mastery <= 0.7f) {
            review.push_back(concept);
        }
    }

    // Limit to 5 suggestions
    if (review.size() > 5) {
        review.resize(5);
    }

    return review;
}

//==============================================================================
// Data Import/Export
//==============================================================================

bool KnowledgeGraph::loadFromJSON(const std::string& dataDirectory) {
    // Load concepts from JSON file
    std::string conceptsPath = dataDirectory + "/music_theory/concepts.json";
    std::ifstream conceptsFile(conceptsPath);

    if (!conceptsFile.is_open()) {
        // Fallback: try without music_theory subdirectory
        conceptsPath = dataDirectory + "/concepts.json";
        conceptsFile.open(conceptsPath);
        if (!conceptsFile.is_open()) {
            // If JSON file not found, use basic initialization
            initializeBasicConcepts();
            return false;
        }
    }

    // Simple JSON parsing for concepts.json
    // Note: This is a basic implementation. For production, use a proper JSON library.
    std::string line;
    std::string jsonContent;
    while (std::getline(conceptsFile, line)) {
        jsonContent += line;
    }
    conceptsFile.close();

    // Parse concepts array (simplified - assumes valid JSON structure)
    // Look for concept objects
    size_t pos = jsonContent.find("\"concepts\"");
    if (pos == std::string::npos) {
        initializeBasicConcepts();
        return false;
    }

    // Find opening bracket of concepts array
    pos = jsonContent.find('[', pos);
    if (pos == std::string::npos) {
        initializeBasicConcepts();
        return false;
    }

    // Parse each concept object
    // This is a simplified parser - in production, use a proper JSON library like nlohmann/json
    // For now, we'll parse key fields we need

    // Extract concept blocks
    int braceDepth = 0;
    bool inConcept = false;
    std::string currentConcept;
    size_t startPos = pos + 1;

    for (size_t i = startPos; i < jsonContent.length(); ++i) {
        if (jsonContent[i] == '{') {
            if (braceDepth == 0) {
                inConcept = true;
                currentConcept.clear();
            }
            braceDepth++;
        } else if (jsonContent[i] == '}') {
            braceDepth--;
            if (braceDepth == 0 && inConcept) {
                // Parse this concept
                parseConceptFromJSON(currentConcept);
                inConcept = false;
            }
        }

        if (inConcept) {
            currentConcept += jsonContent[i];
        }

        if (jsonContent[i] == ']' && braceDepth == 0) {
            break;
        }
    }

    // If no concepts were parsed, use basic initialization
    if (concepts_.empty()) {
        initializeBasicConcepts();
    }

    return true;
}

// Helper method to parse a single concept from JSON string
void KnowledgeGraph::parseConceptFromJSON(const std::string& jsonStr) {
    KnowledgeNode node;

    // Extract name
    size_t namePos = jsonStr.find("\"name\"");
    if (namePos != std::string::npos) {
        size_t colonPos = jsonStr.find(':', namePos);
        size_t quoteStart = jsonStr.find('"', colonPos);
        if (quoteStart != std::string::npos) {
            size_t quoteEnd = jsonStr.find('"', quoteStart + 1);
            if (quoteEnd != std::string::npos) {
                node.conceptName = jsonStr.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
            }
        }
    }

    // Extract category
    size_t catPos = jsonStr.find("\"category\"");
    if (catPos != std::string::npos) {
        size_t colonPos = jsonStr.find(':', catPos);
        size_t quoteStart = jsonStr.find('"', colonPos);
        if (quoteStart != std::string::npos) {
            size_t quoteEnd = jsonStr.find('"', quoteStart + 1);
            if (quoteEnd != std::string::npos) {
                node.category = jsonStr.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
            }
        }
    }

    // Extract prerequisites
    size_t prereqPos = jsonStr.find("\"prerequisites\"");
    if (prereqPos != std::string::npos) {
        size_t bracketStart = jsonStr.find('[', prereqPos);
        if (bracketStart != std::string::npos) {
            size_t bracketEnd = jsonStr.find(']', bracketStart);
            if (bracketEnd != std::string::npos) {
                std::string prereqStr = jsonStr.substr(bracketStart + 1, bracketEnd - bracketStart - 1);
                // Parse array of strings
                size_t quotePos = 0;
                while ((quotePos = prereqStr.find('"', quotePos)) != std::string::npos) {
                    size_t quoteEnd = prereqStr.find('"', quotePos + 1);
                    if (quoteEnd != std::string::npos) {
                        std::string prereq = prereqStr.substr(quotePos + 1, quoteEnd - quotePos - 1);
                        node.prerequisites.push_back(prereq);
                        quotePos = quoteEnd + 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // Extract related concepts
    size_t relatedPos = jsonStr.find("\"related_concepts\"");
    if (relatedPos != std::string::npos) {
        size_t bracketStart = jsonStr.find('[', relatedPos);
        if (bracketStart != std::string::npos) {
            size_t bracketEnd = jsonStr.find(']', bracketStart);
            if (bracketEnd != std::string::npos) {
                std::string relatedStr = jsonStr.substr(bracketStart + 1, bracketEnd - bracketStart - 1);
                size_t quotePos = 0;
                while ((quotePos = relatedStr.find('"', quotePos)) != std::string::npos) {
                    size_t quoteEnd = relatedStr.find('"', quotePos + 1);
                    if (quoteEnd != std::string::npos) {
                        std::string related = relatedStr.substr(quotePos + 1, quoteEnd - quotePos - 1);
                        node.relatedConcepts.push_back(related);
                        quotePos = quoteEnd + 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // Extract explanations
    size_t expPos = jsonStr.find("\"explanations\"");
    if (expPos != std::string::npos) {
        // Parse explanation styles
        std::vector<std::string> styles = {"intuitive", "mathematical", "historical", "acoustic"};
        for (const auto& style : styles) {
            size_t stylePos = jsonStr.find("\"" + style + "\"", expPos);
            if (stylePos != std::string::npos) {
                size_t colonPos = jsonStr.find(':', stylePos);
                size_t quoteStart = jsonStr.find('"', colonPos);
                if (quoteStart != std::string::npos) {
                    size_t quoteEnd = jsonStr.find('"', quoteStart + 1);
                    if (quoteEnd != std::string::npos) {
                        std::string explanation = jsonStr.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
                        // Map style string to ExplanationType
                        ExplanationType type = ExplanationType::Intuitive;
                        if (style == "mathematical") type = ExplanationType::Mathematical;
                        else if (style == "historical") type = ExplanationType::Historical;
                        else if (style == "acoustic") type = ExplanationType::Acoustic;

                        // Insert directly into map (explanations is std::map<ExplanationType, std::string>)
                        node.explanations[type] = explanation;
                    }
                }
            }
        }
    }

    // Only add if we have at least a name
    if (!node.conceptName.empty()) {
        addConcept(node);
    }
}

bool KnowledgeGraph::saveToJSON(const std::string& dataDirectory) const {
    // Would write to JSON files
    return true;
}

bool KnowledgeGraph::loadUserProfile(
    const std::string& filePath,
    UserProfile& profile) const
{
    // Would parse JSON user profile
    return true;
}

bool KnowledgeGraph::saveUserProfile(
    const std::string& filePath,
    const UserProfile& profile) const
{
    // Would write JSON user profile
    return true;
}

std::string KnowledgeGraph::exportProgressReport(
    const UserProfile& userProfile) const
{
    std::ostringstream report;

    report << "Music Theory Progress Report\n";
    report << "=============================\n\n";

    report << "Total Exercises Completed: " << userProfile.totalExercisesCompleted << "\n";
    report << "Average Success Rate: " << (userProfile.averageSuccessRate * 100) << "%\n\n";

    report << "Concept Mastery:\n";
    for (const auto& [concept, mastery] : userProfile.conceptMastery) {
        report << "  " << concept << ": " << (mastery * 100) << "%\n";
    }

    report << "\nStruggling Concepts:\n";
    for (const auto& concept : userProfile.strugglingConcepts) {
        report << "  - " << concept << "\n";
    }

    return report.str();
}

//==============================================================================
// Concept Validation
//==============================================================================

std::vector<std::string> KnowledgeGraph::validateGraph() const {
    std::vector<std::string> issues;

    // Check for circular dependencies
    if (hasCircularDependencies()) {
        issues.push_back("Graph contains circular dependencies");
    }

    // Check for orphaned concepts
    auto orphans = findOrphanedConcepts();
    if (!orphans.empty()) {
        issues.push_back("Found " + std::to_string(orphans.size()) + " orphaned concepts");
    }

    // Check for missing prerequisites
    for (const auto& [name, node] : concepts_) {
        for (const auto& prereq : node.prerequisites) {
            if (concepts_.find(prereq) == concepts_.end()) {
                issues.push_back("Concept '" + name + "' has missing prerequisite: " + prereq);
            }
        }
    }

    return issues;
}

bool KnowledgeGraph::hasCircularDependencies() const {
    std::set<std::string> visited;
    std::set<std::string> recursionStack;

    for (const auto& [name, _] : concepts_) {
        if (visited.find(name) == visited.end()) {
            if (hasCycleDFS(name, visited, recursionStack)) {
                return true;
            }
        }
    }

    return false;
}

std::vector<std::string> KnowledgeGraph::findOrphanedConcepts() const {
    std::vector<std::string> orphans;

    for (const auto& [name, node] : concepts_) {
        // Check if concept has no prerequisites and no related concepts
        if (node.prerequisites.empty() && node.relatedConcepts.empty()) {
            // Check if any other concept references this one
            bool isReferenced = false;

            for (const auto& [otherName, otherNode] : concepts_) {
                if (otherName == name) continue;

                auto& prereqs = otherNode.prerequisites;
                auto& related = otherNode.relatedConcepts;

                if (std::find(prereqs.begin(), prereqs.end(), name) != prereqs.end() ||
                    std::find(related.begin(), related.end(), name) != related.end()) {
                    isReferenced = true;
                    break;
                }
            }

            if (!isReferenced) {
                orphans.push_back(name);
            }
        }
    }

    return orphans;
}

//==============================================================================
// Internal Helpers - Graph Algorithms
//==============================================================================

std::vector<std::string> KnowledgeGraph::topologicalSort(
    const std::string& startConcept) const
{
    std::vector<std::string> sorted;
    std::set<std::string> visited;
    std::stack<std::string> stack;

    // DFS from start concept
    std::function<void(const std::string&)> dfs = [&](const std::string& concept) {
        visited.insert(concept);

        auto it = prerequisiteGraph_.find(concept);
        if (it != prerequisiteGraph_.end()) {
            for (const auto& prereq : it->second) {
                if (visited.find(prereq) == visited.end()) {
                    dfs(prereq);
                }
            }
        }

        stack.push(concept);
    };

    dfs(startConcept);

    // Extract sorted order
    while (!stack.empty()) {
        sorted.push_back(stack.top());
        stack.pop();
    }

    return sorted;
}

std::vector<std::string> KnowledgeGraph::breadthFirstSearch(
    const std::string& start,
    const std::string& goal) const
{
    std::queue<std::vector<std::string>> queue;
    std::set<std::string> visited;

    queue.push({start});
    visited.insert(start);

    while (!queue.empty()) {
        auto path = queue.front();
        queue.pop();

        std::string current = path.back();

        if (current == goal) {
            return path;
        }

        // Explore related concepts
        auto it = relatedConceptsGraph_.find(current);
        if (it != relatedConceptsGraph_.end()) {
            for (const auto& neighbor : it->second) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    auto newPath = path;
                    newPath.push_back(neighbor);
                    queue.push(newPath);
                }
            }
        }
    }

    return {}; // No path found
}

std::vector<std::string> KnowledgeGraph::depthFirstSearch(
    const std::string& start,
    std::set<std::string>& visited) const
{
    std::vector<std::string> path;

    visited.insert(start);
    path.push_back(start);

    auto it = prerequisiteGraph_.find(start);
    if (it != prerequisiteGraph_.end()) {
        for (const auto& prereq : it->second) {
            if (visited.find(prereq) == visited.end()) {
                auto subPath = depthFirstSearch(prereq, visited);
                path.insert(path.end(), subPath.begin(), subPath.end());
            }
        }
    }

    return path;
}

bool KnowledgeGraph::hasPath(
    const std::string& from,
    const std::string& to,
    std::set<std::string>& visited) const
{
    if (from == to) return true;

    visited.insert(from);

    auto it = prerequisiteGraph_.find(from);
    if (it != prerequisiteGraph_.end()) {
        for (const auto& prereq : it->second) {
            if (visited.find(prereq) == visited.end()) {
                if (hasPath(prereq, to, visited)) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool KnowledgeGraph::hasCycleDFS(
    const std::string& node,
    std::set<std::string>& visited,
    std::set<std::string>& recursionStack) const
{
    visited.insert(node);
    recursionStack.insert(node);

    auto it = prerequisiteGraph_.find(node);
    if (it != prerequisiteGraph_.end()) {
        for (const auto& prereq : it->second) {
            if (visited.find(prereq) == visited.end()) {
                if (hasCycleDFS(prereq, visited, recursionStack)) {
                    return true;
                }
            } else if (recursionStack.find(prereq) != recursionStack.end()) {
                return true; // Cycle detected
            }
        }
    }

    recursionStack.erase(node);
    return false;
}

std::vector<std::string> KnowledgeGraph::findShortestPath(
    const std::string& from,
    const std::string& to) const
{
    return breadthFirstSearch(from, to);
}

//==============================================================================
// Internal Helpers - Calculations
//==============================================================================

float KnowledgeGraph::calculateRelationshipStrength(
    const std::string& concept1,
    const std::string& concept2) const
{
    // Simple heuristic: check shared prerequisites
    auto prereqs1 = getPrerequisites(concept1);
    auto prereqs2 = getPrerequisites(concept2);

    int sharedCount = 0;
    for (const auto& p1 : prereqs1) {
        if (std::find(prereqs2.begin(), prereqs2.end(), p1) != prereqs2.end()) {
            sharedCount++;
        }
    }

    float maxSize = std::max(prereqs1.size(), prereqs2.size());
    if (maxSize == 0) return 0.3f;

    return 0.3f + (sharedCount / maxSize) * 0.7f;
}

float KnowledgeGraph::calculateConceptSimilarity(
    const std::string& description,
    const KnowledgeNode& concept) const
{
    // Simple keyword matching
    auto descTokens = tokenize(description);
    auto conceptTokens = tokenize(concept.conceptName);

    int matches = 0;
    for (const auto& token1 : descTokens) {
        for (const auto& token2 : conceptTokens) {
            if (fuzzyMatch(token1, token2) > 0.8f) {
                matches++;
            }
        }
    }

    if (descTokens.empty()) return 0.0f;

    return static_cast<float>(matches) / descTokens.size();
}

int KnowledgeGraph::estimateLearningTime(const std::string& conceptName) const {
    auto it = metadata_.find(conceptName);
    if (it != metadata_.end()) {
        return it->second.avgTimeToMaster;
    }

    // Default estimate
    return 30; // 30 minutes
}

std::vector<std::string> KnowledgeGraph::optimizePathOrder(
    const std::vector<std::string>& concepts) const
{
    // Topological sort to respect dependencies
    std::vector<std::string> ordered;
    std::set<std::string> processed;

    for (const auto& concept : concepts) {
        if (processed.find(concept) != processed.end()) continue;

        // Get all dependencies
        auto deps = findDependencies(concept);

        for (const auto& dep : deps) {
            if (processed.find(dep) == processed.end()) {
                ordered.push_back(dep);
                processed.insert(dep);
            }
        }
    }

    return ordered;
}

float KnowledgeGraph::calculateMasteryDecay(
    float currentMastery,
    float daysSinceLastPractice) const
{
    // Exponential decay
    float decayRate = 0.05f;
    return currentMastery * std::exp(-decayRate * daysSinceLastPractice);
}

float KnowledgeGraph::calculateNewMastery(
    float currentMastery,
    bool correct,
    float responseTime,
    int attemptCount) const
{
    float delta = correct ? 0.1f : -0.05f;

    // Bonus for quick responses
    if (correct && responseTime < 5.0f) {
        delta += 0.05f;
    }

    float newMastery = currentMastery + delta;
    return std::clamp(newMastery, 0.0f, 1.0f);
}

//==============================================================================
// Exercise Generators
//==============================================================================

Exercise KnowledgeGraph::generateIntervalExercise(DifficultyLevel level) const {
    Exercise exercise;
    exercise.type = ExerciseType::IntervalRecognition;
    exercise.level = level;
    exercise.concept = "Interval";
    exercise.instruction = "Identify the interval between these two notes";

    // Generate notes (simplified)
    exercise.notes = {60, 67}; // C to G (Perfect Fifth)
    exercise.correctAnswer = "Perfect Fifth";
    exercise.hints.push_back("Count the semitones");

    return exercise;
}

Exercise KnowledgeGraph::generateChordExercise(DifficultyLevel level) const {
    Exercise exercise;
    exercise.type = ExerciseType::ChordQualityRecognition;
    exercise.level = level;
    exercise.concept = "Chord";
    exercise.instruction = "Identify the chord quality";

    exercise.notes = {60, 64, 67}; // C Major triad
    exercise.correctAnswer = "Major";
    exercise.hints.push_back("Listen to the third");

    return exercise;
}

Exercise KnowledgeGraph::generateProgressionExercise(DifficultyLevel level) const {
    Exercise exercise;
    exercise.type = ExerciseType::ProgressionAnalysis;
    exercise.level = level;
    exercise.concept = "Chord Progression";
    exercise.instruction = "Analyze this chord progression";

    exercise.correctAnswer = "I-V-vi-IV";
    exercise.hints.push_back("Identify scale degrees");

    return exercise;
}

Exercise KnowledgeGraph::generateRhythmExercise(DifficultyLevel level) const {
    Exercise exercise;
    exercise.type = ExerciseType::RhythmDictation;
    exercise.level = level;
    exercise.concept = "Rhythm";
    exercise.instruction = "Notate this rhythm";

    exercise.onsets = {0.0f, 0.5f, 1.0f, 1.5f};
    exercise.correctAnswer = "Four eighth notes";
    exercise.hints.push_back("Count the subdivisions");

    return exercise;
}

//==============================================================================
// String Helpers
//==============================================================================

std::vector<std::string> KnowledgeGraph::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string token;

    while (stream >> token) {
        // Convert to lowercase
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        tokens.push_back(token);
    }

    return tokens;
}

float KnowledgeGraph::fuzzyMatch(const std::string& str1, const std::string& str2) const {
    if (str1 == str2) return 1.0f;

    // Simple Levenshtein-inspired similarity
    int maxLen = std::max(str1.length(), str2.length());
    if (maxLen == 0) return 1.0f;

    int distance = 0;
    for (size_t i = 0; i < std::min(str1.length(), str2.length()); ++i) {
        if (str1[i] != str2[i]) distance++;
    }
    distance += std::abs(static_cast<int>(str1.length() - str2.length()));

    return 1.0f - (static_cast<float>(distance) / maxLen);
}

//==============================================================================
// Initialization (would be from JSON in production)
//==============================================================================

void KnowledgeGraph::initializeBasicConcepts() {
    // Sample concepts - in production, these would be loaded from JSON

    KnowledgeNode interval;
    interval.concept = "Interval";
    interval.category = "Core Theory";
    interval.prerequisites = {};
    interval.relatedConcepts = {"Scale", "Chord"};
    interval.applications = {"Melody writing", "Harmony analysis"};
    interval.explanations[ExplanationType::Intuitive] =
        "The distance between two notes, like steps on a ladder";
    interval.explanations[ExplanationType::Acoustic] =
        "The frequency ratio between two pitches";

    addConcept(interval);

    KnowledgeNode scale;
    scale.concept = "Scale";
    scale.category = "Core Theory";
    scale.prerequisites = {"Interval"};
    scale.relatedConcepts = {"Key", "Mode"};
    scale.applications = {"Improvisation", "Composition"};
    scale.explanations[ExplanationType::Intuitive] =
        "A collection of notes that sound good together";

    addConcept(scale);

    KnowledgeNode chord;
    chord.concept = "Chord";
    chord.category = "Harmony";
    chord.prerequisites = {"Interval", "Scale"};
    chord.relatedConcepts = {"Triad", "Seventh Chord"};
    chord.applications = {"Accompaniment", "Songwriting"};

    addConcept(chord);

    // Add metadata
    metadata_["Interval"] = {20, 0.3f, {ExerciseType::IntervalRecognition}};
    metadata_["Scale"] = {30, 0.5f, {ExerciseType::IntervalRecognition}};
    metadata_["Chord"] = {40, 0.6f, {ExerciseType::ChordQualityRecognition}};
}

} // namespace midikompanion::theory
