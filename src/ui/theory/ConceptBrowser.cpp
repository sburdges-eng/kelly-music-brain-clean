#include "ConceptBrowser.h"

namespace kelly {

ConceptBrowser::ConceptBrowser(midikompanion::theory::MusicTheoryBrain* brain)
    : brain_(brain)
{
    setupComponents();
    refreshConcepts();
}

void ConceptBrowser::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff2a2a2a));
}

void ConceptBrowser::resized() {
    auto bounds = getLocalBounds();
    const int margin = 10;
    const int searchHeight = 30;
    const int labelHeight = 20;
    const int detailsHeight = 200;

    // Search box at top
    searchLabel_.setBounds(margin, margin, bounds.getWidth() - 2 * margin, labelHeight);
    searchBox_.setBounds(margin, margin + labelHeight, bounds.getWidth() - 2 * margin, searchHeight);

    // Concept list in middle
    int listTop = margin + labelHeight + searchHeight + margin;
    int listHeight = bounds.getHeight() - listTop - detailsHeight - labelHeight - 2 * margin;
    conceptList_.setBounds(margin, listTop, bounds.getWidth() - 2 * margin, listHeight);

    // Details at bottom
    int detailsTop = listTop + listHeight + margin;
    detailsLabel_.setBounds(margin, detailsTop, bounds.getWidth() - 2 * margin, labelHeight);
    conceptDetails_.setBounds(margin, detailsTop + labelHeight,
                             bounds.getWidth() - 2 * margin, detailsHeight);
}

void ConceptBrowser::setupComponents() {
    // Search box
    searchBox_.setTextToShowWhenEmpty("Search concepts...", juce::Colours::grey);
    searchBox_.onTextChange = [this] {
        searchConcepts(searchBox_.getText());
    };
    addAndMakeVisible(searchBox_);
    addAndMakeVisible(searchLabel_);

    // Concept list
    conceptList_.setModel(&listModel_);
    addAndMakeVisible(conceptList_);

    // Details display
    conceptDetails_.setMultiLine(true);
    conceptDetails_.setReadOnly(true);
    conceptDetails_.setFont(juce::Font(12.0f));
    addAndMakeVisible(conceptDetails_);
    addAndMakeVisible(detailsLabel_);
}

void ConceptBrowser::setMusicTheoryBrain(midikompanion::theory::MusicTheoryBrain* brain) {
    brain_ = brain;
    refreshConcepts();
}

void ConceptBrowser::refreshConcepts() {
    loadConceptsFromGraph();
    filteredConcepts_ = allConcepts_;
    conceptList_.updateContent();
}

void ConceptBrowser::loadConceptsFromGraph() {
    allConcepts_.clear();

    if (!brain_) {
        return;
    }

    // Get concepts by category
    const auto& knowledge = brain_->getKnowledge();

    // Common categories
    std::vector<std::string> categories = {
        "Interval", "Scale", "Chord", "Progression", "Rhythm", "Voice Leading"
    };

    for (const auto& category : categories) {
        auto concepts = knowledge.getConceptsByCategory(category);
        for (const auto& conceptNode : concepts) {
            ConceptItem item;
            item.name = conceptNode.conceptName;  // KnowledgeNode uses 'conceptName' field
            item.category = category;
            item.prerequisites = knowledge.getPrerequisites(conceptNode.conceptName);

            // getRelatedConcepts returns ConceptRelationship structs, extract concept names
            auto relatedRelationships = knowledge.getRelatedConcepts(conceptNode.conceptName);
            for (const auto& rel : relatedRelationships) {
                item.related.push_back(rel.conceptName);
            }

            allConcepts_.push_back(item);
        }
    }

    // If no concepts found, add some defaults
    if (allConcepts_.empty()) {
        // Add basic concepts as fallback
        ConceptItem interval;
        interval.name = "Perfect Fifth";
        interval.category = "Interval";
        allConcepts_.push_back(interval);

        ConceptItem scale;
        scale.name = "Major Scale";
        scale.category = "Scale";
        scale.prerequisites = {"Interval"};
        allConcepts_.push_back(scale);

        ConceptItem chord;
        chord.name = "Major Triad";
        chord.category = "Chord";
        chord.prerequisites = {"Interval", "Major Scale"};
        allConcepts_.push_back(chord);
    }
}

void ConceptBrowser::searchConcepts(const juce::String& query) {
    filteredConcepts_.clear();

    if (query.isEmpty()) {
        filteredConcepts_ = allConcepts_;
    } else {
        juce::String lowerQuery = query.toLowerCase();
        for (const auto& conceptNode : allConcepts_) {
            juce::String name(conceptNode.name);  // ConceptItem uses 'name' field
            juce::String category(conceptNode.category);
            if (name.toLowerCase().contains(lowerQuery) ||
                category.toLowerCase().contains(lowerQuery)) {
                filteredConcepts_.push_back(conceptNode);
            }
        }
    }

    conceptList_.updateContent();
}

void ConceptBrowser::selectConcept(const std::string& conceptName) {
    updateConceptDetails(conceptName);

    if (onConceptSelected) {
        onConceptSelected(conceptName);
    }
}

void ConceptBrowser::updateConceptDetails(const std::string& conceptName) {
    juce::String details;

    // Find concept
    ConceptItem* found = nullptr;
    for (auto& conceptItem : allConcepts_) {
        if (conceptItem.name == conceptName) {
            found = &conceptItem;
            break;
        }
    }

    if (!found) {
        conceptDetails_.setText("Concept not found.");
        return;
    }

    details += "Name: " + juce::String(found->name) + "\n";
    details += "Category: " + juce::String(found->category) + "\n\n";

    if (!found->prerequisites.empty()) {
        details += "Prerequisites:\n";
        for (const auto& prereq : found->prerequisites) {
            details += "  - " + juce::String(prereq) + "\n";
        }
        details += "\n";
    }

    if (!found->related.empty()) {
        details += "Related Concepts:\n";
        for (const auto& related : found->related) {
            details += "  - " + juce::String(related) + "\n";
        }
    }

    // Try to get explanation from KnowledgeGraph
    if (brain_) {
        const auto& knowledge = brain_->getKnowledge();
        auto conceptNode = knowledge.getConcept(found->name);
        if (conceptNode.has_value()) {
            details += "\nExplanation:\n";
            // explanations is a map, get first available explanation
            if (!conceptNode->explanations.empty()) {
                details += juce::String(conceptNode->explanations.begin()->second);
            }
        }
    }

    conceptDetails_.setText(details);
}

} // namespace kelly
