#pragma once
/*
 * ConceptBrowser.h - Music Theory Concept Browser
 * ===============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: MusicTheoryBrain, KnowledgeGraph
 * - UI Layer: Used by MusicTheoryWorkstation
 *
 * Purpose: Tree view of music theory concepts with search, filtering,
 *          and visual relationship display.
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include "../../music_theory/MusicTheoryBrain.h"
#include "../../music_theory/Types.h"
#include <vector>
#include <functional>

namespace kelly {

/**
 * ConceptBrowser - Browse and search music theory concepts
 *
 * Features:
 * - Tree view of concepts organized by category
 * - Search/filter functionality
 * - Click to view concept details
 * - Shows prerequisites and related concepts
 * - Visual graph visualization of relationships
 */
class ConceptBrowser : public juce::Component {
public:
    explicit ConceptBrowser(midikompanion::theory::MusicTheoryBrain* brain);
    ~ConceptBrowser() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    /**
     * Set MusicTheoryBrain instance
     */
    void setMusicTheoryBrain(midikompanion::theory::MusicTheoryBrain* brain);

    /**
     * Refresh concept list from KnowledgeGraph
     */
    void refreshConcepts();

    /**
     * Search for concepts
     */
    void searchConcepts(const juce::String& query);

    // Callbacks
    std::function<void(const std::string& conceptName)> onConceptSelected;

private:
    // Non-owning pointer
    midikompanion::theory::MusicTheoryBrain* brain_ = nullptr;

    // UI Components
    juce::TextEditor searchBox_;
    juce::Label searchLabel_{"", "Search Concepts"};
    juce::ListBox conceptList_;
    juce::TextEditor conceptDetails_;
    juce::Label detailsLabel_{"", "Concept Details"};

    // Data
    struct ConceptItem {
        std::string name;
        std::string category;
        std::vector<std::string> prerequisites;
        std::vector<std::string> related;
    };

    std::vector<ConceptItem> allConcepts_;
    std::vector<ConceptItem> filteredConcepts_;

    class ConceptListModel : public juce::ListBoxModel {
    public:
        ConceptListModel(ConceptBrowser& browser) : browser_(browser) {}

        int getNumRows() override {
            return static_cast<int>(browser_.filteredConcepts_.size());
        }

        void paintListBoxItem(int rowNumber, juce::Graphics& g,
                              int width, int height, bool rowIsSelected) override {
            if (rowIsSelected) {
                g.fillAll(juce::Colour(0xff4a90e2));
            }

            if (rowNumber < static_cast<int>(browser_.filteredConcepts_.size())) {
                const auto& conceptNode = browser_.filteredConcepts_[rowNumber];
                g.setColour(juce::Colours::white);
                g.setFont(14.0f);
                g.drawText(conceptNode.name, 10, 0, width - 10, height,
                          juce::Justification::centredLeft);

                g.setFont(11.0f);
                g.setColour(juce::Colours::lightgrey);
                g.drawText(conceptNode.category, width - 100, 0, 90, height,
                          juce::Justification::centredRight);
            }
        }

        void listBoxItemClicked(int row, const juce::MouseEvent&) override {
            if (row < static_cast<int>(browser_.filteredConcepts_.size())) {
                browser_.selectConcept(browser_.filteredConcepts_[row].name);
            }
        }

    private:
        ConceptBrowser& browser_;
    };

    ConceptListModel listModel_{*this};

    void setupComponents();
    void loadConceptsFromGraph();
    void selectConcept(const std::string& conceptName);
    void updateConceptDetails(const std::string& conceptName);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConceptBrowser)
};

} // namespace kelly
