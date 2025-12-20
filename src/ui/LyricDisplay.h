#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "../voice/LyricTypes.h"
#include "../voice/LyriSync.h"
#include <vector>
#include <memory>

namespace kelly {

/**
 * LyricDisplay - Displays lyric lines with syllable breakdown and playback synchronization.
 *
 * Features:
 * - Display lyric lines
 * - Show syllable breakdown
 * - Highlight current line during playback
 * - Sync with audio playback timing
 */
class LyricDisplay : public juce::Component,
                     public juce::Timer {
public:
    LyricDisplay();
    ~LyricDisplay() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;

    /**
     * Set lyrics to display.
     * @param lyrics Lyric structure to display
     */
    void setLyrics(const LyricStructure& lyrics);

    /**
     * Set synchronization data for playback highlighting.
     * @param syncResult Synchronization result from LyriSync
     */
    void setSyncResult(const LyriSync::SyncResult& syncResult);

    /**
     * Update current playback position.
     * @param currentBeat Current beat position
     */
    void setCurrentBeat(double currentBeat);

    /**
     * Clear displayed lyrics.
     */
    void clear();

    /**
     * Set font size for lyric text.
     */
    void setFontSize(float fontSize) { fontSize_ = fontSize; }

    /**
     * Enable/disable syllable highlighting.
     */
    void setHighlightSyllables(bool highlight) { highlightSyllables_ = highlight; }

    /**
     * Set title for accessibility.
     */
    void setTitle(const juce::String& title) { setComponentID(title); }

    /**
     * Set description for accessibility.
     */
    void setDescription(const juce::String& description) {
        // Note: setTooltip() may not be available in all JUCE versions
        // Using setComponentID for now as alternative
        if (description.isNotEmpty()) {
            setComponentID(description);
        }
    }

private:
    LyricStructure lyrics_;
    LyriSync::SyncResult syncResult_;
    double currentBeat_ = 0.0;
    float fontSize_ = 16.0f;
    bool highlightSyllables_ = true;

    // Colors
    juce::Colour normalColor_ = juce::Colours::white;
    juce::Colour highlightedColor_ = juce::Colour(0xff4a9eff);  // Bright blue
    juce::Colour syllableColor_ = juce::Colour(0xffaaaaaa);     // Light gray

    /**
     * Paint a single lyric line.
     */
    void paintLine(juce::Graphics& g, const LyricLine& line, int lineIndex,
                   const juce::Rectangle<int>& bounds, bool isHighlighted);

    /**
     * Paint syllable breakdown.
     */
    void paintSyllables(juce::Graphics& g, const LyricLine& line,
                       const juce::Rectangle<int>& bounds, int highlightedSyllableIndex);

    /**
     * Get current highlighted line index.
     */
    int getCurrentLineIndex() const;

    /**
     * Get current highlighted syllable index for a line.
     */
    int getCurrentSyllableIndex(const LyricLine& line) const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LyricDisplay)
};

} // namespace kelly
