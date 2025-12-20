#include "MusicTheoryWorkstation.h"
#include <juce_audio_formats/juce_audio_formats.h>

namespace kelly {

MusicTheoryWorkstation::MusicTheoryWorkstation(midikompanion::theory::MusicTheoryBrain* brain)
    : brain_(brain)
    , conceptBrowser_(brain)
    , learningPanel_(brain)
    , virtualKeyboard_()
{
    setLookAndFeel(&lookAndFeel_);
    setupTabs();
    setupAnalysisPanel();
    setupPracticePanel();
}

void MusicTheoryWorkstation::paint(juce::Graphics& g) {
    g.fillAll(lookAndFeel_.backgroundDark);
}

void MusicTheoryWorkstation::resized() {
    auto bounds = getLocalBounds();
    tabs_.setBounds(bounds);
}

void MusicTheoryWorkstation::setupTabs() {
    tabs_.addTab("Learning", juce::Colours::transparentBlack, &learningPanel_, false);
    tabs_.addTab("Concepts", juce::Colours::transparentBlack, &conceptBrowser_, false);
    tabs_.addTab("Analysis", juce::Colours::transparentBlack, &analysisDisplay_, false);
    tabs_.addTab("Practice", juce::Colours::transparentBlack, &practiceDisplay_, false);
    tabs_.addTab("Keyboard", juce::Colours::transparentBlack, &virtualKeyboard_, false);

    // Connect concept browser to learning panel
    conceptBrowser_.onConceptSelected = [this](const std::string& conceptName) {
        showConcept(conceptName);
        tabs_.setCurrentTabIndex(0); // Switch to Learning tab
    };

    addAndMakeVisible(tabs_);
}

void MusicTheoryWorkstation::setupAnalysisPanel() {
    analysisDisplay_.setMultiLine(true);
    analysisDisplay_.setReadOnly(true);
    analysisDisplay_.setFont(juce::Font(14.0f));
    analysisDisplay_.setText("Load a MIDI file to see analysis results here.");
}

void MusicTheoryWorkstation::setupPracticePanel() {
    practiceDisplay_.setMultiLine(true);
    practiceDisplay_.setReadOnly(true);
    practiceDisplay_.setFont(juce::Font(14.0f));
    practiceDisplay_.setText("Start a practice session to see exercises here.");
}

void MusicTheoryWorkstation::analyzeMIDI(const juce::MidiFile& midi) {
    if (!brain_) {
        analysisDisplay_.setText("Error: MusicTheoryBrain not initialized.");
        return;
    }

    // Extract MIDI data
    std::vector<int> midiNotes;
    std::vector<float> onsetTimes;
    std::vector<int> velocities;

    // Convert JUCE MidiFile to vectors
    for (int track = 0; track < midi.getNumTracks(); ++track) {
        const auto* sequence = midi.getTrack(track);
        if (sequence == nullptr) continue;

        for (const auto* event : *sequence) {
            if (event->message.isNoteOn()) {
                midiNotes.push_back(event->message.getNoteNumber());
                onsetTimes.push_back(event->message.getTimeStamp() / 1000.0f); // Convert to seconds
                velocities.push_back(event->message.getVelocity());
            }
        }
    }

    // Perform analysis
    auto analysis = brain_->analyzeMIDI(midiNotes, onsetTimes, velocities);

    // Format results
    juce::String result;
    result += "=== MIDI Analysis ===\n\n";
    result += "Key: " + juce::String(analysis.detectedKey) + "\n";
    result += "Scale: " + juce::String(analysis.detectedScale.name) + "\n\n";

    result += "Chords Detected: " + juce::String(analysis.chords.size()) + "\n";
    for (size_t i = 0; i < analysis.chords.size() && i < 10; ++i) {
        result += "  - " + juce::String(analysis.chords[i].symbol) + "\n";
    }
    result += "\n";

    result += "Time Signature: " + juce::String(analysis.timeSignature.numerator) + "/" +
              juce::String(analysis.timeSignature.denominator) + "\n\n";

    result += "Detected Concepts:\n";
    for (const auto& detectedConcept : analysis.concepts) {
        result += "  - " + juce::String(detectedConcept.conceptName) + "\n";
    }
    result += "\n";

    result += "Explanation:\n" + juce::String(analysis.overallExplanation) + "\n";

    analysisDisplay_.setText(result);
    tabs_.setCurrentTabIndex(2); // Switch to Analysis tab

    if (onMIDIAnalyzed) {
        onMIDIAnalyzed(midi);
    }
}

void MusicTheoryWorkstation::showConcept(const std::string& conceptName) {
    if (!brain_) {
        return;
    }

    learningPanel_.displayConcept(conceptName);
    tabs_.setCurrentTabIndex(0); // Switch to Learning tab

    if (onConceptSelected) {
        onConceptSelected(conceptName);
    }
}

void MusicTheoryWorkstation::startPracticeSession() {
    if (!brain_) {
        practiceDisplay_.setText("Error: MusicTheoryBrain not initialized.");
        return;
    }

    // Create default user profile
    midikompanion::theory::UserProfile profile;
    profile.name = "User";
    profile.currentLevel = midikompanion::theory::DifficultyLevel::Beginner;

    // Generate practice session
    auto session = brain_->generatePracticeSession(profile, 30);

    // Format results
    juce::String result;
    result += "=== Practice Session ===\n\n";
    result += "Goal: " + juce::String(session.sessionGoal) + "\n";
    result += "Duration: " + juce::String(session.estimatedDuration) + " minutes\n\n";

    result += "Focus Areas:\n";
    for (const auto& area : session.focusAreas) {
        result += "  - " + juce::String(area) + "\n";
    }
    result += "\n";

    result += "Exercises:\n";
    for (size_t i = 0; i < session.exercises.size(); ++i) {
        const auto& ex = session.exercises[i];
        result += juce::String(i + 1) + ". " + juce::String(ex.conceptName) + "\n";
        result += "   " + juce::String(ex.instruction) + "\n\n";
    }

    practiceDisplay_.setText(result);
    tabs_.setCurrentTabIndex(3); // Switch to Practice tab
}

void MusicTheoryWorkstation::displayExplanation(const std::string& text, midikompanion::theory::ExplanationType style) {
    learningPanel_.displayExplanation(text, style);
    tabs_.setCurrentTabIndex(0); // Switch to Learning tab
}

void MusicTheoryWorkstation::setMusicTheoryBrain(midikompanion::theory::MusicTheoryBrain* brain) {
    brain_ = brain;
    conceptBrowser_.setMusicTheoryBrain(brain);
    learningPanel_.setMusicTheoryBrain(brain);
}

} // namespace kelly
