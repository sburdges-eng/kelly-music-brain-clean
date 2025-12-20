#include "WorkstationPanel.h"
#include "KellyLookAndFeel.h"
#include <algorithm>
#include <cmath>

namespace kelly {

WorkstationPanel::WorkstationPanel()
    : viewport_("WorkstationViewport")
{
    setOpaque(true);
    
    // Setup viewport
    viewport_.setViewedComponent(&trackList_, false);
    viewport_.setScrollBarsShown(true, false);
    addAndMakeVisible(viewport_);
    
    // Initialize tracks
    initializeTracks();
    
    // Start timer for updates
    startTimer(30);  // ~30fps
}

WorkstationPanel::~WorkstationPanel() = default;

void WorkstationPanel::initializeTracks() {
    tracks_.clear();
    trackControls_.clear();
    
    // Define track structure
    struct TrackDef {
        std::string name;
        std::string icon;
        int defaultChannel;
        int defaultInstrument;
        juce::Colour colour;
    };
    
    std::vector<TrackDef> trackDefs = {
        {"Chords", "🎹", 0, 0, juce::Colour(0xff4a90e2)},      // Blue
        {"Melody", "🎵", 1, 0, juce::Colour(0xff50c878)},     // Green
        {"Bass", "🎸", 2, 32, juce::Colour(0xffe74c3c)},      // Red
        {"Counter", "🎺", 3, 56, juce::Colour(0xffffa500)},  // Orange
        {"Pad", "☁️", 4, 88, juce::Colour(0xff9b59b6)},       // Purple
        {"Strings", "🎻", 5, 48, juce::Colour(0xff1abc9c)},  // Teal
        {"Fills", "🥁", 9, 0, juce::Colour(0xff34495e)}       // Dark gray
    };
    
    for (size_t i = 0; i < trackDefs.size(); ++i) {
        TrackInfo track;
        track.name = trackDefs[i].name;
        track.icon = trackDefs[i].icon;
        track.channel = trackDefs[i].defaultChannel;
        track.instrument = trackDefs[i].defaultInstrument;
        track.trackColour = trackDefs[i].colour;
        track.volume = 1.0f;
        track.pan = 0.0f;
        track.muted = false;
        track.soloed = false;
        tracks_.push_back(track);
        
        // Create controls for this track
        auto control = std::make_unique<TrackControl>();
        
        // Mute button
        control->muteButton.setButtonText("M");
        control->muteButton.setTooltip("Mute track");
        control->muteButton.setToggleState(false, juce::dontSendNotification);
        control->muteButton.onClick = [this, i] {
            tracks_[i].muted = trackControls_[i]->muteButton.getToggleState();
            if (onTrackMuteChanged) onTrackMuteChanged(static_cast<int>(i), tracks_[i].muted);
            repaint();
        };
        trackList_.addAndMakeVisible(control->muteButton);
        
        // Solo button
        control->soloButton.setButtonText("S");
        control->soloButton.setTooltip("Solo track");
        control->soloButton.setToggleState(false, juce::dontSendNotification);
        control->soloButton.onClick = [this, i] {
            tracks_[i].soloed = trackControls_[i]->soloButton.getToggleState();
            if (onTrackSoloChanged) onTrackSoloChanged(static_cast<int>(i), tracks_[i].soloed);
            repaint();
        };
        trackList_.addAndMakeVisible(control->soloButton);
        
        // Volume slider
        control->volumeSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        control->volumeSlider.setRange(0.0, 1.0, 0.01);
        control->volumeSlider.setValue(1.0);
        control->volumeSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
        control->volumeSlider.setTooltip("Track volume");
        control->volumeSlider.onValueChange = [this, i] {
            tracks_[i].volume = static_cast<float>(trackControls_[i]->volumeSlider.getValue());
            if (onTrackVolumeChanged) onTrackVolumeChanged(static_cast<int>(i), tracks_[i].volume);
            repaint();
        };
        trackList_.addAndMakeVisible(control->volumeSlider);
        
        // Pan slider
        control->panSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        control->panSlider.setRange(-1.0, 1.0, 0.01);
        control->panSlider.setValue(0.0);
        control->panSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
        control->panSlider.setTooltip("Track pan");
        control->panSlider.onValueChange = [this, i] {
            tracks_[i].pan = static_cast<float>(trackControls_[i]->panSlider.getValue());
            if (onTrackPanChanged) onTrackPanChanged(static_cast<int>(i), tracks_[i].pan);
            repaint();
        };
        trackList_.addAndMakeVisible(control->panSlider);
        
        // Name label
        control->nameLabel.setText(track.name, juce::dontSendNotification);
        control->nameLabel.setFont(juce::FontOptions(14.0f).withStyle("Bold"));
        control->nameLabel.setColour(juce::Label::textColourId, juce::Colours::white);
        trackList_.addAndMakeVisible(control->nameLabel);
        
        // Instrument label
        control->instrumentLabel.setText("Ch " + juce::String(track.channel + 1), juce::dontSendNotification);
        control->instrumentLabel.setFont(juce::FontOptions(11.0f));
        control->instrumentLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
        trackList_.addAndMakeVisible(control->instrumentLabel);
        
        trackControls_.push_back(std::move(control));
    }
    
    updateTrackControls();
}

void WorkstationPanel::updateTracks(const GeneratedMidi& midi) {
    if (tracks_.size() < 7) return;
    
    // Update track notes
    tracks_[0].notes.clear();  // Chords - convert from Chord to MidiNote
    for (const auto& chord : midi.chords) {
        for (const auto& pitch : chord.pitches) {
            MidiNote note;
            note.pitch = pitch;
            note.velocity = 100;  // Default velocity (Chord struct doesn't have velocity field)
            note.startBeat = chord.startBeat;
            note.duration = chord.duration;
            tracks_[0].notes.push_back(note);
        }
    }
    
    tracks_[1].notes = midi.melody;        // Melody
    tracks_[2].notes = midi.bass;         // Bass
    tracks_[3].notes = midi.counterMelody; // Counter-melody
    tracks_[4].notes = midi.pad;          // Pad
    tracks_[5].notes = midi.strings;       // Strings
    tracks_[6].notes = midi.fills;        // Fills
    
    // Update track list size
    trackList_.setSize(getWidth(), static_cast<int>(tracks_.size()) * TRACK_HEIGHT);
    
    repaint();
}

void WorkstationPanel::paint(juce::Graphics& g) {
    // Use documented background colors
    g.fillAll(KellyLookAndFeel::backgroundDark);  // #121212 - Optimal dark mode background
    
    // Draw header with gradient
    auto headerBounds = getLocalBounds().removeFromTop(40);
    juce::ColourGradient headerGradient(
        KellyLookAndFeel::backgroundLight,
        static_cast<float>(headerBounds.getX()), static_cast<float>(headerBounds.getY()),
        KellyLookAndFeel::surfaceColor,
        static_cast<float>(headerBounds.getX()), static_cast<float>(headerBounds.getBottom()),
        false
    );
    g.setGradientFill(headerGradient);
    g.fillRect(headerBounds);
    
    // Header text with modern styling
    g.setColour(KellyLookAndFeel::textPrimary);  // #E8E8E8 - Off-white
    g.setFont(juce::FontOptions(18.0f).withStyle("Bold"));
    g.drawText("MIDI Workstation", headerBounds.removeFromLeft(200), juce::Justification::centredLeft);
    
    // Draw track count with secondary text color
    g.setFont(juce::FontOptions(12.0f));
    g.setColour(KellyLookAndFeel::textSecondary);  // #B8B8B8
    g.drawText(juce::String(tracks_.size()) + " Tracks", headerBounds.removeFromRight(100), juce::Justification::centredRight);
}

void WorkstationPanel::resized() {
    auto bounds = getLocalBounds();
    
    // Header
    auto headerBounds = bounds.removeFromTop(40);
    
    // Viewport takes remaining space
    viewport_.setBounds(bounds);
    
    // Update track list size
    trackList_.setSize(getWidth(), static_cast<int>(tracks_.size()) * TRACK_HEIGHT);
    
    // Layout track controls
    for (size_t i = 0; i < tracks_.size(); ++i) {
        auto trackBounds = juce::Rectangle<int>(0, static_cast<int>(i) * TRACK_HEIGHT, getWidth(), TRACK_HEIGHT);
        
        if (i < trackControls_.size() && trackControls_[i]) {
            auto& ctrl = *trackControls_[i];
            
            // Track header (left side)
            auto headerBounds = trackBounds.removeFromLeft(TRACK_HEADER_WIDTH);
            auto nameBounds = headerBounds.removeFromTop(TRACK_HEIGHT / 2);
            auto instrumentBounds = headerBounds;
            
            ctrl.nameLabel.setBounds(nameBounds.removeFromLeft(120).reduced(5));
            ctrl.muteButton.setBounds(nameBounds.removeFromLeft(30).reduced(2));
            ctrl.soloButton.setBounds(nameBounds.removeFromLeft(30).reduced(2));
            ctrl.instrumentLabel.setBounds(instrumentBounds.reduced(5, 0));
            
            // Track content area (right side)
            auto contentBounds = trackBounds;
            auto volumeBounds = contentBounds.removeFromLeft(100);
            auto panBounds = contentBounds.removeFromLeft(100);
            
            ctrl.volumeSlider.setBounds(volumeBounds.reduced(5));
            ctrl.panSlider.setBounds(panBounds.reduced(5));
        }
    }
}

void WorkstationPanel::timerCallback() {
    // Update visualizations if needed
    repaint();
}

void WorkstationPanel::paintTrack(juce::Graphics& g, const TrackInfo& track, int index, const juce::Rectangle<int>& bounds) {
    // Track background
    bool isSelected = (index == selectedTrackIndex_);
    bool isHovered = (index == hoveredTrackIndex_);
    
    // Use documented surface colors
    juce::Colour bgColour = KellyLookAndFeel::surfaceColor;  // #2A2A2A
    if (isSelected) {
        bgColour = track.trackColour.withAlpha(0.3f);
    } else if (isHovered) {
        bgColour = KellyLookAndFeel::backgroundLight;  // #1E1E1E
    }
    
    if (track.muted) {
        bgColour = bgColour.withAlpha(0.5f);
    }
    
    g.setColour(bgColour);
    g.fillRect(bounds);
    
    // Track border with documented border color
    g.setColour(isSelected ? track.trackColour : KellyLookAndFeel::borderColor);  // #404040
    g.drawRect(bounds, 1);
    
    // Draw track content
    auto contentBounds = bounds.withTrimmedLeft(TRACK_HEADER_WIDTH);
    paintTrackContent(g, track, contentBounds);
}

void WorkstationPanel::paintTrackHeader(juce::Graphics& g, const TrackInfo& track, int index, const juce::Rectangle<int>& bounds) {
    // Icon and name with documented text colors
    auto workingBounds = bounds;
    g.setColour(track.trackColour);
    g.setFont(juce::FontOptions(20.0f));
    g.drawText(track.icon, workingBounds.removeFromLeft(30), juce::Justification::centred);
    
    g.setColour(KellyLookAndFeel::textPrimary);  // #E8E8E8 - Off-white
    g.setFont(juce::FontOptions(14.0f).withStyle("Bold"));
    g.drawText(track.name, workingBounds.removeFromLeft(100), juce::Justification::centredLeft);
}

void WorkstationPanel::paintTrackContent(juce::Graphics& g, const TrackInfo& track, const juce::Rectangle<int>& bounds) {
    if (track.notes.empty()) {
        // Draw empty state
        g.setColour(juce::Colours::darkgrey);
        g.setFont(juce::FontOptions(12.0f));
        g.drawText("No notes", bounds, juce::Justification::centred);
        return;
    }
    
    // Find time range
    double minTime = std::numeric_limits<double>::max();
    double maxTime = 0.0;
    int minPitch = 127;
    int maxPitch = 0;
    
    for (const auto& note : track.notes) {
        minTime = std::min(minTime, note.startBeat);
        maxTime = std::max(maxTime, note.startBeat + note.duration);
        minPitch = std::min(minPitch, note.pitch);
        maxPitch = std::max(maxPitch, note.pitch);
    }
    
    if (maxTime <= minTime) return;
    
    double timeRange = maxTime - minTime;
    int pitchRange = maxPitch - minPitch;
    if (pitchRange == 0) pitchRange = 12;  // At least one octave
    
    // Draw piano roll style
    g.setColour(track.trackColour.withAlpha(0.2f));
    
    for (const auto& note : track.notes) {
        double x = bounds.getX() + (note.startBeat - minTime) / timeRange * bounds.getWidth();
        double width = note.duration / timeRange * bounds.getWidth();
        double y = bounds.getBottom() - (note.pitch - minPitch) / static_cast<double>(pitchRange) * bounds.getHeight();
        double height = bounds.getHeight() / static_cast<double>(pitchRange);
        
        // Draw note rectangle
        juce::Colour noteColour = track.trackColour;
        if (track.muted) {
            noteColour = noteColour.withAlpha(0.4f);
        }
        
        g.setColour(noteColour);
        g.fillRect(static_cast<float>(x), static_cast<float>(y - height), 
                   static_cast<float>(width), static_cast<float>(height));
        
        // Draw note border
        g.setColour(noteColour.brighter(0.3f));
        g.drawRect(static_cast<float>(x), static_cast<float>(y - height), 
                   static_cast<float>(width), static_cast<float>(height), 1.0f);
    }
}

void WorkstationPanel::mouseDown(const juce::MouseEvent& e) {
    int trackIndex = getTrackIndexFromPosition(e.getPosition());
    if (trackIndex >= 0 && trackIndex < static_cast<int>(tracks_.size())) {
        selectedTrackIndex_ = trackIndex;
        if (onTrackSelected) onTrackSelected(trackIndex);
        repaint();
    }
}

void WorkstationPanel::mouseMove(const juce::MouseEvent& e) {
    int trackIndex = getTrackIndexFromPosition(e.getPosition());
    if (trackIndex != hoveredTrackIndex_) {
        hoveredTrackIndex_ = trackIndex;
        repaint();
    }
}

void WorkstationPanel::mouseExit(const juce::MouseEvent& e) {
    hoveredTrackIndex_ = -1;
    repaint();
}

void WorkstationPanel::updateTrackControls() {
    // Controls are already created in initializeTracks
    // This can be used for dynamic updates if needed
}

int WorkstationPanel::getTrackIndexFromPosition(const juce::Point<int>& pos) const {
    int trackIndex = pos.y / TRACK_HEIGHT;
    if (trackIndex >= 0 && trackIndex < static_cast<int>(tracks_.size())) {
        return trackIndex;
    }
    return -1;
}

} // namespace kelly
