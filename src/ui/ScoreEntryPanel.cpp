#include "ScoreEntryPanel.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_graphics/juce_graphics.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace midikompanion {

namespace {
float noteValueToBeats(NoteValue value) {
    switch (value) {
        case NoteValue::Whole: return 4.0f;
        case NoteValue::Half: return 2.0f;
        case NoteValue::Quarter: return 1.0f;
        case NoteValue::Eighth: return 0.5f;
        case NoteValue::Sixteenth: return 0.25f;
        case NoteValue::ThirtySecond: return 0.125f;
        case NoteValue::Dotted: return 1.5f;
        default: return 1.0f;
    }
}

int dynamicToVelocity(Dynamic dynamic) {
    switch (dynamic) {
        case Dynamic::Pianissimo: return 40;
        case Dynamic::Piano: return 55;
        case Dynamic::MezzoPiano: return 70;
        case Dynamic::MezzoForte: return 90;
        case Dynamic::Forte: return 110;
        case Dynamic::Fortissimo: return 120;
        default: return 90;
    }
}

int defaultBeatsPerBar(const std::vector<TimeSignatureChange>& timeSigs, int measure) {
    if (timeSigs.empty()) {
        return 4;
    }
    TimeSignatureChange current = timeSigs.front();
    for (const auto& ts : timeSigs) {
        if (ts.measure <= measure) {
            current = ts;
        }
    }
    return std::max(1, current.numerator);
}
} // namespace

ScoreEntryPanel::ScoreEntryPanel()
    : entryMode_(EntryMode::Standard),
      viewMode_(ViewMode::SingleStaff),
      currentClef_(Clef::Treble),
      currentNoteValue_(NoteValue::Quarter),
      currentDynamic_(Dynamic::MezzoForte),
      currentArticulation_(Articulation::None),
      cursorMeasure_(1),
      cursorBeat_(1.0f),
      showChordSymbols_(true),
      showLyrics_(false),
      showDynamics_(true),
      zoomFactor_(1.0f)
{
    theoryBrain_ = std::make_unique<theory::MusicTheoryBrain>();
    initializeTemplates();
    initializeComponents();
}

void ScoreEntryPanel::initializeComponents() {
    entryModeSelector_ = std::make_unique<juce::ComboBox>();
    entryModeSelector_->addItem("Simple", 1);
    entryModeSelector_->addItem("Standard", 2);
    entryModeSelector_->addItem("Professional", 3);
    entryModeSelector_->addItem("Chord", 4);
    entryModeSelector_->setSelectedId(2);
    entryModeSelector_->onChange = [this] {
        const int id = entryModeSelector_->getSelectedId();
        if (id == 1) setEntryMode(EntryMode::Simple);
        else if (id == 2) setEntryMode(EntryMode::Standard);
        else if (id == 3) setEntryMode(EntryMode::Professional);
        else if (id == 4) setEntryMode(EntryMode::Chord);
    };
    addAndMakeVisible(*entryModeSelector_);

    scoreViewport_ = std::make_unique<juce::Viewport>();
    scoreDisplay_ = std::make_unique<juce::Component>();
    scoreDisplay_->setInterceptsMouseClicks(false, false);
    scoreViewport_->setViewedComponent(scoreDisplay_.get(), false);
    addAndMakeVisible(*scoreViewport_);

    auto makeNoteButton = [this](std::unique_ptr<juce::TextButton>& btn, const juce::String& text, NoteValue value) {
        btn = std::make_unique<juce::TextButton>(text);
        btn->onClick = [this, value] { onNoteValueSelected(value); };
        addAndMakeVisible(*btn);
    };

    makeNoteButton(wholeNoteButton_, "1", NoteValue::Whole);
    makeNoteButton(halfNoteButton_, "1/2", NoteValue::Half);
    makeNoteButton(quarterNoteButton_, "1/4", NoteValue::Quarter);
    makeNoteButton(eighthNoteButton_, "1/8", NoteValue::Eighth);
    makeNoteButton(sixteenthNoteButton_, "1/16", NoteValue::Sixteenth);

    dottedButton_ = std::make_unique<juce::TextButton>("Dotted");
    dottedButton_->setClickingTogglesState(true);
    addAndMakeVisible(*dottedButton_);

    tripletButton_ = std::make_unique<juce::TextButton>("Triplet");
    tripletButton_->setClickingTogglesState(true);
    addAndMakeVisible(*tripletButton_);

    auto makeDynamicButton = [this](std::unique_ptr<juce::TextButton>& btn, const juce::String& text, Dynamic dyn) {
        btn = std::make_unique<juce::TextButton>(text);
        btn->onClick = [this, dyn] { onDynamicSelected(dyn); };
        addAndMakeVisible(*btn);
    };
    makeDynamicButton(ppButton_, "pp", Dynamic::Pianissimo);
    makeDynamicButton(pButton_, "p", Dynamic::Piano);
    makeDynamicButton(mpButton_, "mp", Dynamic::MezzoPiano);
    makeDynamicButton(mfButton_, "mf", Dynamic::MezzoForte);
    makeDynamicButton(fButton_, "f", Dynamic::Forte);
    makeDynamicButton(ffButton_, "ff", Dynamic::Fortissimo);

    auto makeArticulationButton = [this](std::unique_ptr<juce::TextButton>& btn, const juce::String& text, Articulation art) {
        btn = std::make_unique<juce::TextButton>(text);
        btn->onClick = [this, art] { onArticulationSelected(art); };
        addAndMakeVisible(*btn);
    };
    makeArticulationButton(staccatoButton_, "Stac", Articulation::Staccato);
    makeArticulationButton(legatoButton_, "Leg", Articulation::Legato);
    makeArticulationButton(accentButton_, "Acc", Articulation::Accent);
    makeArticulationButton(tenutoButton_, "Ten", Articulation::Tenuto);

    quickEntryInput_ = std::make_unique<juce::TextEditor>();
    quickEntryInput_->setText("C major scale, quarter notes");
    addAndMakeVisible(*quickEntryInput_);

    quickEntryButton_ = std::make_unique<juce::TextButton>("Quick Entry");
    quickEntryButton_->onClick = [this] { onQuickEntryExecuted(); };
    addAndMakeVisible(*quickEntryButton_);

    playButton_ = std::make_unique<juce::TextButton>("Play");
    playButton_->onClick = [this] { onPlayClicked(); };
    addAndMakeVisible(*playButton_);

    stopButton_ = std::make_unique<juce::TextButton>("Stop");
    stopButton_->onClick = [this] { onStopClicked(); };
    addAndMakeVisible(*stopButton_);

    metronomeButton_ = std::make_unique<juce::TextButton>("Metronome");
    metronomeButton_->setClickingTogglesState(true);
    metronomeButton_->onClick = [this] { onMetronomeToggled(); };
    addAndMakeVisible(*metronomeButton_);

    timeSignatureLabel_ = std::make_unique<juce::Label>("", "4/4");
    timeSignatureLabel_->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(*timeSignatureLabel_);

    keySignatureLabel_ = std::make_unique<juce::Label>("", "C");
    keySignatureLabel_->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(*keySignatureLabel_);

    tempoLabel_ = std::make_unique<juce::Label>("", "120 BPM");
    tempoLabel_->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(*tempoLabel_);
}

void ScoreEntryPanel::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff1d1d1d));

    auto bounds = getLocalBounds().reduced(10);
    const int controlHeight = 140;
    auto scoreArea = bounds.removeFromBottom(std::max(120, bounds.getHeight() - controlHeight));

    drawStaff(g, scoreArea);

    // Draw existing elements
    for (const auto& ts : timeSignatures_) {
        drawTimeSignature(g, ts.numerator, ts.denominator,
                          {scoreArea.getX() + 8, scoreArea.getY() + 8});
    }

    if (!keySignatures_.empty()) {
        const auto& key = keySignatures_.back();
        drawKeySignature(g, key.key, {scoreArea.getX() + 8, scoreArea.getY() + 40});
    }

    // Draw notes
    for (const auto& note : notes_) {
        int measureWidth = std::max(120, static_cast<int>(scoreArea.getWidth() / std::max(1, getMaxMeasureIndex())));
        int beatsPerBar = defaultBeatsPerBar(timeSignatures_, note.measure);
        float beatOffset = (note.beat - 1.0f) / std::max(1, beatsPerBar);
        int x = scoreArea.getX() + static_cast<int>((note.measure - 1) * measureWidth + beatOffset * measureWidth + 20);
        int y = pitchToStaffPosition(note.pitch, currentClef_);
        drawNote(g, note, {x, y});
    }

    // Draw chord symbols
    if (showChordSymbols_) {
        for (const auto& chord : chordSymbols_) {
            int measureWidth = std::max(120, static_cast<int>(scoreArea.getWidth() / std::max(1, getMaxMeasureIndex())));
            float beatOffset = (chord.beat - 1.0f) / defaultBeatsPerBar(timeSignatures_, chord.measure);
            int x = scoreArea.getX() + static_cast<int>((chord.measure - 1) * measureWidth + beatOffset * measureWidth + 12);
            int y = scoreArea.getY() + 10;
            drawChordSymbol(g, chord, {x, y});
        }
    }

    drawCursor(g);
}

void ScoreEntryPanel::resized() {
    auto bounds = getLocalBounds().reduced(10);
    auto topRow = bounds.removeFromTop(30);

    entryModeSelector_->setBounds(topRow.removeFromLeft(180));
    quickEntryInput_->setBounds(topRow.removeFromLeft(300));
    quickEntryButton_->setBounds(topRow.removeFromLeft(120));
    playButton_->setBounds(topRow.removeFromLeft(80));
    stopButton_->setBounds(topRow.removeFromLeft(80));
    metronomeButton_->setBounds(topRow.removeFromLeft(120));

    auto noteRow = bounds.removeFromTop(30);
    wholeNoteButton_->setBounds(noteRow.removeFromLeft(60));
    halfNoteButton_->setBounds(noteRow.removeFromLeft(60));
    quarterNoteButton_->setBounds(noteRow.removeFromLeft(60));
    eighthNoteButton_->setBounds(noteRow.removeFromLeft(60));
    sixteenthNoteButton_->setBounds(noteRow.removeFromLeft(60));
    dottedButton_->setBounds(noteRow.removeFromLeft(80));
    tripletButton_->setBounds(noteRow.removeFromLeft(80));

    auto dynRow = bounds.removeFromTop(30);
    ppButton_->setBounds(dynRow.removeFromLeft(50));
    pButton_->setBounds(dynRow.removeFromLeft(50));
    mpButton_->setBounds(dynRow.removeFromLeft(50));
    mfButton_->setBounds(dynRow.removeFromLeft(50));
    fButton_->setBounds(dynRow.removeFromLeft(50));
    ffButton_->setBounds(dynRow.removeFromLeft(50));

    auto artRow = bounds.removeFromTop(30);
    staccatoButton_->setBounds(artRow.removeFromLeft(60));
    legatoButton_->setBounds(artRow.removeFromLeft(60));
    accentButton_->setBounds(artRow.removeFromLeft(60));
    tenutoButton_->setBounds(artRow.removeFromLeft(60));

    auto infoRow = bounds.removeFromTop(30);
    timeSignatureLabel_->setBounds(infoRow.removeFromLeft(80));
    keySignatureLabel_->setBounds(infoRow.removeFromLeft(80));
    tempoLabel_->setBounds(infoRow.removeFromLeft(120));

    scoreViewport_->setBounds(bounds);
}

void ScoreEntryPanel::mouseDown(const juce::MouseEvent& event) {
    const auto area = getScoreArea();
    if (!area.contains(event.getPosition())) {
        return;
    }

    const int pitch = staffPositionToPitch(event.y, currentClef_);
    addNote(pitch, currentNoteValue_, dottedButton_ && dottedButton_->getToggleState());
}

void ScoreEntryPanel::mouseDrag(const juce::MouseEvent& event) {
    mouseDown(event);
}

void ScoreEntryPanel::setEntryMode(EntryMode mode) {
    entryMode_ = mode;
}

void ScoreEntryPanel::setTimeSignature(int numerator, int denominator) {
    timeSignatures_.push_back({numerator, denominator, cursorMeasure_});
    timeSignatureLabel_->setText(juce::String(numerator) + "/" + juce::String(denominator), juce::dontSendNotification);
    repaint();
}

void ScoreEntryPanel::setKeySignature(const std::string& key) {
    keySignatures_.push_back({key, cursorMeasure_});
    keySignatureLabel_->setText(juce::String(key), juce::dontSendNotification);
    repaint();
}

void ScoreEntryPanel::setTempo(float bpm, const std::string& description) {
    tempos_.push_back({bpm, description, cursorMeasure_});
    tempoLabel_->setText(juce::String(bpm, 1) + " BPM", juce::dontSendNotification);
}

void ScoreEntryPanel::setClef(Clef clef) {
    currentClef_ = clef;
    repaint();
}

void ScoreEntryPanel::addNote(int pitch, NoteValue duration, bool dotted) {
    NotationNote note;
    note.pitch = juce::jlimit(0, 127, pitch);
    note.duration = duration;
    note.dotted = dotted || (dottedButton_ && dottedButton_->getToggleState());
    note.triplet = (tripletButton_ && tripletButton_->getToggleState());
    note.dynamic = currentDynamic_;
    note.articulation = currentArticulation_;
    note.tie = false;
    note.lyric = "";
    note.measure = cursorMeasure_;
    note.beat = cursorBeat_;

    notes_.push_back(note);

    float beats = noteValueToBeats(duration);
    if (note.dotted) beats *= 1.5f;
    if (note.triplet) beats *= (2.0f / 3.0f);

    cursorBeat_ += beats;
    auto beatsPerBar = defaultBeatsPerBar(timeSignatures_, cursorMeasure_);
    while (cursorBeat_ > beatsPerBar) {
        cursorBeat_ -= beatsPerBar;
        cursorMeasure_ += 1;
    }
    repaint();
}

void ScoreEntryPanel::addChord(const std::vector<int>& pitches, NoteValue duration) {
    const int measure = cursorMeasure_;
    const float beat = cursorBeat_;

    for (int pitch : pitches) {
        NotationNote note;
        note.pitch = juce::jlimit(0, 127, pitch);
        note.duration = duration;
        note.dotted = dottedButton_ && dottedButton_->getToggleState();
        note.triplet = tripletButton_ && tripletButton_->getToggleState();
        note.dynamic = currentDynamic_;
        note.articulation = currentArticulation_;
        note.tie = false;
        note.measure = measure;
        note.beat = beat;
        notes_.push_back(note);
    }

    addRest(duration);
}

void ScoreEntryPanel::addChordSymbol(const std::string& symbol) {
    ChordSymbol chord;
    chord.symbol = symbol;
    chord.measure = cursorMeasure_;
    chord.beat = cursorBeat_;
    chordSymbols_.push_back(chord);
    repaint();
}

void ScoreEntryPanel::addRest(NoteValue duration) {
    float beats = noteValueToBeats(duration);
    if (dottedButton_ && dottedButton_->getToggleState()) beats *= 1.5f;
    if (tripletButton_ && tripletButton_->getToggleState()) beats *= (2.0f / 3.0f);

    cursorBeat_ += beats;
    auto beatsPerBar = defaultBeatsPerBar(timeSignatures_, cursorMeasure_);
    while (cursorBeat_ > beatsPerBar) {
        cursorBeat_ -= beatsPerBar;
        cursorMeasure_ += 1;
    }
    repaint();
}

void ScoreEntryPanel::setDynamic(Dynamic dynamic) {
    currentDynamic_ = dynamic;
}

void ScoreEntryPanel::setArticulation(Articulation articulation) {
    currentArticulation_ = articulation;
}

void ScoreEntryPanel::addLyric(const std::string& text) {
    if (!notes_.empty()) {
        notes_.back().lyric = text;
        repaint();
    }
}

void ScoreEntryPanel::toggleDot() {
    if (!notes_.empty()) {
        notes_.back().dotted = !notes_.back().dotted;
        repaint();
    } else if (dottedButton_) {
        dottedButton_->setToggleState(!dottedButton_->getToggleState(), juce::dontSendNotification);
    }
}

void ScoreEntryPanel::makeTriplet() {
    if (!notes_.empty()) {
        notes_.back().triplet = true;
        repaint();
    } else if (tripletButton_) {
        tripletButton_->setToggleState(true, juce::dontSendNotification);
    }
}

void ScoreEntryPanel::toggleTie() {
    if (!notes_.empty()) {
        notes_.back().tie = !notes_.back().tie;
        repaint();
    }
}

void ScoreEntryPanel::moveCursorForward() {
    cursorBeat_ += 1.0f;
    auto beatsPerBar = defaultBeatsPerBar(timeSignatures_, cursorMeasure_);
    if (cursorBeat_ > beatsPerBar) {
        cursorBeat_ = 1.0f;
        cursorMeasure_ += 1;
    }
    repaint();
}

void ScoreEntryPanel::moveCursorBackward() {
    cursorBeat_ -= 1.0f;
    if (cursorBeat_ < 1.0f) {
        cursorMeasure_ = std::max(1, cursorMeasure_ - 1);
        cursorBeat_ = static_cast<float>(defaultBeatsPerBar(timeSignatures_, cursorMeasure_));
    }
    repaint();
}

void ScoreEntryPanel::moveCursorToMeasure(int measure) {
    cursorMeasure_ = std::max(1, measure);
    cursorBeat_ = 1.0f;
    repaint();
}

void ScoreEntryPanel::moveCursorToNextMeasure() {
    cursorMeasure_ += 1;
    cursorBeat_ = 1.0f;
    repaint();
}

void ScoreEntryPanel::loadTemplate(const ScoreTemplate& template_) {
    notes_ = template_.notes;
    chordSymbols_ = template_.chords;
    cursorMeasure_ = 1;
    cursorBeat_ = 1.0f;
    repaint();
}

std::vector<ScoreEntryPanel::ScoreTemplate> ScoreEntryPanel::getAvailableTemplates() const {
    return templates_;
}

void ScoreEntryPanel::quickEntry(const std::string& description) {
    parseQuickEntry(description);
    repaint();
}

juce::MidiBuffer ScoreEntryPanel::toMidiBuffer() const {
    juce::MidiBuffer buffer;
    const int ticksPerQuarter = 480;
    const int channel = 1;

    for (const auto& note : notes_) {
        int beatsPerBar = defaultBeatsPerBar(timeSignatures_, note.measure);
        float beats = noteValueToBeats(note.duration);
        if (note.dotted) beats *= 1.5f;
        if (note.triplet) beats *= (2.0f / 3.0f);

        int ticksPerBar = ticksPerQuarter * beatsPerBar;
        int startTick = (note.measure - 1) * ticksPerBar +
                        static_cast<int>((note.beat - 1.0f) * ticksPerQuarter);
        int durationTicks = static_cast<int>(beats * ticksPerQuarter);

        const int velocity = dynamicToVelocity(note.dynamic);
        buffer.addEvent(juce::MidiMessage::noteOn(channel, note.pitch, (juce::uint8)velocity), startTick);
        buffer.addEvent(juce::MidiMessage::noteOff(channel, note.pitch), startTick + durationTicks);
    }

    return buffer;
}

void ScoreEntryPanel::fromMidiBuffer(const juce::MidiBuffer& buffer) {
    notes_.clear();
    std::map<int, int> noteOnTick;
    std::map<int, int> noteOnVelocity;

    const int ticksPerQuarter = 480;
    const int beatsPerBar = defaultBeatsPerBar(timeSignatures_, 1);
    const int ticksPerBar = ticksPerQuarter * beatsPerBar;

    for (const auto metadata : buffer) {
        const auto msg = metadata.getMessage();
        const int tick = metadata.samplePosition;
        if (msg.isNoteOn()) {
            noteOnTick[msg.getNoteNumber()] = tick;
            noteOnVelocity[msg.getNoteNumber()] = msg.getVelocity();
        } else if (msg.isNoteOff()) {
            int pitch = msg.getNoteNumber();
            auto it = noteOnTick.find(pitch);
            if (it != noteOnTick.end()) {
                int start = it->second;
                int duration = std::max(1, tick - start);
                NotationNote note;
                note.pitch = pitch;
                note.dynamic = currentDynamic_;
                note.articulation = currentArticulation_;
                note.duration = NoteValue::Quarter;
                note.dotted = false;
                note.triplet = false;
                note.tie = false;
                note.measure = start / ticksPerBar + 1;
                note.beat = ((start % ticksPerBar) / static_cast<float>(ticksPerQuarter)) + 1.0f;
                notes_.push_back(note);
                noteOnTick.erase(it);
            }
        }
    }
    repaint();
}

void ScoreEntryPanel::playFromStart() {
    cursorMeasure_ = 1;
    cursorBeat_ = 1.0f;
    onPlayClicked();
}

void ScoreEntryPanel::playFromCursor() {
    onPlayClicked();
}

void ScoreEntryPanel::stop() {
    onStopClicked();
}

void ScoreEntryPanel::toggleMetronome() {
    onMetronomeToggled();
}

void ScoreEntryPanel::setViewMode(ViewMode mode) {
    viewMode_ = mode;
    repaint();
}

void ScoreEntryPanel::setZoom(float zoomFactor) {
    zoomFactor_ = juce::jlimit(0.5f, 2.0f, zoomFactor);
    repaint();
}

void ScoreEntryPanel::setShowChordSymbols(bool show) {
    showChordSymbols_ = show;
    repaint();
}

void ScoreEntryPanel::setShowLyrics(bool show) {
    showLyrics_ = show;
    repaint();
}

void ScoreEntryPanel::setShowDynamics(bool show) {
    showDynamics_ = show;
    repaint();
}

bool ScoreEntryPanel::exportMusicXML(const juce::File& outputFile) {
    juce::FileOutputStream stream(outputFile);
    if (!stream.openedOk()) {
        return false;
    }

    stream << "<score-partwise>\n";
    for (const auto& note : notes_) {
        stream << "  <note pitch=\"" << note.pitch << "\" measure=\"" << note.measure
               << "\" beat=\"" << note.beat << "\" />\n";
    }
    stream << "</score-partwise>\n";
    stream.flush();
    return true;
}

bool ScoreEntryPanel::exportPDF(const juce::File& outputFile) {
    // Placeholder: export a simple text-based representation for now
    juce::FileOutputStream stream(outputFile);
    if (!stream.openedOk()) {
        return false;
    }
    stream << "Score Export (PDF placeholder)\n";
    for (const auto& note : notes_) {
        stream << "Note " << note.pitch << " @ " << note.measure << ":" << note.beat << "\n";
    }
    stream.flush();
    return true;
}

bool ScoreEntryPanel::exportMIDI(const juce::File& outputFile) {
    juce::MidiFile midi;
    midi.setTicksPerQuarterNote(480);
    juce::MidiMessageSequence seq;

    const int channel = 1;
    for (const auto& note : notes_) {
        int beatsPerBar = defaultBeatsPerBar(timeSignatures_, note.measure);
        float beats = noteValueToBeats(note.duration);
        if (note.dotted) beats *= 1.5f;
        if (note.triplet) beats *= (2.0f / 3.0f);

        int ticksPerBar = 480 * beatsPerBar;
        int startTick = (note.measure - 1) * ticksPerBar +
                        static_cast<int>((note.beat - 1.0f) * 480);
        int durationTicks = static_cast<int>(beats * 480);
        const int velocity = dynamicToVelocity(note.dynamic);

        seq.addEvent(juce::MidiMessage::noteOn(channel, note.pitch, (juce::uint8)velocity), startTick);
        seq.addEvent(juce::MidiMessage::noteOff(channel, note.pitch), startTick + durationTicks);
    }
    seq.updateMatchedPairs();
    midi.addTrack(seq);

    juce::FileOutputStream stream(outputFile);
    if (!stream.openedOk()) {
        return false;
    }
    midi.writeTo(stream);
    return true;
}

void ScoreEntryPanel::drawStaff(juce::Graphics& g, juce::Rectangle<int> area) {
    g.setColour(juce::Colours::white);
    const int staffHeight = 80 * zoomFactor_;
    const int staffTop = area.getY() + area.getHeight() / 3;
    const int lineSpacing = std::max(8, static_cast<int>(staffHeight / 4.0f));

    for (int i = 0; i < 5; ++i) {
        int y = staffTop + i * lineSpacing;
        g.drawLine((float)area.getX(), (float)y, (float)area.getRight(), (float)y, 1.5f);
    }

    // Bar lines for existing measures
    const int totalMeasures = std::max(1, getMaxMeasureIndex());
    const int measureWidth = std::max(120, static_cast<int>(area.getWidth() / totalMeasures));
    for (int m = 0; m <= totalMeasures; ++m) {
        int x = area.getX() + m * measureWidth;
        drawBarline(g, x);
    }
}

void ScoreEntryPanel::drawClef(juce::Graphics& g, Clef clef, juce::Point<int> position) {
    juce::String clefSymbol = "G";
    if (clef == Clef::Bass) clefSymbol = "F";
    else if (clef == Clef::Alto) clefSymbol = "C";
    else if (clef == Clef::Tenor) clefSymbol = "C";
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(22.0f));
    g.drawText(clefSymbol, position.x, position.y, 20, 40, juce::Justification::centred);
}

void ScoreEntryPanel::drawTimeSignature(juce::Graphics& g, int numerator, int denominator,
                                        juce::Point<int> position) {
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(18.0f));
    g.drawFittedText(juce::String(numerator) + "\n" + juce::String(denominator),
                     position.x, position.y, 20, 40, juce::Justification::centred, 2);
}

void ScoreEntryPanel::drawKeySignature(juce::Graphics& g, const std::string& key,
                                       juce::Point<int> position) {
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(16.0f));
    g.drawText(key, position.x, position.y, 60, 20, juce::Justification::left);
}

void ScoreEntryPanel::drawNote(juce::Graphics& g, const NotationNote& note,
                               juce::Point<int> position) {
    g.setColour(juce::Colours::white);
    const int size = 12;
    g.fillEllipse((float)position.x, (float)position.y - size / 2.0f, (float)size, (float)size);

    if (note.tie) {
        g.setColour(juce::Colours::lightgrey);
        g.drawArc((float)position.x, (float)position.y - 6.0f,
                  (float)size + 12.0f, 12.0f, juce::MathConstants<float>::pi, 2.0f * juce::MathConstants<float>::pi, 1.0f);
    }

    if (showLyrics_ && !note.lyric.empty()) {
        g.setColour(juce::Colours::lightgrey);
        g.drawText(note.lyric, position.x - 6, position.y + 14, 40, 16, juce::Justification::left);
    }
}

void ScoreEntryPanel::drawChordSymbol(juce::Graphics& g, const ChordSymbol& chord,
                                      juce::Point<int> position) {
    g.setColour(juce::Colours::lightgreen);
    g.setFont(juce::Font(14.0f, juce::Font::bold));
    g.drawText(chord.symbol, position.x, position.y, 60, 20, juce::Justification::left);
}

void ScoreEntryPanel::drawBarline(juce::Graphics& g, int x) {
    g.setColour(juce::Colours::white);
    auto area = getScoreArea();
    const int staffHeight = 80 * zoomFactor_;
    const int staffTop = area.getY() + area.getHeight() / 3;
    const int lineSpacing = std::max(8, static_cast<int>(staffHeight / 4.0f));
    int top = staffTop;
    int bottom = staffTop + 4 * lineSpacing;
    g.drawLine((float)x, (float)top, (float)x, (float)bottom, 1.0f);
}

void ScoreEntryPanel::drawCursor(juce::Graphics& g) {
    g.setColour(juce::Colours::yellow);
    auto area = getScoreArea();
    const int totalMeasures = std::max(1, getMaxMeasureIndex());
    const int measureWidth = std::max(120, static_cast<int>(area.getWidth() / totalMeasures));
    const int beatsPerBar = defaultBeatsPerBar(timeSignatures_, cursorMeasure_);
    float beatOffset = (cursorBeat_ - 1.0f) / std::max(1, beatsPerBar);
    int x = area.getX() + static_cast<int>((cursorMeasure_ - 1) * measureWidth + beatOffset * measureWidth);

    g.drawLine((float)x, (float)area.getY(), (float)x, (float)area.getBottom(), 1.2f);
}

int ScoreEntryPanel::staffPositionToPitch(int yPosition, Clef clef) const {
    auto area = getScoreArea();
    const int staffHeight = 80 * zoomFactor_;
    const int staffTop = area.getY() + area.getHeight() / 3;
    const float halfStep = std::max(4.0f, staffHeight / 8.0f);
    const int centerLine = staffTop + static_cast<int>(2 * halfStep);

    int basePitch = 71; // B4 on middle line for treble
    if (clef == Clef::Bass) basePitch = 50; // D3
    else if (clef == Clef::Alto) basePitch = 60;
    else if (clef == Clef::Tenor) basePitch = 55;

    int semitoneSteps = static_cast<int>(std::round((centerLine - yPosition) / halfStep));
    return juce::jlimit(0, 127, basePitch + semitoneSteps);
}

int ScoreEntryPanel::pitchToStaffPosition(int pitch, Clef clef) const {
    auto area = getScoreArea();
    const int staffHeight = 80 * zoomFactor_;
    const int staffTop = area.getY() + area.getHeight() / 3;
    const float halfStep = std::max(4.0f, staffHeight / 8.0f);
    const int centerLine = staffTop + static_cast<int>(2 * halfStep);

    int basePitch = 71; // B4
    if (clef == Clef::Bass) basePitch = 50;
    else if (clef == Clef::Alto) basePitch = 60;
    else if (clef == Clef::Tenor) basePitch = 55;

    int semitoneSteps = basePitch - pitch;
    return static_cast<int>(centerLine + semitoneSteps * halfStep);
}

void ScoreEntryPanel::parseQuickEntry(const std::string& text) {
    const auto lower = juce::String(text).toLowerCase();
    notes_.clear();
    cursorMeasure_ = 1;
    cursorBeat_ = 1.0f;

    if (lower.contains("c major scale")) {
        std::vector<int> scale = {60, 62, 64, 65, 67, 69, 71, 72};
        for (int note : scale) {
            addNote(note, NoteValue::Quarter, false);
        }
    } else if (lower.contains("arpeggio")) {
        std::vector<int> arp = {60, 64, 67, 72};
        for (int note : arp) {
            addNote(note, NoteValue::Eighth, false);
        }
    } else if (lower.contains("chord")) {
        addChord({60, 64, 67}, NoteValue::Whole);
    } else {
        // Default to a simple rhythm
        addNote(60, NoteValue::Quarter, false);
        addNote(62, NoteValue::Quarter, false);
        addNote(64, NoteValue::Quarter, false);
        addRest(NoteValue::Quarter);
    }
}

void ScoreEntryPanel::initializeTemplates() {
    ScoreTemplate scaleTemplate;
    scaleTemplate.name = "C Major Scale";
    scaleTemplate.description = "One octave C major scale in quarter notes.";
    float beat = 1.0f;
    int measure = 1;
    for (int note : {60, 62, 64, 65, 67, 69, 71, 72}) {
        NotationNote n{};
        n.pitch = note;
        n.duration = NoteValue::Quarter;
        n.dotted = false;
        n.triplet = false;
        n.dynamic = Dynamic::MezzoForte;
        n.articulation = Articulation::None;
        n.tie = false;
        n.measure = measure;
        n.beat = beat;
        scaleTemplate.notes.push_back(n);
        beat += 1.0f;
        if (beat > 4.0f) {
            beat = 1.0f;
            ++measure;
        }
    }
    templates_.push_back(scaleTemplate);

    ScoreTemplate chordTemplate;
    chordTemplate.name = "I - V - vi - IV";
    chordTemplate.description = "Common pop progression in C major.";
    beat = 1.0f;
    measure = 1;
    for (const auto& chord : {"C", "G", "Am", "F"}) {
        chordTemplate.chords.push_back({chord, measure, beat});
        measure += 1;
    }
    templates_.push_back(chordTemplate);
}

void ScoreEntryPanel::onNoteValueSelected(NoteValue value) {
    currentNoteValue_ = value;
}

void ScoreEntryPanel::onDynamicSelected(Dynamic dynamic) {
    currentDynamic_ = dynamic;
}

void ScoreEntryPanel::onArticulationSelected(Articulation articulation) {
    currentArticulation_ = articulation;
}

void ScoreEntryPanel::onQuickEntryExecuted() {
    quickEntry(quickEntryInput_->getText().toStdString());
}

void ScoreEntryPanel::onPlayClicked() {
    // Placeholder for playback integration
    juce::Logger::writeToLog("ScoreEntryPanel: play from measure " + juce::String(cursorMeasure_));
}

void ScoreEntryPanel::onStopClicked() {
    juce::Logger::writeToLog("ScoreEntryPanel: stop playback");
}

void ScoreEntryPanel::onMetronomeToggled() {
    bool enabled = metronomeButton_ && metronomeButton_->getToggleState();
    juce::Logger::writeToLog(juce::String("Metronome ") + (enabled ? "enabled" : "disabled"));
}

int ScoreEntryPanel::getMaxMeasureIndex() const {
    int maxMeasure = cursorMeasure_;
    for (const auto& note : notes_) {
        maxMeasure = std::max(maxMeasure, note.measure);
    }
    for (const auto& chord : chordSymbols_) {
        maxMeasure = std::max(maxMeasure, chord.measure);
    }
    return std::max(1, maxMeasure);
}

juce::Rectangle<int> ScoreEntryPanel::getScoreArea() const {
    auto bounds = getLocalBounds().reduced(10);
    bounds.removeFromTop(140);
    return bounds;
}

} // namespace midikompanion
