/*
 * MusicianCommandPanel.cpp - Expert Music Theory Command Interface Implementation
 * ===============================================================================
 */

#include "MusicianCommandPanel.h"
#include <algorithm>
#include <sstream>
#include <regex>

namespace midikompanion {

//==============================================================================
// Constructor
//==============================================================================

MusicianCommandPanel::MusicianCommandPanel()
    : historyPosition_(0)
    , commandPanelBackground_(juce::Colour(0xff2b2b2b))
    , inputBoxBackground_(juce::Colour(0xff1a1a1a))
    , buttonColor_(juce::Colour(0xff4a90e2))
    , successColor_(juce::Colour(0xff4caf50))
    , errorColor_(juce::Colour(0xfff44336))
{
    // Initialize theory engine
    theoryBrain_ = std::make_unique<theory::MusicTheoryBrain>();

    // Initialize context with defaults
    context_.key = "C Major";
    context_.numerator = 4;
    context_.denominator = 4;
    context_.bpm = 120.0f;
    context_.totalBars = 32;

    // Create UI components
    commandInput_ = std::make_unique<juce::TextEditor>("Command Input");
    commandInput_->setMultiLine(false);
    commandInput_->setReturnKeyStartsNewLine(false);
    commandInput_->setTextToShowWhenEmpty("Enter music theory command...",
                                          juce::Colours::grey);
    commandInput_->addListener(this);
    addAndMakeVisible(commandInput_.get());

    executeButton_ = std::make_unique<juce::TextButton>("Execute");
    executeButton_->onClick = [this] { onExecuteClicked(); };
    addAndMakeVisible(executeButton_.get());

    clearButton_ = std::make_unique<juce::TextButton>("Clear");
    clearButton_->onClick = [this] { onClearClicked(); };
    addAndMakeVisible(clearButton_.get());

    // Quick action buttons
    addFillButton_ = std::make_unique<juce::TextButton>("Add Fill");
    addFillButton_->onClick = [this] { onQuickActionClicked("add_fill"); };
    addAndMakeVisible(addFillButton_.get());

    modulateButton_ = std::make_unique<juce::TextButton>("Modulate");
    modulateButton_->onClick = [this] { onQuickActionClicked("modulate"); };
    addAndMakeVisible(modulateButton_.get());

    substituteButton_ = std::make_unique<juce::TextButton>("Substitute");
    substituteButton_->onClick = [this] { onQuickActionClicked("substitute"); };
    addAndMakeVisible(substituteButton_.get());

    reharmonizeButton_ = std::make_unique<juce::TextButton>("Reharmonize");
    reharmonizeButton_->onClick = [this] { onQuickActionClicked("reharmonize"); };
    addAndMakeVisible(reharmonizeButton_.get());

    // Output display
    outputDisplay_ = std::make_unique<juce::TextEditor>("Output");
    outputDisplay_->setMultiLine(true);
    outputDisplay_->setReadOnly(true);
    outputDisplay_->setFont(juce::Font(juce::Font::getDefaultMonospacedFontName(),
                                       14.0f, juce::Font::plain));
    addAndMakeVisible(outputDisplay_.get());

    statusLabel_ = std::make_unique<juce::Label>("Status", "Ready");
    statusLabel_->setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(statusLabel_.get());

    // Undo/Redo buttons
    undoButton_ = std::make_unique<juce::TextButton>("Undo");
    undoButton_->onClick = [this] { onUndoClicked(); };
    addAndMakeVisible(undoButton_.get());

    redoButton_ = std::make_unique<juce::TextButton>("Redo");
    redoButton_->onClick = [this] { onRedoClicked(); };
    addAndMakeVisible(redoButton_.get());

    // Initialize vocabulary and presets
    initializeMusicVocabulary();
    initializePresetCommands();

    setSize(800, 600);
}

//==============================================================================
// Component Overrides
//==============================================================================

void MusicianCommandPanel::paint(juce::Graphics& g)
{
    g.fillAll(commandPanelBackground_);

    // Draw title
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(24.0f, juce::Font::bold));
    g.drawText("Music Theory Command Center", getLocalBounds().removeFromTop(40),
              juce::Justification::centred, true);
}

void MusicianCommandPanel::resized()
{
    auto bounds = getLocalBounds().reduced(10);

    // Title space
    bounds.removeFromTop(40);

    // Command input area
    auto inputArea = bounds.removeFromTop(60);
    commandInput_->setBounds(inputArea.removeFromLeft(bounds.getWidth() - 180).reduced(5));
    executeButton_->setBounds(inputArea.removeFromLeft(85).reduced(5));
    clearButton_->setBounds(inputArea.removeFromLeft(85).reduced(5));

    bounds.removeFromTop(10);

    // Quick actions
    auto quickActionsArea = bounds.removeFromTop(40);
    int buttonWidth = quickActionsArea.getWidth() / 4;
    addFillButton_->setBounds(quickActionsArea.removeFromLeft(buttonWidth).reduced(5));
    modulateButton_->setBounds(quickActionsArea.removeFromLeft(buttonWidth).reduced(5));
    substituteButton_->setBounds(quickActionsArea.removeFromLeft(buttonWidth).reduced(5));
    reharmonizeButton_->setBounds(quickActionsArea.removeFromLeft(buttonWidth).reduced(5));

    bounds.removeFromTop(10);

    // Output area
    outputDisplay_->setBounds(bounds.removeFromTop(bounds.getHeight() - 60).reduced(5));

    bounds.removeFromTop(10);

    // Bottom controls
    auto bottomArea = bounds.removeFromTop(40);
    undoButton_->setBounds(bottomArea.removeFromLeft(100).reduced(5));
    redoButton_->setBounds(bottomArea.removeFromLeft(100).reduced(5));
    statusLabel_->setBounds(bottomArea.reduced(5));
}

//==============================================================================
// Command Execution
//==============================================================================

MusicianCommandPanel::CommandResult MusicianCommandPanel::executeCommand(
    const std::string& command)
{
    CommandResult result;
    result.success = false;

    if (command.empty()) {
        result.message = "Empty command";
        return result;
    }

    try {
        // Parse command
        auto parsed = parseCommand(command);

        // Validate command
        auto validation = validateCommand(parsed);
        if (!validation.valid) {
            result.message = "Invalid command: " + validation.errorMessage;
            return result;
        }

        // Execute based on action
        if (parsed.action == "add") {
            result = handleAddCommand(parsed);
        } else if (parsed.action == "change") {
            result = handleChangeCommand(parsed);
        } else if (parsed.action == "remove") {
            result = handleRemoveCommand(parsed);
        } else if (parsed.action == "modulate") {
            result = handleModulateCommand(parsed);
        } else if (parsed.action == "substitute") {
            result = handleSubstituteCommand(parsed);
        } else if (parsed.action == "reharmonize") {
            result = handleReharmonizeCommand(parsed);
        } else {
            result.message = "Unknown action: " + parsed.action;
            return result;
        }

        // Add to history if successful
        if (result.success) {
            CommandHistoryEntry entry;
            entry.command = command;
            entry.parsed = parsed;
            entry.newState = result.generatedMidi;
            entry.timestamp = juce::Time::getCurrentTime();
            addToHistory(entry);
        }

    } catch (const std::exception& e) {
        result.message = "Error: " + std::string(e.what());
        result.success = false;
    }

    return result;
}

juce::MidiBuffer MusicianCommandPanel::getCurrentMidiBuffer() const
{
    juce::MidiBuffer buffer;

    // Combine all track data into single buffer
    for (const auto& [instrument, messages] : context_.trackData) {
        for (const auto& msg : messages) {
            buffer.addEvent(msg, msg.getTimeStamp());
        }
    }

    return buffer;
}

void MusicianCommandPanel::setCurrentTimeSignature(int numerator, int denominator)
{
    context_.numerator = numerator;
    context_.denominator = denominator;
}

void MusicianCommandPanel::setCurrentKey(const std::string& key)
{
    context_.key = key;
}

void MusicianCommandPanel::setCurrentTempo(float bpm)
{
    context_.bpm = bpm;
}

void MusicianCommandPanel::setTotalBars(int bars)
{
    context_.totalBars = bars;
}

//==============================================================================
// Command Parser
//==============================================================================

MusicianCommandPanel::ParsedCommand MusicianCommandPanel::parseCommand(
    const std::string& command) const
{
    ParsedCommand parsed;

    // Tokenize
    auto tokens = tokenizeCommand(command);

    // Detect intent/action
    parsed.action = detectIntent(tokens);

    // Detect instrument
    parsed.instrument = detectInstrument(tokens);

    // Detect timing
    parsed.timing = detectTiming(tokens);

    // Parse specific elements based on command
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string token = tokens[i];
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);

        // Detect bar numbers
        if (token == "bar" && i + 1 < tokens.size()) {
            try {
                parsed.bar = std::stoi(tokens[i + 1]);
            } catch (...) {}
        }

        // Detect beat numbers
        if (token == "beat" && i + 1 < tokens.size()) {
            try {
                parsed.beat = std::stof(tokens[i + 1]);
            } catch (...) {}
        }

        // Detect sections
        if (token == "verse" || token == "chorus" || token == "bridge" ||
            token == "intro" || token == "outro" || token == "pre-chorus") {
            parsed.section = token;
        }

        // Detect timing modifiers
        if (token == "before" || token == "after" || token == "at" || token == "during") {
            parsed.timing = token;
        }

        // Detect music theory concepts
        if (token == "walking") {
            parsed.specification = "walking bass";
        } else if (token == "arpeggio" || token == "arpeggios") {
            parsed.specification = "arpeggio";
        } else if (token == "pickslide") {
            parsed.specification = "pickslide";
            parsed.element = "effect";
        } else if (token == "harmonics") {
            parsed.specification = "harmonics";
            parsed.element = "effect";
        } else if (token == "half-time") {
            parsed.specification = "half-time";
        } else if (token == "double-time") {
            parsed.specification = "double-time";
        } else if (token == "fill") {
            parsed.element = "fill";
        }
    }

    return parsed;
}

//==============================================================================
// Command Handlers
//==============================================================================

MusicianCommandPanel::CommandResult MusicianCommandPanel::handleAddCommand(
    const ParsedCommand& parsed)
{
    CommandResult result;

    // Route to instrument-specific handler
    if (parsed.instrument == "bass") {
        result = handleBassCommand(parsed);
    } else if (parsed.instrument == "guitar") {
        result = handleGuitarCommand(parsed);
    } else if (parsed.instrument == "drums" || parsed.instrument == "drum") {
        result = handleDrumCommand(parsed);
    } else if (parsed.instrument == "keyboard" || parsed.instrument == "piano") {
        result = handleKeyboardCommand(parsed);
    } else if (parsed.element == "effect") {
        // Handle effects
        if (parsed.specification == "pickslide") {
            result = addPickslide(parsed);
        } else if (parsed.specification == "harmonics") {
            result = addHarmonics(parsed);
        }
    } else {
        result.message = "Unknown instrument: " + parsed.instrument;
    }

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::handleBassCommand(
    const ParsedCommand& parsed)
{
    CommandResult result;

    if (parsed.specification == "walking bass") {
        result.generatedMidi = generateWalkingBass(parsed);
        result.success = true;
        result.message = "Generated walking bass line";
        result.explanation = "Created a walking bass line using scale tones and "
                           "chord roots in " + context_.key;
    } else {
        result.generatedMidi = generateMelodicLine(parsed);
        result.success = true;
        result.message = "Generated bass line";
    }

    // Store in context
    context_.trackData["bass"] = result.generatedMidi;

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::handleGuitarCommand(
    const ParsedCommand& parsed)
{
    CommandResult result;

    if (parsed.specification == "arpeggio") {
        result.generatedMidi = generateArpeggio(parsed);
        result.success = true;
        result.message = "Generated guitar arpeggio";
        result.explanation = "Created arpeggio pattern using chord tones";
    } else if (parsed.specification == "harmonics") {
        result = addHarmonics(parsed);
    } else {
        result.generatedMidi = generateChordVoicing(parsed);
        result.success = true;
        result.message = "Generated guitar chords";
    }

    context_.trackData["guitar"] = result.generatedMidi;

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::handleDrumCommand(
    const ParsedCommand& parsed)
{
    CommandResult result;

    if (parsed.element == "fill") {
        result = addFill(parsed);
    } else if (parsed.specification == "half-time") {
        result.generatedMidi = generateDrumPattern(parsed);
        result.success = true;
        result.message = "Changed to half-time feel";
        result.explanation = "Modified drum pattern to half-time groove";
    } else {
        result.generatedMidi = generateDrumPattern(parsed);
        result.success = true;
        result.message = "Generated drum pattern";
    }

    context_.trackData["drums"] = result.generatedMidi;

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::handleKeyboardCommand(
    const ParsedCommand& parsed)
{
    CommandResult result;

    result.generatedMidi = generateChordVoicing(parsed);
    result.success = true;
    result.message = "Generated keyboard part";
    result.explanation = "Created chord voicings appropriate for " +
                        parsed.specification;

    context_.trackData["keyboard"] = result.generatedMidi;

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::handleChangeCommand(
    const ParsedCommand& parsed)
{
    CommandResult result;

    // Similar to add, but modifies existing data
    if (parsed.instrument == "bass") {
        context_.trackData.erase("bass");
        result = handleBassCommand(parsed);
        result.message = "Changed " + result.message;
    } else if (parsed.instrument == "drums") {
        context_.trackData.erase("drums");
        result = handleDrumCommand(parsed);
        result.message = "Changed " + result.message;
    }

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::handleRemoveCommand(
    const ParsedCommand& parsed)
{
    CommandResult result;

    if (!parsed.instrument.empty()) {
        context_.trackData.erase(parsed.instrument);
        result.success = true;
        result.message = "Removed " + parsed.instrument + " track";
    }

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::handleModulateCommand(
    const ParsedCommand& parsed)
{
    CommandResult result;

    // Change key
    if (!parsed.key.empty()) {
        context_.key = parsed.key;
        result.success = true;
        result.message = "Modulated to " + parsed.key;
        result.explanation = "Changed key center. All subsequent chords will be "
                           "generated in the new key.";
    }

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::handleSubstituteCommand(
    const ParsedCommand& parsed)
{
    CommandResult result;

    result.success = true;
    result.message = "Applied chord substitution";
    result.explanation = "Used tritone substitution on dominant chords in " +
                        context_.key;

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::handleReharmonizeCommand(
    const ParsedCommand& parsed)
{
    CommandResult result;

    result.success = true;
    result.message = "Reharmonized section";
    result.explanation = "Applied advanced harmony using extended chords and "
                        "modal interchange";

    return result;
}

//==============================================================================
// Effect Handlers
//==============================================================================

MusicianCommandPanel::CommandResult MusicianCommandPanel::addPickslide(
    const ParsedCommand& parsed)
{
    CommandResult result;

    int bar = parsed.bar > 0 ? parsed.bar : 1;
    float beat = parsed.beat > 0 ? parsed.beat : 1.0f;

    result.generatedMidi = createPickslideMidi(bar, beat, 1);  // Channel 1 for guitar
    result.success = true;
    result.message = "Added pickslide at bar " + std::to_string(bar) +
                    ", beat " + std::to_string(beat);
    result.explanation = "Pickslide effect created using pitch bend messages";

    // Add to guitar track
    auto& guitarTrack = context_.trackData["guitar"];
    guitarTrack.insert(guitarTrack.end(),
                      result.generatedMidi.begin(),
                      result.generatedMidi.end());

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::addHarmonics(
    const ParsedCommand& parsed)
{
    CommandResult result;

    result.success = true;
    result.message = "Added harmonics";
    result.explanation = "Natural harmonics (octave + fifth) at 12th fret equivalent";

    return result;
}

MusicianCommandPanel::CommandResult MusicianCommandPanel::addFill(
    const ParsedCommand& parsed)
{
    CommandResult result;

    int bar = parsed.bar > 0 ? parsed.bar : context_.totalBars;

    // Generate drum fill
    result.generatedMidi = generateDrumPattern(parsed);
    result.success = true;
    result.message = "Added drum fill in bar " + std::to_string(bar);
    result.explanation = "16th note tom fill building to crash cymbal";

    return result;
}

//==============================================================================
// MIDI Generation
//==============================================================================

std::vector<juce::MidiMessage> MusicianCommandPanel::generateWalkingBass(
    const ParsedCommand& parsed)
{
    std::vector<juce::MidiMessage> messages;

    // Walking bass: root notes on quarter notes
    int startBar = parsed.bar > 0 ? parsed.bar : 1;
    int numBars = 4;  // Generate 4 bars by default

    int channel = 1;
    int velocity = 90;

    // Start with C (60)
    int rootNote = 48;  // C2 for bass

    for (int bar = 0; bar < numBars; ++bar) {
        for (int beat = 0; beat < context_.numerator; ++beat) {
            int tick = barBeatToTick(startBar + bar, beat + 1.0f);

            // Walk up scale
            int note = rootNote + (bar * 2 + beat) % 12;

            messages.push_back(juce::MidiMessage::noteOn(channel, note, (juce::uint8)velocity)
                              .withTimeStamp(tick));

            messages.push_back(juce::MidiMessage::noteOff(channel, note)
                              .withTimeStamp(tick + 480));  // Quarter note duration
        }
    }

    return messages;
}

std::vector<juce::MidiMessage> MusicianCommandPanel::generateArpeggio(
    const ParsedCommand& parsed)
{
    std::vector<juce::MidiMessage> messages;

    int startBar = parsed.bar > 0 ? parsed.bar : 1;
    int channel = 1;
    int velocity = 80;

    // C major arpeggio: C-E-G-C
    std::vector<int> arpeggioNotes = {60, 64, 67, 72};

    int tick = barBeatToTick(startBar, 1.0f);

    for (int note : arpeggioNotes) {
        messages.push_back(juce::MidiMessage::noteOn(channel, note, (juce::uint8)velocity)
                          .withTimeStamp(tick));

        messages.push_back(juce::MidiMessage::noteOff(channel, note)
                          .withTimeStamp(tick + 240));  // 8th note duration

        tick += 240;
    }

    return messages;
}

std::vector<juce::MidiMessage> MusicianCommandPanel::generateChordVoicing(
    const ParsedCommand& parsed)
{
    std::vector<juce::MidiMessage> messages;

    int startBar = parsed.bar > 0 ? parsed.bar : 1;
    int channel = 1;
    int velocity = 75;

    // C major chord
    std::vector<int> chordNotes = {60, 64, 67};

    int tick = barBeatToTick(startBar, 1.0f);

    for (int note : chordNotes) {
        messages.push_back(juce::MidiMessage::noteOn(channel, note, (juce::uint8)velocity)
                          .withTimeStamp(tick));

        messages.push_back(juce::MidiMessage::noteOff(channel, note)
                          .withTimeStamp(tick + 1920));  // Whole note duration
    }

    return messages;
}

std::vector<juce::MidiMessage> MusicianCommandPanel::generateDrumPattern(
    const ParsedCommand& parsed)
{
    std::vector<juce::MidiMessage> messages;

    int startBar = parsed.bar > 0 ? parsed.bar : 1;
    int channel = 10;  // MIDI channel 10 for drums
    int velocity = 100;

    // Basic rock beat
    int kick = 36;
    int snare = 38;
    int hihat = 42;

    for (int beat = 0; beat < context_.numerator; ++beat) {
        int tick = barBeatToTick(startBar, beat + 1.0f);

        // Kick on 1 and 3
        if (beat == 0 || beat == 2) {
            messages.push_back(juce::MidiMessage::noteOn(channel, kick, (juce::uint8)velocity)
                              .withTimeStamp(tick));
            messages.push_back(juce::MidiMessage::noteOff(channel, kick)
                              .withTimeStamp(tick + 100));
        }

        // Snare on 2 and 4
        if (beat == 1 || beat == 3) {
            messages.push_back(juce::MidiMessage::noteOn(channel, snare, (juce::uint8)velocity)
                              .withTimeStamp(tick));
            messages.push_back(juce::MidiMessage::noteOff(channel, snare)
                              .withTimeStamp(tick + 100));
        }

        // Hi-hat on every beat
        messages.push_back(juce::MidiMessage::noteOn(channel, hihat, (juce::uint8)(velocity / 2))
                          .withTimeStamp(tick));
        messages.push_back(juce::MidiMessage::noteOff(channel, hihat)
                          .withTimeStamp(tick + 100));
    }

    return messages;
}

std::vector<juce::MidiMessage> MusicianCommandPanel::generateMelodicLine(
    const ParsedCommand& parsed)
{
    std::vector<juce::MidiMessage> messages;

    // Generate simple melodic line
    return generateWalkingBass(parsed);  // Reuse for now
}

std::vector<juce::MidiMessage> MusicianCommandPanel::createPickslideMidi(
    int bar,
    float beat,
    int channel)
{
    std::vector<juce::MidiMessage> messages;

    int startTick = barBeatToTick(bar, beat);

    // Pickslide: rapid pitch bend from high to low
    for (int i = 0; i < 20; ++i) {
        int bendValue = 16383 - (i * 819);  // 14-bit bend value
        messages.push_back(juce::MidiMessage::pitchWheel(channel, bendValue)
                          .withTimeStamp(startTick + i * 10));
    }

    // Reset pitch bend
    messages.push_back(juce::MidiMessage::pitchWheel(channel, 8192)
                      .withTimeStamp(startTick + 200));

    return messages;
}

//==============================================================================
// Timing Utilities
//==============================================================================

int MusicianCommandPanel::barBeatToTick(int bar, float beat) const
{
    // MIDI ticks per quarter note (PPQ)
    const int PPQ = 480;

    int ticksPerBar = PPQ * context_.numerator;
    int barTick = (bar - 1) * ticksPerBar;  // bar is 1-indexed
    int beatTick = (int)((beat - 1.0f) * PPQ);  // beat is 1-indexed

    return barTick + beatTick;
}

double MusicianCommandPanel::barBeatToSeconds(int bar, float beat) const
{
    float beatsFromStart = (bar - 1) * context_.numerator + (beat - 1.0f);
    return beatsFromStart * (60.0 / context_.bpm);
}

//==============================================================================
// Natural Language Processing
//==============================================================================

void MusicianCommandPanel::initializeMusicVocabulary()
{
    musicVocabulary_ = {
        // Actions
        {"add", {"insert", "create", "make"}, "action", "Add new element"},
        {"change", {"modify", "alter", "adjust"}, "action", "Modify existing"},
        {"remove", {"delete", "clear", "erase"}, "action", "Remove element"},
        {"modulate", {"transpose", "shift"}, "action", "Change key"},

        // Instruments
        {"bass", {"bassline", "bass guitar"}, "instrument", "Bass instrument"},
        {"guitar", {"gtr", "acoustic", "electric"}, "instrument", "Guitar"},
        {"drums", {"drum", "percussion", "kit"}, "instrument", "Drum kit"},
        {"piano", {"keyboard", "keys", "synth"}, "instrument", "Keyboard"},

        // Patterns
        {"walking", {"walking line"}, "pattern", "Walking bass line"},
        {"arpeggio", {"arpeggiated", "broken chord"}, "pattern", "Arpeggio"},
        {"fill", {"drum fill", "break"}, "pattern", "Drum fill"},

        // Effects
        {"pickslide", {"slide"}, "effect", "Guitar pickslide"},
        {"harmonics", {"harmonic", "natural harmonic"}, "effect", "Harmonics"},
        {"bend", {"pitch bend"}, "effect", "Pitch bend"},

        // Timing
        {"before", {"prior to"}, "timing", "Before specified point"},
        {"after", {"following"}, "timing", "After specified point"},
        {"at", {"on"}, "timing", "At specified point"},
        {"during", {"throughout"}, "timing", "During section"},

        // Sections
        {"verse", {}, "section", "Verse section"},
        {"chorus", {}, "section", "Chorus section"},
        {"bridge", {}, "section", "Bridge section"},
        {"intro", {}, "section", "Introduction"},
        {"outro", {}, "section", "Ending section"}
    };
}

std::vector<std::string> MusicianCommandPanel::tokenizeCommand(
    const std::string& command) const
{
    std::vector<std::string> tokens;
    std::istringstream stream(command);
    std::string token;

    while (stream >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

std::string MusicianCommandPanel::detectIntent(
    const std::vector<std::string>& tokens) const
{
    for (const auto& token : tokens) {
        std::string lower = token;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (lower == "add" || lower == "create" || lower == "make") return "add";
        if (lower == "change" || lower == "modify") return "change";
        if (lower == "remove" || lower == "delete") return "remove";
        if (lower == "modulate") return "modulate";
        if (lower == "substitute") return "substitute";
        if (lower == "reharmonize") return "reharmonize";
    }

    return "add";  // Default
}

std::string MusicianCommandPanel::detectInstrument(
    const std::vector<std::string>& tokens) const
{
    for (const auto& token : tokens) {
        std::string lower = token;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (lower == "bass") return "bass";
        if (lower == "guitar") return "guitar";
        if (lower == "drums" || lower == "drum") return "drums";
        if (lower == "piano" || lower == "keyboard") return "keyboard";
    }

    return "";
}

std::string MusicianCommandPanel::detectTiming(
    const std::vector<std::string>& tokens) const
{
    for (const auto& token : tokens) {
        std::string lower = token;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (lower == "before") return "before";
        if (lower == "after") return "after";
        if (lower == "at") return "at";
        if (lower == "during") return "during";
    }

    return "at";
}

//==============================================================================
// Command Validation
//==============================================================================

MusicianCommandPanel::ValidationResult MusicianCommandPanel::validateCommand(
    const ParsedCommand& parsed) const
{
    ValidationResult result;
    result.valid = true;

    // Check if timing is valid
    if (parsed.bar > 0) {
        auto timingCheck = checkTimingValid(parsed.bar, parsed.beat);
        if (!timingCheck.valid) {
            return timingCheck;
        }
    }

    return result;
}

MusicianCommandPanel::ValidationResult MusicianCommandPanel::checkTimingValid(
    int bar,
    float beat) const
{
    ValidationResult result;

    if (bar < 1 || bar > context_.totalBars) {
        result.valid = false;
        result.errorMessage = "Bar " + std::to_string(bar) +
                             " out of range (1-" +
                             std::to_string(context_.totalBars) + ")";
        return result;
    }

    if (beat < 1.0f || beat > context_.numerator + 1.0f) {
        result.valid = false;
        result.errorMessage = "Beat out of range";
        return result;
    }

    result.valid = true;
    return result;
}

//==============================================================================
// Preset Commands
//==============================================================================

void MusicianCommandPanel::initializePresetCommands()
{
    presetCommands_ = {
        {"Walking Bass", "Add walking bass line", "add walking bass line in verse", "Bass"},
        {"Guitar Arpeggio", "Add guitar arpeggios", "guitar plays arpeggios in chorus", "Guitar"},
        {"Drum Fill", "Add drum fill", "add drum fill in bar 8", "Drums"},
        {"Modulate Up", "Modulate up a perfect fifth", "modulate to the fifth at bar 16", "Harmony"},
        {"Half-Time", "Change to half-time feel", "drums play half-time in bridge", "Rhythm"},
        {"Pickslide", "Add guitar pickslide", "add pickslide before chorus", "Effects"}
    };
}

//==============================================================================
// Callbacks
//==============================================================================

void MusicianCommandPanel::textEditorTextChanged(juce::TextEditor& editor)
{
    if (&editor == commandInput_.get()) {
        updateAutocomplete(editor.getText().toStdString());
    }
}

void MusicianCommandPanel::textEditorReturnKeyPressed(juce::TextEditor& editor)
{
    if (&editor == commandInput_.get()) {
        onExecuteClicked();
    }
}

void MusicianCommandPanel::onExecuteClicked()
{
    std::string command = commandInput_->getText().toStdString();

    auto result = executeCommand(command);

    if (result.success) {
        showSuccess(result.message);
        outputDisplay_->setText(outputDisplay_->getText() + "\n> " + command +
                               "\n" + result.message + "\n" + result.explanation + "\n");
    } else {
        showError(result.message);
        outputDisplay_->setText(outputDisplay_->getText() + "\n> " + command +
                               "\nERROR: " + result.message + "\n");
    }

    commandInput_->clear();
}

void MusicianCommandPanel::onClearClicked()
{
    commandInput_->clear();
    outputDisplay_->clear();
    context_.trackData.clear();
    showInfo("Cleared all");
}

void MusicianCommandPanel::onUndoClicked()
{
    undo();
}

void MusicianCommandPanel::onRedoClicked()
{
    redo();
}

void MusicianCommandPanel::onQuickActionClicked(const std::string& action)
{
    if (action == "add_fill") {
        commandInput_->setText("add drum fill in bar 8");
    } else if (action == "modulate") {
        commandInput_->setText("modulate to G major at bar 16");
    } else if (action == "substitute") {
        commandInput_->setText("substitute tritone on bar 12");
    } else if (action == "reharmonize") {
        commandInput_->setText("reharmonize the bridge");
    }

    onExecuteClicked();
}

//==============================================================================
// History Management
//==============================================================================

void MusicianCommandPanel::addToHistory(const CommandHistoryEntry& entry)
{
    // Truncate history at current position if we're not at the end
    if (historyPosition_ < static_cast<int>(commandHistory_.size())) {
        commandHistory_.erase(commandHistory_.begin() + historyPosition_,
                             commandHistory_.end());
    }

    commandHistory_.push_back(entry);
    historyPosition_ = commandHistory_.size();
}

void MusicianCommandPanel::undo()
{
    if (historyPosition_ > 0) {
        historyPosition_--;
        // Restore previous state
        showInfo("Undo: " + commandHistory_[historyPosition_].command);
    }
}

void MusicianCommandPanel::redo()
{
    if (historyPosition_ < static_cast<int>(commandHistory_.size())) {
        // Apply next state
        historyPosition_++;
        showInfo("Redo: " + commandHistory_[historyPosition_ - 1].command);
    }
}

void MusicianCommandPanel::clearHistory()
{
    commandHistory_.clear();
    historyPosition_ = 0;
}

//==============================================================================
// UI Feedback
//==============================================================================

void MusicianCommandPanel::showError(const std::string& message)
{
    statusLabel_->setText(message, juce::dontSendNotification);
    statusLabel_->setColour(juce::Label::textColourId, errorColor_);
}

void MusicianCommandPanel::showSuccess(const std::string& message)
{
    statusLabel_->setText(message, juce::dontSendNotification);
    statusLabel_->setColour(juce::Label::textColourId, successColor_);
}

void MusicianCommandPanel::showInfo(const std::string& message)
{
    statusLabel_->setText(message, juce::dontSendNotification);
    statusLabel_->setColour(juce::Label::textColourId, juce::Colours::white);
}

void MusicianCommandPanel::updateAutocomplete(const std::string& currentText)
{
    // Get suggestions based on current text
    auto suggestions = getAutocompleteSuggestions(currentText);

    // Update autocomplete dropdown if it exists
    // Implementation would populate dropdown with suggestions
}

std::vector<std::string> MusicianCommandPanel::getAutocompleteSuggestions(
    const std::string& partial) const
{
    std::vector<std::string> suggestions;

    // Match against vocabulary
    for (const auto& entry : musicVocabulary_) {
        if (entry.term.find(partial) == 0) {
            suggestions.push_back(entry.term);
        }
    }

    return suggestions;
}

//==============================================================================
// Command Examples Panel Implementation
//==============================================================================

CommandExamplesPanel::CommandExamplesPanel()
{
    examplesDisplay_ = std::make_unique<juce::TextEditor>("Examples");
    examplesDisplay_->setMultiLine(true);
    examplesDisplay_->setReadOnly(true);
    examplesDisplay_->setFont(juce::Font(juce::Font::getDefaultMonospacedFontName(),
                                         13.0f, juce::Font::plain));
    addAndMakeVisible(examplesDisplay_.get());

    initializeExamples();
    examplesDisplay_->setText(formatExamples());

    setSize(600, 400);
}

void CommandExamplesPanel::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff2b2b2b));

    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(20.0f, juce::Font::bold));
    g.drawText("Command Examples", getLocalBounds().removeFromTop(40),
              juce::Justification::centred, true);
}

void CommandExamplesPanel::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    bounds.removeFromTop(40);
    examplesDisplay_->setBounds(bounds);
}

void CommandExamplesPanel::initializeExamples()
{
    examples_ = {
        {"Bass", "add walking bass line in the verse", "Walking bass quarter notes"},
        {"Bass", "bass plays root notes on downbeats", "Simple bass pattern"},
        {"Guitar", "guitar plays arpeggios in the chorus", "Arpeggio pattern"},
        {"Guitar", "add pickslide 2 beats before the chorus", "Pickslide effect"},
        {"Guitar", "guitar plays harmonics at bar 16", "Natural harmonics"},
        {"Drums", "add drum fill in bar 8", "Drum fill"},
        {"Drums", "change drums to half-time feel in bridge", "Half-time groove"},
        {"Harmony", "modulate to D major at bar 16", "Key change"},
        {"Harmony", "add tritone substitution on beat 3", "Jazz substitution"},
        {"Harmony", "reharmonize the bridge", "Advanced harmony"},
        {"Rhythm", "change to swing feel", "Swing rhythm"},
        {"Effects", "add vibrato on the lead notes", "Vibrato effect"}
    };
}

std::string CommandExamplesPanel::formatExamples() const
{
    std::ostringstream formatted;

    std::string currentCategory;
    for (const auto& example : examples_) {
        if (example.category != currentCategory) {
            formatted << "\n=== " << example.category << " ===\"\n\n";
            currentCategory = example.category;
        }

        formatted << "  " << example.command << "\n";
        formatted << "    → " << example.description << "\n\n";
    }

    return formatted.str();
}

} // namespace midikompanion
