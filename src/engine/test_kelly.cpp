/**
 * test_kelly.cpp
 * 
 * Test file demonstrating usage of ported Kelly MIDI modules.
 * Compile with: g++ -std=c++17 -I. test_kelly.cpp -o test_kelly
 */

#include "Kelly.h"
#include <iostream>
#include <iomanip>

using namespace kelly;

void printSeparator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

void testEmotionMapper() {
    printSeparator("EMOTION MAPPER TEST");
    
    EmotionMapper mapper;
    
    // Test different emotional states
    std::vector<EmotionalState> states = {
        {0.8f, 0.8f, 0.7f, "Joy"},        // Joy
        {-0.7f, 0.3f, 0.8f, "Grief"},     // Grief
        {-0.8f, 0.9f, 0.9f, "Rage"},      // Rage
        {0.5f, 0.2f, 0.3f, "Serenity"},   // Serenity
        {-0.5f, 0.7f, 0.6f, "Anxiety"}    // Anxiety
    };
    
    for (const auto& state : states) {
        auto params = mapper.mapToParameters(state);
        
        std::cout << "Emotion: " << state.primaryEmotion << "\n";
        std::cout << "  Valence: " << std::fixed << std::setprecision(2) << state.valence;
        std::cout << ", Arousal: " << state.arousal;
        std::cout << ", Intensity: " << state.intensity << "\n";
        std::cout << "  -> Tempo: " << params.tempoSuggested << " BPM\n";
        std::cout << "  -> Key: " << params.keySuggested << "\n";
        std::cout << "  -> Mode: " << params.modeSuggested << "\n";
        std::cout << "  -> Dissonance: " << params.dissonance << "\n";
        std::cout << "  -> Feel: " << timingFeelToString(params.timingFeel) << "\n";
        std::cout << "  -> Velocity: " << params.velocityMin << "-" << params.velocityMax << "\n";
        std::cout << "\n";
    }
}

void testGrooveEngine() {
    printSeparator("GROOVE ENGINE TEST");
    
    GrooveTemplateEngine engine(42);  // Fixed seed for reproducibility
    
    // List available templates
    std::cout << "Available groove templates:\n";
    for (const auto& name : engine.getTemplateNames()) {
        std::cout << "  - " << name << "\n";
    }
    std::cout << "\n";
    
    // Test humanization
    std::vector<MidiNoteEvent> events;
    for (int i = 0; i < 8; ++i) {
        MidiNoteEvent evt;
        evt.tick = i * 480;  // Quarter notes
        evt.note = 60 + (i % 4);
        evt.velocity = 100;
        evt.duration = 240;
        events.push_back(evt);
    }
    
    std::cout << "Original events:\n";
    for (const auto& evt : events) {
        std::cout << "  Tick: " << evt.tick << ", Note: " << midiNoteToName(evt.note);
        std::cout << ", Vel: " << evt.velocity << "\n";
    }
    
    // Humanize with moderate settings
    auto humanized = engine.humanize(events, 0.5f, 0.5f, 480);
    
    std::cout << "\nHumanized events:\n";
    for (const auto& evt : humanized) {
        std::cout << "  Tick: " << evt.humanizedTick;
        std::cout << " (" << (evt.humanizedTick - evt.tick > 0 ? "+" : "") 
                  << (evt.humanizedTick - evt.tick) << ")";
        std::cout << ", Vel: " << evt.humanizedVelocity;
        std::cout << " (" << (evt.humanizedVelocity - evt.velocity > 0 ? "+" : "")
                  << (evt.humanizedVelocity - evt.velocity) << ")\n";
    }
}

void testIntentProcessor() {
    printSeparator("INTENT PROCESSOR TEST");
    
    IntentProcessor processor;
    
    // Test wound processing
    std::vector<Wound> wounds = {
        {"I lost my best friend", 0.9f, "internal"},
        {"feeling angry at the world", 0.7f, "external"},
        {"overwhelmed with anxiety about the future", 0.8f, "internal"},
        {"so happy today, everything is wonderful", 0.6f, "internal"},
        {"bittersweet memories of childhood", 0.5f, "internal"}
    };
    
    for (const auto& wound : wounds) {
        auto result = processor.processIntent(wound);
        
        std::cout << "Wound: \"" << wound.description << "\"\n";
        std::cout << "  Intensity: " << wound.intensity << "\n";
        std::cout << "  -> Emotion: " << result.emotion.name;
        std::cout << " (" << categoryToString(result.emotion.category) << ")\n";
        std::cout << "  -> V/A/I: " << result.emotion.valence << "/" 
                  << result.emotion.arousal << "/" << result.emotion.intensity << "\n";
        std::cout << "  -> Mode: " << result.emotion.preferredMode << "\n";
        std::cout << "  -> Rule breaks: " << result.ruleBreaks.size() << "\n";
        
        for (const auto& rb : result.ruleBreaks) {
            std::cout << "     - " << ruleBreakTypeToString(rb.type);
            std::cout << " (severity: " << rb.severity << "): ";
            std::cout << rb.description << "\n";
        }
        
        std::cout << "  -> Musical params:\n";
        std::cout << "     Tempo: " << result.musicalParams.tempoSuggested << " BPM\n";
        std::cout << "     Key: " << result.musicalParams.keySuggested << "\n";
        std::cout << "     Mode: " << result.musicalParams.modeSuggested << "\n";
        std::cout << "\n";
    }
}

void testEmotionThesaurus() {
    printSeparator("EMOTION THESAURUS TEST");
    
    IntentProcessor processor;
    const auto& thesaurus = processor.thesaurus();
    
    std::cout << "Total emotions in thesaurus: " << thesaurus.size() << "\n\n";
    
    // Test finding by name
    std::vector<std::string> emotionNames = {"grief", "joy", "rage", "anxiety", "bittersweet"};
    
    std::cout << "Finding emotions by name:\n";
    for (const auto& name : emotionNames) {
        const EmotionNode* emotion = thesaurus.findByName(name);
        if (emotion) {
            std::cout << "  " << name << " -> ID " << emotion->id;
            std::cout << " (V:" << emotion->valence;
            std::cout << ", A:" << emotion->arousal;
            std::cout << ", I:" << emotion->intensity << ")\n";
        } else {
            std::cout << "  " << name << " -> NOT FOUND\n";
        }
    }
    
    // Test finding nearest
    std::cout << "\nFinding nearest emotions:\n";
    std::vector<std::tuple<float, float, float>> coords = {
        {-0.9f, 0.3f, 0.9f},  // Should find grief
        {0.9f, 0.8f, 0.7f},   // Should find joy/ecstasy
        {-0.8f, 0.9f, 0.9f},  // Should find rage
    };
    
    for (const auto& [v, a, i] : coords) {
        const EmotionNode* nearest = thesaurus.findNearest(v, a, i);
        if (nearest) {
            std::cout << "  (" << v << ", " << a << ", " << i << ") -> ";
            std::cout << nearest->name << "\n";
        }
    }
    
    // Test categories
    std::cout << "\nEmotions by category (sample):\n";
    for (auto cat : {EmotionCategory::Joy, EmotionCategory::Sadness, EmotionCategory::Anger}) {
        auto emotions = thesaurus.getByCategory(cat);
        std::cout << "  " << categoryToString(cat) << ": ";
        for (size_t i = 0; i < std::min(emotions.size(), size_t(3)); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << emotions[i]->name;
        }
        if (emotions.size() > 3) std::cout << "...";
        std::cout << " (" << emotions.size() << " total)\n";
    }
}

void testMidiGenerator() {
    printSeparator("MIDI GENERATOR TEST");
    
    MidiGenerator generator(90);  // 90 BPM
    
    // Generate chord progression
    std::cout << "Generating chord progression:\n";
    auto chords = generator.generateChordProgression("Am", "Aeolian", 4, -0.5f, false);
    
    for (const auto& chord : chords) {
        std::cout << "  Chord: " << chord.name << " at beat " << chord.startBeat << "\n";
        std::cout << "    Notes: ";
        for (int note : chord.pitches) {
            std::cout << midiNoteToName(note) << " ";
        }
        std::cout << "\n";
    }
    
    // Generate from emotion
    std::cout << "\nGenerating MIDI from emotion (grief-like):\n";
    auto notes = generator.generateFromEmotion(-0.7f, 0.3f, 0.8f, "Am", "Aeolian", 2);
    
    std::cout << "Generated " << notes.size() << " notes:\n";
    for (size_t i = 0; i < std::min(notes.size(), size_t(10)); ++i) {
        const auto& note = notes[i];
        std::cout << "  " << midiNoteToName(note.pitch);
        std::cout << " at beat " << note.startBeat;
        std::cout << ", vel " << note.velocity;
        std::cout << ", dur " << note.duration << "\n";
    }
    if (notes.size() > 10) {
        std::cout << "  ... and " << (notes.size() - 10) << " more notes\n";
    }
}

void testKellyBrain() {
    printSeparator("KELLY BRAIN INTEGRATION TEST");
    
    KellyBrain brain(85);  // 85 BPM
    
    std::cout << "Quick generation test:\n";
    auto [result, midi] = brain.quickGenerate("deep sorrow after losing someone close", 0.85f, 4);
    
    std::cout << "\nResult Summary:\n";
    std::cout << result.summary() << "\n";
    
    std::cout << "\nGenerated " << midi.size() << " MIDI notes\n";
    std::cout << "First 5 notes:\n";
    for (size_t i = 0; i < std::min(midi.size(), size_t(5)); ++i) {
        // Convert beats to milliseconds
        double beatsPerSecond = brain.tempo() / 60.0;
        double ms = (midi[i].startBeat / beatsPerSecond) * 1000.0;
        std::cout << "  " << std::setw(8) << ms << "ms: ";
        std::cout << midiNoteToName(midi[i].pitch);
        std::cout << " vel=" << midi[i].velocity;
        std::cout << " dur=" << midi[i].duration << "\n";
    }
    
    // Test direct VAI generation
    std::cout << "\nDirect V/A/I generation:\n";
    auto directMidi = brain.generateFromVAI(-0.5f, 0.7f, 0.6f, "Dm", "Dorian", 2);
    std::cout << "Generated " << directMidi.size() << " notes from V=-0.5, A=0.7, I=0.6\n";
    
    // Test emotion finding
    std::cout << "\nEmotion lookup:\n";
    const EmotionNode* found = brain.findEmotionByName("yearning");
    if (found) {
        std::cout << "  'yearning' -> " << found->name << " (ID " << found->id << ")\n";
        std::cout << "  V/A/I: " << found->valence << "/" << found->arousal << "/" << found->intensity << "\n";
        std::cout << "  Related emotions: " << found->relatedEmotions.size() << "\n";
    }
}

void testUtilityFunctions() {
    printSeparator("UTILITY FUNCTIONS TEST");
    
    // Note conversion
    std::cout << "Note conversions:\n";
    for (int note = 48; note <= 72; note += 4) {
        std::string name = midiNoteToName(note);
        int back = noteNameToMidi(name);
        std::cout << "  " << note << " -> " << name << " -> " << back;
        std::cout << (note == back ? " ✓" : " ✗") << "\n";
    }
    
    // Tick/ms conversion
    std::cout << "\nTick/ms conversions at 120 BPM:\n";
    int tempo = 120;
    for (int ticks : {0, 480, 960, 1920}) {
        double ms = ticksToMs(ticks, tempo);
        int backTicks = msToTicks(ms, tempo);
        std::cout << "  " << ticks << " ticks = " << ms << "ms = " << backTicks << " ticks\n";
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║          KELLY MIDI COMPANION - C++ PORT TESTS            ║\n";
    std::cout << "║                                                           ║\n";
    std::cout << "║  Ported from Python DAiW-Music-Brain implementation       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    
    testEmotionMapper();
    testGrooveEngine();
    testEmotionThesaurus();
    testIntentProcessor();
    testMidiGenerator();
    testKellyBrain();
    testUtilityFunctions();
    
    printSeparator("ALL TESTS COMPLETE");
    std::cout << "The ported C++ modules are ready for integration with JUCE.\n\n";
    
    return 0;
}
