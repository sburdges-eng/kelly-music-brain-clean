# DAW UI Design Patterns Guide

A reference for modern DAW user interface patterns using React/TypeScript, featuring creative metaphors, AI-assisted workflows, and real-time audio monitoring.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [The Dual-Face Metaphor](#the-dual-face-metaphor)
3. [Component Architecture](#component-architecture)
4. [AI-Assisted Creative Tools](#ai-assisted-creative-tools)
5. [MIDI Routing Architecture](#midi-routing-architecture)
6. [Real-Time Monitoring](#real-time-monitoring)
7. [CSS & Animation Patterns](#css--animation-patterns)
8. [Implementation Guidelines](#implementation-guidelines)

---

## Design Philosophy

### Core Principles

1. **Physical Metaphors**: Use familiar physical objects (cassette tapes, patch bays, VU meters) to ground digital interfaces
2. **Dual Modes**: Separate creative/emotional workflows from technical/engineering workflows
3. **AI as Ghost**: AI suggestions appear as ethereal overlays, never overwriting user content
4. **Latency Awareness**: Always surface audio timing information to prevent drift
5. **Interrogate Before Generate**: Ask clarifying questions before AI generation

### User Experience Goals

- **Side A (Creative)**: Warm, vintage, inviting—bone white, soft edges, lyric-focused
- **Side B (Technical)**: Dark, precise, engineering—matrix green, hard data, MIDI-focused
- **Transition**: Satisfying 3D flip animation that signals mode change

---

## The Dual-Face Metaphor

### Concept

A cassette tape that flips between two distinct interfaces:

| Aspect | Side A (Creative) | Side B (Technical) |
|--------|-------------------|-------------------|
| Purpose | Writing, vibes, emotion | Patching, routing, timing |
| Visual | Warm cream/bone white | Dark with matrix green |
| Tools | Ghost Writer, Interrogator | MIDI Patch Bay, Latency Monitor |
| Mindset | "What does it feel like?" | "How does it connect?" |

### React Implementation

```tsx
// Core component structure
interface TapeDeckProps {
  projectLabel?: string;
  initialSide?: 'A' | 'B';
}

const TapeDeck: React.FC<TapeDeckProps> = ({ 
  projectLabel = "Untitled Project 01",
  initialSide = 'A'
}) => {
  const [isSideB, setIsSideB] = useState(initialSide === 'B');
  const [cassetteLabel, setCassetteLabel] = useState(projectLabel);

  const handleFlip = () => {
    setIsSideB(!isSideB);
    // Optional: Trigger haptic feedback or sound effect
  };

  return (
    <div className="perspective-container">
      <div className={`cassette-body ${isSideB ? 'flipped' : ''}`}>
        
        {/* SIDE A: CREATIVE MODE */}
        <div className="face side-a">
          <div className="screw-hole top-left" />
          <div className="screw-hole top-right" />
          <div className="label-area">
            <h3>SIDE A: {cassetteLabel}</h3>
            <GhostWriterOverlay />
            <CreativeInterrogator />
          </div>
          <button className="flip-btn" onClick={handleFlip}>
            ↻ FLIP TAPE (Edit Logic)
          </button>
        </div>

        {/* SIDE B: TECHNICAL MODE */}
        <div className="face side-b">
          <div className="label-area dark-mode">
            <h3>SIDE B: LOGIC & PATCHING</h3>
            <MidiPatchBay />
            <LatencyWatchdog />
          </div>
          <button className="flip-btn" onClick={handleFlip}>
            ↻ FLIP TAPE (Edit Vibe)
          </button>
        </div>

      </div>
    </div>
  );
};
```

---

## Component Architecture

### Directory Structure

```
src/
├── components/
│   ├── TapeDeck/
│   │   ├── TapeDeck.tsx
│   │   ├── TapeDeck.css
│   │   └── index.ts
│   ├── Creative/
│   │   ├── GhostWriterOverlay.tsx
│   │   ├── CreativeInterrogator.tsx
│   │   └── MoodSelector.tsx
│   └── Technical/
│       ├── MidiPatchBay.tsx
│       ├── LatencyWatchdog.tsx
│       └── ChannelRouter.tsx
├── engine/
│   ├── MidiManager.ts
│   ├── AudioEngine.ts
│   └── LatencyMonitor.ts
└── styles/
    ├── variables.css
    └── animations.css
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `TapeDeck` | Container, flip state, project label |
| `GhostWriterOverlay` | AI lyric/chord suggestions |
| `CreativeInterrogator` | Pre-generation questioning |
| `MidiPatchBay` | Visual MIDI routing |
| `LatencyWatchdog` | Real-time timing display |

---

## AI-Assisted Creative Tools

### Ghost Writer Pattern

The Ghost Writer shows AI suggestions as semi-transparent overlay text that appears after the user stops typing. Key principles:

1. **Never overwrite**: Suggestions appear beside/below user text
2. **Fade in**: Use opacity transitions for gentle appearance
3. **Easy dismiss**: Any keystroke clears the suggestion
4. **Latency trigger**: Wait 1-2 seconds of inactivity before suggesting

```tsx
interface GhostWriterProps {
  suggestionDelay?: number; // ms before AI suggestion appears
  ghostOpacity?: number;    // 0.0 - 1.0
}

const GhostWriterOverlay: React.FC<GhostWriterProps> = ({
  suggestionDelay = 1500,
  ghostOpacity = 0.4
}) => {
  const [userText, setUserText] = useState('');
  const [ghostSuggestion, setGhostSuggestion] = useState('');

  useEffect(() => {
    const timer = setTimeout(() => {
      if (userText.length > 5) {
        // Call AI service for continuation
        fetchAISuggestion(userText).then(setGhostSuggestion);
      }
    }, suggestionDelay);
    
    return () => clearTimeout(timer);
  }, [userText, suggestionDelay]);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setUserText(e.target.value);
    setGhostSuggestion(''); // Clear ghost on any input
  };

  return (
    <div className="editor-container">
      {/* Ghost Layer (Behind) */}
      <div className="ghost-layer" style={{ opacity: ghostOpacity }}>
        {userText}
        <span className="ghost-text">{ghostSuggestion}</span>
      </div>
      
      {/* Real Input (Front) */}
      <textarea 
        value={userText} 
        onChange={handleChange}
        placeholder="Start writing..."
      />
    </div>
  );
};
```

### Interrogate Before Generate Pattern

Before AI generates content, ask clarifying questions to improve output quality:

```tsx
type InterrogationPhase = 'INIT' | 'TEXTURE' | 'MOOD' | 'TEMPO' | 'GENERATING';

interface InterrogatorState {
  phase: InterrogationPhase;
  texture: string;   // "gritty", "underwater", "crystalline"
  mood: string;      // "melancholic", "triumphant", "anxious"
  tempo: string;     // "languid", "driving", "erratic"
}

const PHASE_PROMPTS: Record<InterrogationPhase, string> = {
  INIT: "What is the texture? (e.g., Gritty, underwater, crystalline...)",
  TEXTURE: "What is the emotional color? (e.g., Melancholic, triumphant...)",
  MOOD: "What is the energy? (e.g., Languid, driving, erratic...)",
  TEMPO: "Processing your vision...",
  GENERATING: ""
};

const CreativeInterrogator: React.FC = () => {
  const [state, setState] = useState<InterrogatorState>({
    phase: 'INIT',
    texture: '',
    mood: '',
    tempo: ''
  });
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      advancePhase(input);
      setInput('');
    }
  };

  const advancePhase = (value: string) => {
    switch (state.phase) {
      case 'INIT':
        setState(s => ({ ...s, texture: value, phase: 'TEXTURE' }));
        break;
      case 'TEXTURE':
        setState(s => ({ ...s, mood: value, phase: 'MOOD' }));
        break;
      case 'MOOD':
        setState(s => ({ ...s, tempo: value, phase: 'TEMPO' }));
        // Trigger generation with collected parameters
        triggerGeneration(state);
        break;
    }
  };

  return (
    <div className="interrogator-terminal">
      <span className="prompt-cursor">{">"}</span>
      <p>{PHASE_PROMPTS[state.phase]}</p>
      <input 
        type="text"
        className="invisible-input"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleSubmit}
        autoFocus
      />
    </div>
  );
};
```

### Mapping Mood to Music Theory

```typescript
interface TheoryMapping {
  scales: string[];
  intervals: string[];
  chordTypes: string[];
  tempo: [number, number]; // BPM range
}

const MOOD_THEORY_MAP: Record<string, TheoryMapping> = {
  sad: {
    scales: ['natural_minor', 'dorian', 'phrygian'],
    intervals: ['minor_2nd', 'minor_9th', 'tritone'],
    chordTypes: ['min7', 'min9', 'dim'],
    tempo: [50, 80]
  },
  triumphant: {
    scales: ['major', 'lydian', 'mixolydian'],
    intervals: ['perfect_5th', 'major_3rd', 'octave'],
    chordTypes: ['maj', 'sus4', 'add9'],
    tempo: [100, 140]
  },
  anxious: {
    scales: ['harmonic_minor', 'locrian', 'whole_tone'],
    intervals: ['minor_2nd', 'tritone', 'minor_7th'],
    chordTypes: ['dim7', 'aug', 'min(maj7)'],
    tempo: [120, 180]
  },
  ethereal: {
    scales: ['lydian', 'whole_tone', 'pentatonic_major'],
    intervals: ['perfect_4th', 'major_7th', 'perfect_5th'],
    chordTypes: ['maj7', 'add9', 'sus2'],
    tempo: [60, 90]
  }
};
```

---

## MIDI Routing Architecture

### IMIDI vs EMIDI Separation

Distinguish between internal (software) and external (hardware) MIDI:

```typescript
type MidiChannelType = 'IMIDI' | 'EMIDI';

interface MidiChannel {
  id: string;
  name: string;
  type: MidiChannelType;
  port?: MIDIOutput; // Only for EMIDI
}

interface NoteData {
  note: number;      // 0-127
  velocity: number;  // 0-127
  channel: number;   // 1-16
  duration?: number; // ms
}

class MidiManager {
  private imidiChannels: Map<string, MidiChannel> = new Map();
  private emidiChannels: Map<string, MidiChannel> = new Map();
  private midiAccess: MIDIAccess | null = null;

  /**
   * Initialize external MIDI hardware detection
   */
  async initExternal(): Promise<void> {
    if (!navigator.requestMIDIAccess) {
      console.warn('WebMIDI not supported in this browser');
      return;
    }

    try {
      this.midiAccess = await navigator.requestMIDIAccess({ sysex: false });
      
      this.midiAccess.outputs.forEach((output) => {
        this.emidiChannels.set(output.id, {
          id: output.id,
          name: output.name || 'Unknown Device',
          type: 'EMIDI',
          port: output
        });
        console.log(`EMIDI DETECTED: ${output.name}`);
      });

      // Listen for hot-plugged devices
      this.midiAccess.onstatechange = this.handleMidiStateChange.bind(this);
    } catch (err) {
      console.error('Failed to access MIDI devices:', err);
    }
  }

  /**
   * Register internal virtual instruments
   */
  registerInternal(id: string, name: string): void {
    this.imidiChannels.set(id, {
      id,
      name,
      type: 'IMIDI'
    });
    console.log(`IMIDI REGISTERED: ${name}`);
  }

  /**
   * Route note data to appropriate destination
   */
  routeSignal(noteData: NoteData, targetId: string): void {
    const imidiTarget = this.imidiChannels.get(targetId);
    const emidiTarget = this.emidiChannels.get(targetId);

    if (imidiTarget) {
      this.triggerInternalSynth(noteData, imidiTarget);
    } else if (emidiTarget) {
      this.triggerExternalHardware(noteData, emidiTarget);
    } else {
      console.warn(`Unknown MIDI target: ${targetId}`);
    }
  }

  private triggerInternalSynth(note: NoteData, channel: MidiChannel): void {
    // Dispatch to internal synth engine (Web Audio API, Tone.js, etc.)
    console.log(`IMIDI: Playing note ${note.note} on ${channel.name}`);
    window.dispatchEvent(new CustomEvent('imidi-note', { detail: { note, channel } }));
  }

  private triggerExternalHardware(note: NoteData, channel: MidiChannel): void {
    if (!channel.port) return;

    const noteOn = [0x90 | (note.channel - 1), note.note, note.velocity];
    channel.port.send(noteOn);
    
    if (note.duration) {
      setTimeout(() => {
        const noteOff = [0x80 | (note.channel - 1), note.note, 0];
        channel.port?.send(noteOff);
      }, note.duration);
    }
  }

  private handleMidiStateChange(event: MIDIConnectionEvent): void {
    const port = event.port;
    if (port.type === 'output') {
      if (port.state === 'connected') {
        this.emidiChannels.set(port.id, {
          id: port.id,
          name: port.name || 'Unknown',
          type: 'EMIDI',
          port: port as MIDIOutput
        });
      } else {
        this.emidiChannels.delete(port.id);
      }
    }
  }
}

// Singleton instance
export const midiManager = new MidiManager();
```

---

## Real-Time Monitoring

### Latency Watchdog Component

Monitor main thread blocking and audio buffer health:

```tsx
type SyncStatus = 'LOCKED' | 'DRIFT_DETECTED' | 'CRITICAL' | 'OFFLINE';

interface LatencyWatchdogProps {
  warningThreshold?: number;  // ms
  criticalThreshold?: number; // ms
  sampleInterval?: number;    // ms
}

const LatencyWatchdog: React.FC<LatencyWatchdogProps> = ({
  warningThreshold = 25,
  criticalThreshold = 50,
  sampleInterval = 1000
}) => {
  const [latency, setLatency] = useState<number>(0);
  const [status, setStatus] = useState<SyncStatus>('LOCKED');
  const [history, setHistory] = useState<number[]>([]);

  useEffect(() => {
    let lastFrameTime = performance.now();
    let frameCount = 0;
    let accumulatedJitter = 0;

    const measureFrame = () => {
      const now = performance.now();
      const frameDelta = now - lastFrameTime;
      const expectedDelta = 16.67; // 60fps
      const jitter = Math.abs(frameDelta - expectedDelta);
      
      accumulatedJitter += jitter;
      frameCount++;
      lastFrameTime = now;
      
      requestAnimationFrame(measureFrame);
    };

    const reportInterval = setInterval(() => {
      if (frameCount > 0) {
        const avgJitter = accumulatedJitter / frameCount;
        const calculatedLatency = avgJitter + 5; // base audio buffer
        
        setLatency(Number(calculatedLatency.toFixed(2)));
        setHistory(h => [...h.slice(-59), calculatedLatency]);
        
        if (calculatedLatency > criticalThreshold) {
          setStatus('CRITICAL');
        } else if (calculatedLatency > warningThreshold) {
          setStatus('DRIFT_DETECTED');
        } else {
          setStatus('LOCKED');
        }
        
        accumulatedJitter = 0;
        frameCount = 0;
      }
    }, sampleInterval);

    const frameId = requestAnimationFrame(measureFrame);

    return () => {
      cancelAnimationFrame(frameId);
      clearInterval(reportInterval);
    };
  }, [warningThreshold, criticalThreshold, sampleInterval]);

  const handleResync = async () => {
    // Trigger audio engine resync
    window.dispatchEvent(new CustomEvent('audio-resync'));
    setStatus('LOCKED');
  };

  return (
    <div className={`latency-panel ${status.toLowerCase().replace('_', '-')}`}>
      <h4>SYSTEM CLOCK</h4>
      
      <div className="led-display">
        <span className="ms-count">{latency}ms</span>
        <span className={`status-indicator ${status.toLowerCase()}`} />
        <span className="status-text">AUDIO SYNC: {status.replace('_', ' ')}</span>
      </div>

      {/* Mini histogram */}
      <div className="latency-histogram">
        {history.map((h, i) => (
          <div 
            key={i} 
            className="histogram-bar"
            style={{ 
              height: `${Math.min(h * 2, 100)}%`,
              backgroundColor: h > warningThreshold ? '#ff4444' : '#00ff00'
            }}
          />
        ))}
      </div>

      {status !== 'LOCKED' && (
        <button className="panic-btn" onClick={handleResync}>
          ⚡ RESYNC ENGINE
        </button>
      )}
    </div>
  );
};
```

---

## CSS & Animation Patterns

### Core Styles

```css
/* Variables */
:root {
  --side-a-bg: #f4f1ea;      /* Bone white / Vintage */
  --side-a-text: #333;
  --side-b-bg: #1a1a1a;      /* Technical Dark */
  --side-b-text: #00ff00;    /* Matrix Green */
  --flip-duration: 0.8s;
  --flip-timing: cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

/* 3D Container */
.perspective-container {
  perspective: 1000px;
  width: 800px;
  height: 500px;
  margin: 0 auto;
}

/* Cassette Body */
.cassette-body {
  position: relative;
  width: 100%;
  height: 100%;
  transition: transform var(--flip-duration) var(--flip-timing);
  transform-style: preserve-3d;
}

.cassette-body.flipped {
  transform: rotateY(180deg);
}

/* Faces */
.face {
  position: absolute;
  width: 100%;
  height: 100%;
  backface-visibility: hidden;
  border-radius: 15px;
  padding: 20px;
  box-shadow: 
    0 10px 30px rgba(0, 0, 0, 0.2),
    inset 0 1px 1px rgba(255, 255, 255, 0.1);
}

/* Side A: Creative Mode */
.side-a {
  background: var(--side-a-bg);
  color: var(--side-a-text);
  background-image: 
    linear-gradient(135deg, transparent 40%, rgba(0,0,0,0.02) 40%);
}

/* Side B: Technical Mode */
.side-b {
  background: var(--side-b-bg);
  color: var(--side-b-text);
  transform: rotateY(180deg);
  border: 2px solid #444;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* Decorative Screw Holes */
.screw-hole {
  position: absolute;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: 
    radial-gradient(circle at 30% 30%, #ddd, #888);
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
}

.screw-hole.top-left { top: 10px; left: 10px; }
.screw-hole.top-right { top: 10px; right: 10px; }

/* Ghost Writer Styles */
.editor-container {
  position: relative;
  width: 100%;
  min-height: 200px;
}

.ghost-layer {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  padding: 12px;
  pointer-events: none;
  font-size: 16px;
  line-height: 1.6;
  white-space: pre-wrap;
}

.ghost-text {
  opacity: 0.4;
  color: #666;
  font-style: italic;
  animation: ghost-fade-in 0.5s ease-out;
}

@keyframes ghost-fade-in {
  from { opacity: 0; }
  to { opacity: 0.4; }
}

.editor-container textarea {
  position: relative;
  width: 100%;
  min-height: 200px;
  background: transparent;
  border: none;
  font-size: 16px;
  line-height: 1.6;
  resize: vertical;
  padding: 12px;
}

/* Interrogator Terminal */
.interrogator-terminal {
  background: #000;
  padding: 15px;
  border-radius: 8px;
  font-family: monospace;
}

.prompt-cursor {
  color: #00ff00;
  margin-right: 8px;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  50% { opacity: 0; }
}

.invisible-input {
  background: transparent;
  border: none;
  color: #00ff00;
  font-family: inherit;
  font-size: inherit;
  outline: none;
  width: 80%;
}

/* Latency Panel */
.latency-panel {
  background: #111;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 15px;
}

.latency-panel.drift-detected {
  border-color: #ff8800;
  animation: pulse-warning 2s infinite;
}

.latency-panel.critical {
  border-color: #ff0000;
  animation: pulse-critical 0.5s infinite;
}

@keyframes pulse-warning {
  0%, 100% { box-shadow: 0 0 5px #ff8800; }
  50% { box-shadow: 0 0 20px #ff8800; }
}

@keyframes pulse-critical {
  0%, 100% { box-shadow: 0 0 10px #ff0000; }
  50% { box-shadow: 0 0 30px #ff0000; }
}

.led-display {
  display: flex;
  align-items: center;
  gap: 15px;
  font-family: 'Digital-7', monospace;
}

.ms-count {
  font-size: 32px;
  color: #00ff00;
  text-shadow: 0 0 10px #00ff00;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.status-indicator.locked {
  background: #00ff00;
  box-shadow: 0 0 8px #00ff00;
}

.status-indicator.drift_detected {
  background: #ff8800;
  box-shadow: 0 0 8px #ff8800;
}

.status-indicator.critical {
  background: #ff0000;
  box-shadow: 0 0 8px #ff0000;
  animation: blink 0.25s infinite;
}

.latency-histogram {
  display: flex;
  align-items: flex-end;
  height: 40px;
  gap: 1px;
  margin-top: 10px;
}

.histogram-bar {
  flex: 1;
  min-width: 2px;
  transition: height 0.1s ease-out;
}

.panic-btn {
  margin-top: 10px;
  padding: 8px 16px;
  background: #ff4444;
  border: none;
  color: white;
  font-weight: bold;
  cursor: pointer;
  border-radius: 4px;
  animation: pulse-critical 1s infinite;
}

/* Flip Button */
.flip-btn {
  position: absolute;
  bottom: 20px;
  right: 20px;
  padding: 10px 20px;
  border: 2px solid currentColor;
  background: transparent;
  color: inherit;
  cursor: pointer;
  border-radius: 25px;
  font-weight: bold;
  transition: all 0.2s;
}

.flip-btn:hover {
  transform: scale(1.05);
}

.side-a .flip-btn:hover {
  background: rgba(0, 0, 0, 0.1);
}

.side-b .flip-btn:hover {
  background: rgba(0, 255, 0, 0.1);
  text-shadow: 0 0 10px #00ff00;
}
```

---

## Implementation Guidelines

### State Management Recommendations

1. **Local State**: Flip state, input values, latency readings
2. **Context**: Project settings, MIDI manager instance
3. **External Store**: Song data, AI generation history, undo stack

### Performance Considerations

1. **Latency Monitoring**: Use `requestAnimationFrame` instead of `setInterval` for accurate timing
2. **Ghost Writer Debounce**: Don't call AI on every keystroke—wait for pause
3. **MIDI Event Batching**: Group rapid note events to prevent UI thrashing
4. **CSS Will-Change**: Add `will-change: transform` to flip container

### Accessibility

1. **Keyboard Navigation**: Tab through flip, inputs, buttons
2. **Screen Reader**: Announce current side, latency alerts
3. **Reduced Motion**: Respect `prefers-reduced-motion` for flip animation
4. **Color Contrast**: Ensure matrix green text meets WCAG AA

### Testing Strategy

```typescript
// Component tests
describe('TapeDeck', () => {
  it('flips to Side B when flip button clicked', () => {
    render(<TapeDeck />);
    fireEvent.click(screen.getByText(/flip tape/i));
    expect(screen.getByText(/side b/i)).toBeVisible();
  });
});

describe('GhostWriterOverlay', () => {
  it('shows suggestion after typing pause', async () => {
    render(<GhostWriterOverlay suggestionDelay={100} />);
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Hello world' } });
    await waitFor(() => {
      expect(screen.getByText(/ghost suggestion/i)).toBeInTheDocument();
    }, { timeout: 200 });
  });
});

describe('LatencyWatchdog', () => {
  it('shows DRIFT status when latency exceeds threshold', async () => {
    // Mock performance.now to simulate lag
    jest.spyOn(performance, 'now').mockReturnValue(Date.now() + 50);
    render(<LatencyWatchdog warningThreshold={20} />);
    await waitFor(() => {
      expect(screen.getByText(/drift/i)).toBeInTheDocument();
    });
  });
});
```

---

## Resources

### Libraries

- **React Three Fiber**: For more complex 3D interactions
- **Framer Motion**: Declarative animations
- **Tone.js**: Web Audio synthesis for IMIDI
- **WebMidi.js**: Simplified WebMIDI API

### Fonts

- **JetBrains Mono**: Technical displays
- **Digital-7**: LED/LCD number displays
- **Inter**: UI labels

### References

- [WebMIDI API Specification](https://www.w3.org/TR/webmidi/)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [CSS 3D Transforms](https://developer.mozilla.org/en-US/docs/Web/CSS/transform-function/rotateY)

---

*This guide provides design patterns for building modern DAW interfaces. Adapt component names, visual styles, and functionality to match your specific product requirements.*
