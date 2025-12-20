# Emotion Wheel & 216-Node Thesaurus Guide

miDiKompanion uses a sophisticated 216-node emotion thesaurus to map emotional states to musical characteristics. This guide explains how the system works.

## Overview

The emotion system is organized hierarchically:
- **6 Base Emotions**: Primary emotional categories
- **36 Sub-Emotions**: 6 variations per base emotion  
- **216 Nodes**: 6 intensity levels per sub-emotion
- **1296 Intensity Tiers**: Fine-grained intensity variations

## The VAD Model

Each emotion node is positioned in 3D emotional space using the **VAD model**:

### Valence (Pleasure)
- **Range**: -1.0 (negative) to +1.0 (positive)
- **Negative**: Sadness, fear, anger, disgust
- **Positive**: Joy, excitement, contentment

### Arousal (Activation)
- **Range**: 0.0 (calm) to 1.0 (excited)
- **Low**: Relaxed, sleepy, calm
- **High**: Alert, excited, agitated

### Dominance (Control)
- **Range**: 0.0 (submissive) to 1.0 (dominant)
- **Low**: Vulnerable, helpless
- **High**: Confident, in control

## Base Emotion Categories

### 1. Happy (Joy)
**VAD Profile**: High Valence, Variable Arousal, High Dominance

Sub-emotions:
- Cheerful: Light, playful joy
- Content: Peaceful satisfaction
- Ecstatic: Intense happiness
- Hopeful: Optimistic anticipation
- Proud: Achievement satisfaction
- Relieved: Released tension

**Musical Mapping**:
- Major keys
- Bright timbres
- Upward melodic contours
- Moderate to fast tempo

### 2. Sad (Sorrow)
**VAD Profile**: Low Valence, Low Arousal, Low Dominance

Sub-emotions:
- Melancholic: Reflective sadness
- Grieving: Deep loss
- Lonely: Isolation
- Disappointed: Unmet expectations
- Regretful: Past-focused sorrow
- Helpless: Powerless sadness

**Musical Mapping**:
- Minor keys
- Muted timbres
- Descending contours
- Slow tempo

### 3. Angry (Rage)
**VAD Profile**: Low Valence, High Arousal, High Dominance

Sub-emotions:
- Irritated: Mild frustration
- Frustrated: Blocked goals
- Furious: Intense anger
- Resentful: Lingering bitterness
- Hostile: Aggressive anger
- Contemptuous: Superior anger

**Musical Mapping**:
- Dissonant harmonies
- Accented rhythms
- Angular melodies
- Fast tempo

### 4. Fear (Anxiety)
**VAD Profile**: Low Valence, High Arousal, Low Dominance

Sub-emotions:
- Nervous: Mild anxiety
- Worried: Anticipatory fear
- Terrified: Intense fear
- Insecure: Self-doubt
- Panicked: Overwhelming fear
- Dread: Dark anticipation

**Musical Mapping**:
- Chromatic passages
- Unstable harmonies
- Irregular rhythms
- Tense dynamics

### 5. Surprise
**VAD Profile**: Variable Valence, High Arousal, Variable Dominance

Sub-emotions:
- Amazed: Positive wonder
- Confused: Disorientation
- Shocked: Sudden impact
- Startled: Brief surprise
- Astonished: Overwhelming surprise
- Curious: Interested surprise

**Musical Mapping**:
- Sudden dynamic changes
- Unexpected harmonic moves
- Rhythmic disruptions
- Wide intervals

### 6. Disgust
**VAD Profile**: Low Valence, Variable Arousal, Variable Dominance

Sub-emotions:
- Repulsed: Physical disgust
- Revolted: Strong rejection
- Nauseated: Visceral disgust
- Uncomfortable: Mild unease
- Disapproving: Moral disgust
- Aversion: Avoidance

**Musical Mapping**:
- Discordant intervals
- Unusual timbres
- Irregular patterns
- Unsettling textures

## Musical Attribute Mapping

### Pitch & Melody
| Emotion | Contour | Range | Interval Preference |
|---------|---------|-------|---------------------|
| Happy | Ascending | Wide | Major 3rds, 5ths |
| Sad | Descending | Narrow | Minor 2nds, 3rds |
| Angry | Angular | Wide | Tritones, 7ths |
| Fear | Uncertain | Variable | Chromatic |
| Surprise | Leaping | Very wide | Octaves, unusual |

### Rhythm & Tempo
| Emotion | Tempo | Rhythm | Articulation |
|---------|-------|--------|--------------|
| Happy | Fast | Regular | Staccato/Legato mix |
| Sad | Slow | Sustained | Legato |
| Angry | Fast | Driving | Accented |
| Fear | Variable | Irregular | Tremolo |
| Calm | Very slow | Flowing | Sustained |

### Harmony
| Emotion | Mode | Tension | Chord Types |
|---------|------|---------|-------------|
| Happy | Major | Low | Triads, 6ths |
| Sad | Minor | Medium | Minor 7ths |
| Angry | Mixed | High | Diminished, Aug |
| Fear | Chromatic | High | Cluster, sus4 |
| Peaceful | Major | Very low | Open voicings |

### Dynamics
| Intensity | Velocity Range | Dynamic Curve |
|-----------|----------------|---------------|
| 0.0-0.2 | 30-60 | Very soft, consistent |
| 0.2-0.4 | 50-80 | Soft, gentle swells |
| 0.4-0.6 | 60-100 | Medium, balanced |
| 0.6-0.8 | 80-115 | Loud, expressive |
| 0.8-1.0 | 100-127 | Very loud, dramatic |

## Using the Emotion System

### Direct Node Selection
1. Choose base emotion from dropdown
2. Fine-tune with VAD sliders
3. The system finds the closest matching node

### Emotion Blending
When VAD values are between nodes:
- System interpolates between nearby nodes
- Musical parameters are blended proportionally
- Smooth transitions between emotional states

### Emotional Journeys
Create progression by:
1. Setting initial emotion
2. Automating VAD parameters
3. Recording the generated MIDI
4. Editing to create narrative arc

## Node Relationships

Each node knows its neighbors:
- **Related Emotions**: Similar emotional states
- **Contrasting Emotions**: Opposite states
- **Transitional Emotions**: Natural progressions

## ML Enhancement & Node Mapping

- **ML Toggle**: Enable hybrid mode to blend ML embeddings with the 216-node structure; disable for pure rule-based behavior.
- **NodeMLMapper**: Converts 64-dim ML embeddings to the nearest VAD node and attaches confidence + related-node context.
- **Hybrid Generation**: ML suggests a node, the thesaurus validates it, and musical attributes (mode/tempo/dynamics) still come from the node.
- **Fallback**: If ML is unavailable, the mapper uses VAD sliders and node relationships only.

This enables:
- Smooth emotional transitions
- Coherent musical progressions
- Meaningful harmonic movement

## Examples

### Creating Tension → Release
1. Start: Fear (Valence: -0.6, Arousal: 0.8)
2. Build: Anger (Valence: -0.4, Arousal: 0.9)
3. Peak: Surprise (Valence: 0.0, Arousal: 1.0)
4. Release: Relief (Valence: 0.6, Arousal: 0.3)

### Melancholic Journey
1. Start: Sad (Valence: -0.5, Arousal: 0.2)
2. Deepen: Grieving (Valence: -0.8, Arousal: 0.1)
3. Reflect: Melancholic (Valence: -0.4, Arousal: 0.3)
4. Hope: Hopeful (Valence: 0.3, Arousal: 0.4)

### Energy Build
1. Start: Calm (Arousal: 0.1)
2. Build: Curious (Arousal: 0.4)
3. Rise: Excited (Arousal: 0.7)
4. Peak: Ecstatic (Arousal: 1.0)

## ML Enhancement (Optional)

When ML enhancement is enabled:
- Audio input is analyzed for emotional content
- 64-dimensional emotion embedding is extracted
- Nearest node is found in VAD space
- ML models enhance generation with learned patterns

The ML system works alongside the node structure:
- Rule-based: Uses node musical attributes
- ML-enhanced: Adds learned patterns and variations
- Hybrid: Combines both for best results

---

For more details on using these emotions in practice, see the [User Guide](USER_GUIDE.md).
