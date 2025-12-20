## What this engine does
- Provides harmonic context, cadences, and modulation paths; anchors melody and groove with functional movement.

## Core basics
- Voice-leading first: minimize leaps, keep common tones, respect chordal tendency tones.
- Functional flow: tonic → predominant → dominant → resolution; use applied/secondary functions for motion.
- Manage density: triads vs. 7ths/9ths, open vs. close voicings per register.
- Cadence design: vary authentic/half/plagal to shape section boundaries.
- Modal interchange and borrowed chords for color without losing tonal center.

## Public repos to study
- cuthbertLab/music21 — Deep toolkit for roman numeral analysis, cadence detection, and voice-leading checks; mine corpora for pivot chords, secondary dominants, and contrary motion statistics.
- Ghadjeres/DeepBach — Pseudo-Gibbs sampling with explicit part-wise constraints; great template for constraint-aware decoding and rejection schemes to avoid parallels/doubled leading tones.
- magenta/magenta — MusicVAE chord models and chord-conditioned VAEs/Transformers; examine chord embeddings, functional role labeling, and modulation tagging for smooth key changes.
- microsoft/muzic — PopMAG and Museformer; multi-track harmony conditioning with style/section tokens, grouped attention for long-form progressions, and experiments on reharmonization control.
- sylvielss/midi-chord-recognition (community) — Strong chord/roman numeral tagger; useful for auto-labeling datasets to supervise functional embeddings and cadence classifiers.

## Two recursive study questions
- How does enforcing species-style penalties change neural harmony? Add voice-leading constraint losses to a MusicVAE chord model in magenta and compare error rates on parallels.
- What embeddings best capture functional roles? Train chord/roman numeral embeddings on music21 corpora and test modulation detection vs. simple pitch-class vectors.

## Advanced techniques to notice
- Constraint-aware sampling (pseudo-Gibbs, rejection, or differentiable penalties) for avoiding parallels/doubled leading tones.
- Tonal-pivot detection for smooth modulations and secondary dominants.
- Register-aware voicing (soprano clarity, tenor overlap avoidance) baked into decoding masks.

## Genre references (50)
- pop, alt pop, synthpop, hyperpop, rock, classic rock, hard rock, punk, pop punk, emo, metal, prog metal, djent, death metal, black metal, blues, jazz, swing, bebop, cool jazz, jazz fusion, bossa nova, samba, salsa, reggaeton, dembow, bachata, merengue, afrobeat, amapiano, highlife, soukous, r&b, neo-soul, gospel, trap, boom bap, drill, lo-fi hip hop, house, deep house, tech house, techno, trance, drum and bass, jungle, dubstep, future bass, uk garage, grime
