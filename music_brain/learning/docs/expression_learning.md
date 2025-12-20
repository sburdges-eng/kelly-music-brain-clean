## What this engine does
- Adds humanized dynamics, articulation, and micro-timing; turns MIDI skeletons into performances with feel.

## Core basics
- Velocity curves shape phrasing; avoid flat dynamics, add swells and decays.
- Timing deviations should support groove and rubato, not random jitter.
- Articulation mapping (legato, staccato, accents) must match instrument idioms.
- Automation of CCs (mod wheel, expression, vibrato, pitch bend) carries emotion.
- Keep performer constraints in mind: breath/bow length, string position shifts, hand span.

## Public repos to study
- magenta/ddsp — Differentiable DSP for expressive synthesis; map continuous controls to timbre, study pitch-shift robustness, and inspect how control priors regularize expressive curves.
- YatingMusic/hierper — Hierarchical performance rendering; timing/velocity/dynamics prediction stacked over score structure, with good ablations for phrase-level vs. note-level timing.
- poppea-project/performer (PerformanceRNN forks) — MIDI performance timing/velocity models; explore swing vs. rubato handling and how teacher-forcing impacts humanization at inference.
- Rakhmanin/partitura — Alignment and expressive score tools; ideal for building aligned score–performance pairs and extracting CC curves for supervised learning.
- AI-Expressive-Piano/MAESTRO-perf-study (community) — Research notebooks on MAESTRO dynamics/timing distributions; helpful for setting realistic priors and evaluation metrics on nuance.

## Two recursive study questions
- How much expressiveness comes from timing vs. velocity? Train separate predictors on MAESTRO data using hierper-style architecture and ablate timing during rendering.
- Which CC automation channels matter most per instrument? Analyze ddsp control trajectories vs. subjective ratings across strings/winds/keys to build instrument-specific control priors.

## Advanced techniques to notice
- Continuous-control decoders (CC curves, pitch bend splines) rather than note-only tokens.
- Performer-conditioned models that learn player-specific timing/velocity fingerprints.
- Timbre-dependent humanization: link articulation and micro-timing to instrument envelopes.

## Genre references (50)
- pop, alt pop, synthpop, hyperpop, rock, classic rock, hard rock, punk, pop punk, emo, metal, prog metal, djent, death metal, black metal, blues, jazz, swing, bebop, cool jazz, jazz fusion, bossa nova, samba, salsa, reggaeton, dembow, bachata, merengue, afrobeat, amapiano, highlife, soukous, r&b, neo-soul, gospel, trap, boom bap, drill, lo-fi hip hop, house, deep house, tech house, techno, trance, drum and bass, jungle, dubstep, future bass, uk garage, grime
