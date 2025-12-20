## What this engine does
- Shapes song arcs, section contrast, and instrument entrances; controls energy, tension, and listener attention over time.

## Core basics
- Define a clear sectional map (intro/verse/pre/chorus/bridge/outro) with energy targets.
- Use layering discipline: add one element at a time; keep focal hierarchy per section.
- Contrast tools: density, register spread, harmony changes, rhythmic texture, and timbre swaps.
- Transitions need cues (riser, fill, harmonic pivot, filter sweep) and pre-landing space.
- Leave headroom early; reserve peak elements (full drums, doubles, top harmonics) for climaxes.

## Public repos to study
- microsoft/muzic — Arranger and PopMAG; inspect section/structure tokens, instrument scheduling policies, and how energy curves are controlled across verses/choruses.
- magenta/magenta — Hierarchical MusicVAE decoders; phrase/bar latents and interpolation tricks for non-looping sections and controlled development.
- facebookresearch/audiocraft — MusicGen long-context demos; examine prompt engineering for macro arcs, text+melody conditioning, and how windowed attention copes with repeats.
- muspy/muspy — Structural feature extraction and dataset utilities; use to label sections, boundaries, and transitions for supervised arrangement learners.
- jukedeck/ai-arranger-experiments (community) — Practical heuristics for entrance/exit scheduling, mute/solo policies, and density envelopes; good baseline to compare learned policies.

## Two recursive study questions
- How effective are section tokens at preventing loopiness? Train a hierarchical VAE (magenta) with bar/section latents and A/B test repetition rate vs. flat baselines.
- Which transition cues best predict perceived smoothness? Label riser/fill types in muspy datasets, train a classifier, and correlate cue presence with listener ratings.

## Advanced techniques to notice
- Hierarchical latents (beat → bar → phrase → section) to keep long-range structure stable.
- Energy-curve conditioning using loudness/entropy trajectories to steer builds and drops.
- Sparse-instrument scheduling policies to avoid overcrowding while keeping momentum.

## Genre references (50)
- pop, alt pop, synthpop, hyperpop, rock, classic rock, hard rock, punk, pop punk, emo, metal, prog metal, djent, death metal, black metal, blues, jazz, swing, bebop, cool jazz, jazz fusion, bossa nova, samba, salsa, reggaeton, dembow, bachata, merengue, afrobeat, amapiano, highlife, soukous, r&b, neo-soul, gospel, trap, boom bap, drill, lo-fi hip hop, house, deep house, tech house, techno, trance, drum and bass, jungle, dubstep, future bass, uk garage, grime
