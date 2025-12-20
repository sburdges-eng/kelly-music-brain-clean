## What this engine does
- Locks harmony to groove with low-frequency motion; defines tonal gravity, drop impact, and drive.

## Core basics
- Anchor roots on strong beats; outline chord tones on weak beats to clarify harmony.
- Design rhythmic interplay with kick: shared onsets, occasional anticipations, controlled syncopation.
- Register management: avoid mud; use octaves/5ths for weight, passing tones for motion.
- Articulation choice (legato vs. staccato) sets genre feel; shape envelopes to leave kick space.
- Simplify during vocals; add fills for transitions and sectional lifts.

## Public repos to study
- magenta/magenta — PerformanceRNN/ImprovRNN bass lines; study chord/beat conditioning, root-motion heuristics, and penalties for over-syncopation when kick anchors are present.
- microsoft/muzic — PopMAG multi-track generation; inspect how bass tracks respond to chord/beat/style tokens and how Museformer’s grouped attention keeps low-end motifs stable.
- spotify/basic-pitch — Solid low-frequency pitch extraction; use to auto-label bass stems, filter by confidence, and build clean training corpora across genres.
- Ghadjeres/DeepBach (adapted) — Pseudo-Gibbs sampling ideas for smooth stepwise motion and controlled leaps; adapt constraints to keep bass within register and avoid mud.
- teropa/drum-rnn-bassline-fork (community) — Lightweight LSTM bassline generator conditioned on drum hits; good for fast kick/bass interplay experiments and ablation of sidechain-aware masks.

## Two recursive study questions
- How does kick/bass coupling affect perceived groove? Train a bass generator conditioned on kick onsets using magenta data and ablate the conditioning during sampling.
- Which scale degrees can move without losing weight? Analyze PopMAG bass outputs by position (1/5/♭7/9) and test listener preference across genres.

## Advanced techniques to notice
- Sidechain-aware decoding: mask notes that collide with kick transients to leave headroom.
- Beat-synchronous latent variables so fills land on barlines while groove stays stable.
- Chord-function-aware embeddings (tonic/dominant/subdominant) to bias root motion choices.

## Genre references (50)
- pop, alt pop, synthpop, hyperpop, rock, classic rock, hard rock, punk, pop punk, emo, metal, prog metal, djent, death metal, black metal, blues, jazz, swing, bebop, cool jazz, jazz fusion, bossa nova, samba, salsa, reggaeton, dembow, bachata, merengue, afrobeat, amapiano, highlife, soukous, r&b, neo-soul, gospel, trap, boom bap, drill, lo-fi hip hop, house, deep house, tech house, techno, trance, drum and bass, jungle, dubstep, future bass, uk garage, grime
