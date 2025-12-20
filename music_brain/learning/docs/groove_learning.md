## What this engine does
- Models rhythmic feel, micro-timing, and dynamics that make patterns swing; stabilizes pocket for the whole track.

## Core basics
- Distinguish grid vs. feel: explicit representation for swing/shuffle/humanization.
- Layer hierarchy: kick/snare anchors, hats/perc glue, ghosts for momentum.
- Velocity as phrasing: emphasize downbeats, taper fills, leave contrast for drops.
- Use motif cells (1–2 bars) with variation across phrases to avoid loop fatigue.
- Pocket depends on tempo + genre references; keep consistent swing ratio and accent map.

## Public repos to study
- magenta/magenta — GrooVAE with timing/velocity deviations; probe latent traversals for swing vs. straight feel and how bar/beat embeddings stabilize pocket.
- harritaylor/Neural-Drum-Machine — LSTM groove generator; inspect velocity handling, sampling temperature, and how kick/snare anchors are enforced across bars.
- madmom/madmom — High-quality onset/beat/downbeat tracking; perfect for evaluating generated groove alignment, swing ratio, and micro-timing histograms.
- microsoft/muzic — Beat/rhythm control-token experiments; see bar/beat positional encoding patterns and constraints that preserve ghost notes during decoding.
- julien-c/BeatNet (community) — Real-time beat/downbeat tracking; useful for adaptive humanization and tempo-aligned groove evaluation in live settings.

## Two recursive study questions
- How do different latent traversals affect swing vs. syncopation? Sweep GrooVAE latents and measure swing ratio, backbeat displacement, and velocity variance.
- Can groove survive tempo changes? Time-stretch GrooVAE outputs, re-quantize with madmom beat tracking, and compare micro-timing histograms to originals.

## Advanced techniques to notice
- Timing-offset embeddings per subdivision for learned swing and push/pull feel.
- Velocity-conditioned self-attention to preserve ghost notes during sampling.
- Bar-level groove templates with stochastic micro-variations to avoid loopiness.

## Genre references (50)
- pop, alt pop, synthpop, hyperpop, rock, classic rock, hard rock, punk, pop punk, emo, metal, prog metal, djent, death metal, black metal, blues, jazz, swing, bebop, cool jazz, jazz fusion, bossa nova, samba, salsa, reggaeton, dembow, bachata, merengue, afrobeat, amapiano, highlife, soukous, r&b, neo-soul, gospel, trap, boom bap, drill, lo-fi hip hop, house, deep house, tech house, techno, trance, drum and bass, jungle, dubstep, future bass, uk garage, grime
