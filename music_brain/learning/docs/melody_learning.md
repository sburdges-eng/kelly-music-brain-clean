## What this engine does
- Drives the lead line that listeners follow and remember; sets motif identity and emotional contour.

## Core basics
- Prioritize clear motifs and short call/response cells.
- Control contour: stepwise motion with selective leaps and gravity to tonal centers.
- Balance repetition vs. variation via rhythmic cells and interval tweaks.
- Phrase breathing: start pickups, land on strong beats, leave space.
- Use register and articulation to contrast sections.

## Public repos to study
- magenta/magenta — MelodyRNN, ImprovRNN, Music Transformer; inspect positional encodings, tie-breaking for repeated tokens, teacher-forced vs. free decoding drift, and phrase-level bar tokens for long arcs.
- facebookresearch/audiocraft — MusicGen melody conditioning and guided decoding; study classifier-free guidance on melodic control tracks and how prompt wording shifts contour stability.
- spotify/basic-pitch — Polyphonic pitch/MIDI extraction that survives noise and tuning drift; ideal for auto-building melodic corpora from stems and measuring transcription confidence for active learning.
- microsoft/muzic — Melody and PopMAG multi-track models; analyze how chord/beat/style tokens gate melodic density and how Museformer’s grouped attention maintains motif coherence over long contexts.
- lucidrains/music-transformer-pytorch (community) — Clean reference of relative attention and ALiBi-style variants; great for rapid ablation of positional schemes on melodic continuity.

## Two recursive study questions
- How does attention weight phrase peaks vs. pickups in Melody Transformer? Trace attention maps around cadential bars in magenta/magenta and try masking them.
- How sensitive is melody conditioning to quantization noise? Feed slightly jittered guide melodies into audiocraft MusicGen and measure note overlap and KL on pitch-class histograms.

## Advanced techniques to notice
- Transformer phrase-level embeddings (bar or phrase tokens) to reduce repetition drift.
- Interval- and contour-aware loss terms that penalize unwanted leaps.
- Guided decoding with control tracks (melody stems, sketch contours, or key/scale tokens).

## Genre references (50)
- pop, alt pop, synthpop, hyperpop, rock, classic rock, hard rock, punk, pop punk, emo, metal, prog metal, djent, death metal, black metal, blues, jazz, swing, bebop, cool jazz, jazz fusion, bossa nova, samba, salsa, reggaeton, dembow, bachata, merengue, afrobeat, amapiano, highlife, soukous, r&b, neo-soul, gospel, trap, boom bap, drill, lo-fi hip hop, house, deep house, tech house, techno, trance, drum and bass, jungle, dubstep, future bass, uk garage, grime
