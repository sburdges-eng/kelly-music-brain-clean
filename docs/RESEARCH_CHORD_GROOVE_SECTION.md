# Research AI: Chord, Groove, Section (Theory + Emotion + Timing)

## Scope
- Background agents producing design/reference notes (no direct generation).
- Sources: music theory/groove/arrangement guides; allowlisted Git/GitHub/papers; existing project docs; emotion_thesaurus for emotional mapping. No broad scraping.

## Behavior
- RAG with citations; “don’t know” if unsupported.
- Local cache default; cloud with permission. Status/pause panel.
- Personas: Kelly voice for writer-block narrative; Dee voice for co-producer/timing notes.

## Chord/Progression (Theory)
- Pipeline: notes → keys → scales → triads → inversions → progressions.
- Topics: cadences, substitutions, borrowed chords, modal interchange, voice leading basics, reharm options.
- Outputs: design notes, progression examples, variation/quality rules.

## Chord/Progression (Emotion Mapping)
- Map what/why/how chords express valence/arousal; when to use borrowed chords, unresolved cadences, pedal points.
- Outputs: emotional-function notes, rule-break guidance with rationale.

## Groove
- Timing feel: swing percentages, humanization ranges, genre pockets (funk/boom-bap/Dilla/trap/straight).
- Microtiming bounds and latency-safe application; note-on/off drift targets.
- Outputs: pocket templates, do/don’t lists, per-genre hints.

## Sections/Arrangement
- Archetypes: verse/chorus/bridge/pre; energy arcs; narrative arcs.
- Instrumentation templates per section/genre; section-length norms; transitions/contrast tips.
- Outputs: section checklists and templates.

## Citations (format)
- Inline short refs: `[1][theory]`, `[2][practice]`, `[3][anecdotal]`, `[4][tooling]`.
- End-of-section bibliography: numbered list with source type tags.

## Not allowed
- No uncited claims; no unlicensed or non-allowlisted sources; no broad web scraping; no direct user-facing generation; no clinical/medical guidance.
