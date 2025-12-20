# QA Fixtures & Scoring

## Golden Fixtures
- **MIDI (per genre/emotion)**: straight pop, lofi swing, boom-bap, Dilla, trap, indie ballad. Include chord progressions, groove offsets, dynamics curves.
- **Audio snippets**: short stems aligned to MIDI for chord/key checks and onset/timing validation.
- **Rule-break cases**: modal interchange, unresolved cadence, parallel motion, tension-curve peaks.

## Checks
- **Key/Chord accuracy**: F1 vs annotated MIDI; tolerance for inversions noted.
- **Timing**: note-on/off drift <1 ms vs schedule for Dee path; swing/humanization within spec.
- **Groove/Humanization**: compare generated offsets to template bounds; fail on over/under-humanize.
- **Tension Curve Scoring**: compute tension vs time (cadence, dissonance density); verify target arc (e.g., climb → climax → release).
- **Arrangement markers**: where used, ensure section boundaries align to expected bars.

## Harness
- Deterministic seeds for generation.
- Batch runner: play fixtures through plugin (Kelly preview, Dee pass-through) and log metrics.
- Outputs: JSON report with key/chord F1, drift stats, swing bounds, tension-curve fit score.

## Pass/Fail
- Drift: p95 ≤1 ms (Dee).
- Swing/humanization within template bounds.
- Key/chord F1 above threshold per genre fixture.
- Tension-curve fit above threshold for rule-break presets.
