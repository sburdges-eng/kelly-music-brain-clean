# Intent UI & Production Guide Spec

## Plugin UI (Two Tabs)
- **Kelly (Intent/Generative)**: full intent editing; simplified default, advanced on demand; preview/import of generated MIDI/arrangements.
- **Dee (Low-Latency Co-Producer)**: pass-through + augmentation; <20–40 ms end-to-end; local/pinned model; intent read-only here (no heavy regeneration).

## Intent Schema Exposure
- **Default (Simple)**: primary emotion, secondary tension, genre, tempo, key/mode (optional), vibe tag, target length.
- **Advanced (On Demand)**: vulnerability scale, narrative arc (inferred default; manual override), rule-to-break + justification, groove feel, energy arc target, section emphasis, instrumentation presets, reference weights (sliders).
- **Narrative Arc**: inferred from emotion + references; user can override.
- **Rule-Breaking Controls**: per-section toggles; presets mapped to emotions; safe by default; trigger rule-break after 10 user-flagged unsuccessful outputs.

## Reference Handling
- Multi-reference blending with weights (0–100% per reference).
- Host tempo/key read; conflict prompt with “use host”, “use detected”, “keep user” options.
- Humanization: global; swing tied to host tempo lock.

## Generation & Preview
- Kelly: preview MIDI (harmony, groove, arrangement snapshots) before commit; import to host or export bundle (MIDI + intent/arrangement JSON + per-section guide).
- Dee: live augmentation, pass-through MIDI; explain terms (local LLM-style helper) without cloud; no blocking on cloud.

## Production Guide (Section-Level Bullets)
- For each section: key/mode, BPM, groove feel/humanization summary, EQ notes (low/mid/high), dynamics (range, compression hint), stereo note, rule-break notes, tension curve note, instrumentation preset used.
- Optional per-track mini-notes (1–2 bullets) only when high confidence; otherwise section-level only.

## Presets
- Shared presets: local storage; optional cloud sync for premium.
- Includes: intent presets, reference-weight presets, groove/humanization presets, instrumentation bundles.

## Latency Expectations
- Dee: <20–40 ms end-to-end, pass-through + augmentation.
- Kelly: heavier generation allowed; must not block Dee; background cloud fallback with user warning.

## Error/Feedback UX
- Confidence/alerts: show low-confidence badges for key/chord/groove detection; offer retry or use-host.
- Rule-break explanation available in premium tiers on demand.
