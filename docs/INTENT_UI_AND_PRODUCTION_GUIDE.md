# Intent UI & Production Guide Spec

## Plugin UI (Two Tabs)
- **Kelly (Intent/Generative)**: full intent editing; simplified default, advanced on demand; preview/import of generated MIDI/arrangements.
- **Dee (Low-Latency Co-Producer)**: pass-through + augmentation; <20–40 ms end-to-end; local/pinned model; intent read-only here (no heavy regeneration).

## Intent Schema Exposure
- **Default (Simple)**: primary emotion, secondary tension, genre, tempo, key/mode (optional), vibe tag, target length.
- **Advanced (On Demand)**: vulnerability scale, narrative arc (inferred default; manual override), rule-to-break + justification, groove feel, energy arc target, section emphasis, instrumentation presets, reference weights (sliders).
- **Narrative Arc**: inferred from emotion + references; user can override.
- **Rule-Breaking Controls**: per-section toggles; presets mapped to emotions; safe by default; trigger rule-break after 10 user-flagged unsuccessful outputs.

## UX States: Simple vs Advanced
- Default (Simple): left rail shows the minimal fields above; narrative arc preview and tension curve appear as read-only badges; advanced controls hidden behind "More control" chevron.
- Advanced: expands a right drawer with granular sliders/fields (rule-break justifications, vulnerability, section emphasis, instrumentation presets, reference weights). Drawer remembers last-open state per session; collapses on Escape or when user reverts to Simple.
- Inline validation: highlight conflicts (e.g., "Lofi + 160 BPM + Vulnerability 0.9") with one-click suggestions; do not block generation unless key/mode conflict toggle is set to "require".
- Quick actions: "Use host tempo/key", "Match reference groove", "Reset swing/humanization". These appear contextually under the tempo/key rows.

## Intent Editing Flows
- First-open: pre-fill tempo/key from host; if detection disagrees with host, show a conflict banner with three options (use host / use detected / keep user) before generation.
- References: up to 3 slots by default; enabling a 4th slot collapses weights into a stacked list with per-reference sliders and a global "normalize weights" toggle.
- Rule-break presets: always mapped to the active emotion; opening the preset picker shows emotion-aligned presets first, with "neutral/safe" pinned.
- State recall: last-used simple/advanced state and per-field values persist per project; "Reset to defaults" clears advanced-only fields first, then simple fields on second click (two-step undo prevention).

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
- Export shape: JSON blob accompanies MIDI bundle, plus a human-readable markdown sheet. JSON keys per section: `name`, `start_bar`, `end_bar`, `key`, `mode`, `bpm`, `groove`, `swing_pct`, `humanization`, `eq_low/mid/high`, `dynamics`, `stereo`, `rule_breaks`, `tension_note`, `instrumentation_preset`, `confidence`.
- Confidence handling: if `confidence < 0.45`, include only section summary and groove/humanization notes; omit EQ/dynamics/stereo to avoid hallucination.

## Presets
- Shared presets: local storage; optional cloud sync for premium.
- Includes: intent presets, reference-weight presets, groove/humanization presets, instrumentation bundles.

## Latency Expectations
- Dee: <20–40 ms end-to-end, pass-through + augmentation.
- Kelly: heavier generation allowed; must not block Dee; background cloud fallback with user warning.

## Error/Feedback UX
- Confidence/alerts: show low-confidence badges for key/chord/groove detection; offer retry or use-host.
- Rule-break explanation available in premium tiers on demand.
