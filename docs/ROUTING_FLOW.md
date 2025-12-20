# Routing & Flows (Kelly vs Dee)

## Dee (Low-Latency Co-Producer)
- **Input**: Host MIDI in → Dee pass-through.
- **Processing**: Local/pinned model augments (humanization, groove tweaks, CC shaping, note suggestions) within <20–40 ms budget.
- **Output**: Host MIDI out (augmented). No blocking on cloud; CoreML/CPU only.
- **Safety**: Bypass toggle; drift guard (<1 ms). Shows provider (CoreML/CPU) and latency badge.
- **Explainer**: Local term helper (what “boom-bap” means) without cloud calls.

## Kelly (Intent/Generative)
- **Input**: Intent (simple → advanced), references with weights, host tempo/key read.
- **Processing**: Shared core + heads; can use cloud fallback with user warning; does not block Dee path.
- **Preview**: Generates harmony/groove/arrangement snapshots; user can audition.
- **Commit/Export**: Import to host (MIDI), or export bundle (MIDI + intent/arrangement JSON + per-section production guide). Stems/templates for paid plugin.
- **Conflict Handling**: If detected key/mode differs from host/user, prompt to choose host/detected/user.

## Rule-Break Coordination
- Global safe default. Rule-break presets mapped to emotions.
- Per-section toggles; trigger rule-break after 10 user-flagged unsuccessful outputs (apps).
- Tension-curve scoring; explanation available in premium.

## Cloud Usage
- Dee: none (local only).
- Kelly: cloud fallback allowed; warn if >200 ms; staged model updates; pin/rollback available.
