# Pricing & SKUs

<<<<<<< Current (Your changes)
## Overview

miDiKompanion is available in three tiers to suit different production needs. For installation and host compatibility, see [Quick Start](QUICK_START.md) and [Host Support Matrix](daw_integration/HOST_SUPPORT_MATRIX.md).

## Format Support Matrix

| Format | Free (Lite) | One-Time Purchase | Premium AI |
|--------|-------------|-------------------|------------|
| AUv3 (macOS/iOS) | ⚠️ Demo | ✅ Full | ✅ Full |
| VST3 (macOS/Win) | ⚠️ Demo | ✅ Full | ✅ Full |
| CLAP (macOS/Win/Linux) | ❌ No | ✅ Full | ✅ Full |
| Standalone | ✅ Basic | ✅ Full | ✅ Full |

*Note: Demo versions may have limited session lengths or restricted save/export capabilities.*

## Feature Comparison

| Feature | Free (Lite) | One-Time Purchase | Premium AI |
|---------|-------------|-------------------|------------|
| MIDI Export | ✅ Yes | ✅ Yes | ✅ Yes |
| Stem/Template Export | ❌ No | ✅ Yes | ✅ Yes |
| Local Inference | ✅ Unlimited | ✅ Unlimited | ✅ Unlimited |
| Daily Cloud Budget | 60 sec | 300 sec | ✅ Unlimited |
| Custom ONNX Upload | ❌ No | ❌ No | ✅ Yes |
| Explainability | ❌ No | ❌ No | ✅ Yes |
| Cloud Preset Sync | ❌ No | ❌ No | ✅ Yes |
| Multi-Ref Blending | ❌ No | ✅ Yes | ✅ Yes |
=======
> **Related**: [Quick Start](QUICK_START.md) | [Export Workflow](EXPORT_WORKFLOW.md) | [Host Support Matrix](daw_integration/HOST_SUPPORT_MATRIX.md)
>>>>>>> Incoming (Background Agent changes)

## Free (Lite)
- Formats: limited (e.g., AUv3/VST3 demo), capped usage.
- Limits: daily cloud calls capped; no cloud sync; no custom ONNX; no explainability; fewer presets; basic references (single ref, limited weights); no export of stems/templates (MIDI only).

## Plugin One-Time Purchase
- Unlocks full plugin formats (AUv3/VST3/CLAP), unlimited local use, stems/templates export, multi-reference blending, per-section rule-break toggles, genre presets, host tempo/key sync.
- No recurring charge for local features.

## Premium AI ($20/mo, unlimited cloud)
- Unlimited cloud inference (no overages).
- Explainability on demand (why chord/groove), classifiers/QA gates, higher-quality models, staged auto-updates with pin/rollback.
- Custom ONNX uploads after review/codesign; promo credits for approved models.
- Cloud preset sync; reference embedding cache in cloud; priority updates.
- Lyric/LLM deferred (not MVP) but reserved for premium when added.

## Metering: Cloud vs Local
- Local inference (ONNX/CoreML/TFLite) is unmetered for all tiers.
- Cloud metering unit: **compute-seconds** per request (wall-clock runtime of the selected model, rounded up to the nearest 100ms). Token-based models also report tokens for observability, but billing/limits use compute-seconds.
- Cached embeddings/reference fingerprints do **not** count toward metering.
- Per-request budget guardrails: if an individual request is estimated to exceed 15 compute-seconds, the client downgrades to the "draft" model or prompts the user to split the request.
- When a tier limit is hit, the client auto-falls back to local models (if available) and surfaces a banner with time-to-reset and a "retry in cloud" option once the window resets.

## Rate Limits by Tier
| Tier | Daily cloud budget | Burst / concurrency | Notes |
|------|-------------------|---------------------|-------|
| Free | 60 compute-seconds/day | 2 concurrent, 6 req/min | Draft model only; no stems/templates export. |
| One-Time Plugin | 300 compute-seconds/day | 4 concurrent, 12 req/min | Cloud used only when local confidence is low; export/stems enabled. |
| Premium AI | Unlimited (fair use) | 8 concurrent, 30 req/min | Priority queue; explainability/classifiers enabled. |

## Overage & Degradation Behavior
- If burst limit is exceeded: queue up to 3 requests; additional requests fall back to local/draft immediately.
- If daily budget is exceeded (non-premium): disable cloud until reset; keep local/draft enabled with a "cloud paused" banner.
- Premium abuse protection: sustained >30 req/min for 2 minutes moves the user to a lower-priority lane until traffic normalizes; no hard stop unless ToS violation is detected.

## Notes
- Opt-in data only; default no retention.
- Latency warnings shown for cloud use.
- User owns MIDI; fair-use/parody disclaimer in UI.

## Related Documentation
- [Quick Start Guide](QUICK_START.md) - Getting started
- [Export Workflow](EXPORT_WORKFLOW.md) - Export options by tier
- [Host Support Matrix](daw_integration/HOST_SUPPORT_MATRIX.md) - DAW compatibility
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
