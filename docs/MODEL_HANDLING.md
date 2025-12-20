# Model Handling: Kelly & Dee

## Targets
- **Dee**: small, local, pinned; no surprise updates mid-session; CoreML when available; CPU fallback.
- **Kelly + heavy modules**: shared core + specialized heads; staged auto-update with rollback/pin.

## CoreML / ONNX
- Prefer CoreML delegate on Apple silicon; fall back to CPU ONNX Runtime.
- Detect and log provider in UI for transparency (CoreML vs CPU).

## Versioning & Provenance
- Each module reports: model id, semantic version, build date, provider (CoreML/CPU), checksum.
- Staged rollout: canary → broad; maintain rollback path to last-known-good per module.
- User pinning allowed (pro users) with one-click revert to latest.

## Update Policy
- Default: staged auto-update when online.
- Safeguards: no update during active session/render; prompt if update queued.
- Offline: use pinned/local copies; warn if model is older than N releases.

## Custom ONNX Intake
- Allowed after review/codesign: validate opset, size, performance budget, and safety (no custom ops).
- Upon approval: sign and cache; expose as user-selectable model with provenance note.
- If rejected: return validation report.

## Telemetry (Opt-In Only)
- Capture anonymized: load time, provider used, failures, rollback events. No content data.

## Latency Budgets
- Dee: <20–40 ms end-to-end, including provider overhead.
- Kelly: background-capable; must not block Dee; can offload to cloud with user warning.
