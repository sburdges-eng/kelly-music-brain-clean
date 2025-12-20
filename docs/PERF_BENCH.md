# Performance Benchmarks & Test Plan

## Targets
- **Dee (pass-through + augmentation)**: end-to-end <20 ms preferred, max 40 ms (buffer + processing).
- **Kelly (generation/preview)**: may exceed 40 ms; must not block Dee; background/cloud allowed with user warning.
- **Memory budget**: Dee model + buffers <150 MB on desktop, <120 MB on iOS; Kelly core + heads <600 MB desktop, <350 MB iOS (active generation).

## Providers
- CoreML delegate when available; CPU fallback. Log provider and timing per block.

## Bench Coverage
- Latency per audio/MIDI block (128/256/512 samples) across hosts (Logic 10.7+, Ableton 11+).
- Jitter under stress (multiple MIDI streams, automation).
- Model load/init time (cold/warm).
- Cloud fallback latency (warn if >200 ms round-trip).
- MIDI timing accuracy: note-on/off drift <1 ms vs schedule.
- Humanization correctness: offsets remain within spec; swing tied to host tempo.

## Test Harness
- Deterministic MIDI fixtures (straight + swing + polyrhythm-lite) to measure drift.
- Synthetic stress: rapid intent edits, reference swaps, preset changes mid-playback.
- AU validation (auval) for AUv3; plugin sandbox tests for Logic/Ableton paths.
- Record per-block timestamps; report p50/p95/p99 latency.

## Pass/Fail Gates
- Dee: p95 ≤40 ms end-to-end; no dropped events.
- Kelly: no blocking of Dee path; background tasks yield; cloud warning when used.
- Memory: stays within budget for target platforms; no growth over 10-minute soak.

## Reporting
- Per-release perf report: provider, host, buffer size, p50/p95/p99 latency, memory peak, failures/rollbacks.
