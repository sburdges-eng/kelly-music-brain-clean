# Host Support Matrix & Transport Sync Scope

Baseline assumptions for the intent-centric plugin (AUv3/VST3/CLAP) and standalone apps. These constraints inform QA, packaging, and transport sync behavior. For pricing information, see [Pricing & SKU](../PRICING_SKU.md). For getting started, see [Quick Start](../QUICK_START.md).

## Platform Baselines
- **macOS**: 13.0 (Ventura) or newer; Apple Silicon native with Intel Rosetta fallback. AUv3 + VST3 shipped.
- **Windows**: 11 22H2 or newer; VST3 only. CLAP optional where supported.
- **iOS/iPadOS**: 17.0 or newer; AUv3 only. Transport/tempo sync via host; standalone uses CoreMIDI clock.
- **Linux**: Experimental; CLAP preferred, VST3 where host supports it (Bitwig/Reaper/Ardour). No QA guarantee.
- **AAX**: Not in scope for MVP; revisit after initial launch.

## Minimum Host Versions & Formats
| Host | Min Version | Formats | Notes |
|------|-------------|---------|-------|
| Logic Pro | 10.7.9 | AUv3 | Offline bounce-safe; AU validation in CI. |
| Ableton Live | 11.3 | VST3 | Link + tempo map sync; capture loop range. |
| FL Studio | 21.2 | VST3 | PPQ-aware; pattern mode tested. |
| Reaper | 6.80 | VST3 / CLAP | Per-project sample rate; anticipative FX off for Kelly. |
| Studio One | 6.5 | VST3 | Pre-roll + punch tested; channel color sync optional. |
| Bitwig | 5.1 | CLAP | Transport + phase sync; per-note expression passthrough. |
| Pro Tools | 2023.6 | N/A (monitor only) | Use standalone + virtual MIDI; no AAX yet. |
| MainStage (mac) | 3.6 | AUv3 | Low-latency Dee mode profile. |

## Transport Sync Scope
- Read: start/stop, song position (beats + samples), tempo map (ramps), time signature changes, loop/locators, pre-roll/punch, sample rate and block size.
- Write: optional start/stop when "Follow Host" is enabled; never asserts tempo/time signature changes back to host.
- Latency reporting: reports plugin latency for Kelly (heavy generation) paths; Dee path reports zero and bypasses PDC changes.
- Offline/bounce: detects offline renders; switches to deterministic render mode, disables external cloud calls unless explicitly allowed.
- Link/Network: Ableton Link enabled in standalone and Live; session join/leave mirrored in UI.

## Validation Targets
- AU validation (auval) for Logic/MainStage in CI.
- VST3 validator for Windows/macOS builds.
- CLAP lint/validator for Bitwig/Reaper targets.
- Host smoke tests per release: load, transport follow, tempo change, loop playback, offline bounce, reopen session.
