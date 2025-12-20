# Preset Storage & Sync

## Storage Model
- **Local** (all tiers): presets stored in user home (per-user, per-host), with export/import JSON.
- **Cloud** (premium): optional sync; user opt-in; conflict resolution via timestamp + “last writer wins” with manual restore option.

## Scope
- Intent presets (simple + advanced fields).
- Reference-weight presets (multi-reference blends).
- Groove/humanization presets; swing tied to host tempo lock.
- Instrumentation bundles (genre presets).
- Production guide templates (bullet structures).

## Safety & Versioning
- Each preset includes version, checksum, created/updated timestamps, model/provider tags.
- Host and platform metadata stored for compatibility warnings.
- Offline mode: use cached local; queue sync when online.

## UX
- Preset browser shared across plugin/desktop; tabs for Kelly/Dee.
- Export/import buttons (JSON); reset to factory.
- Cloud toggle (premium): enable/disable; clear cloud cache on disable.

## Limits
- Free: local only; limited preset count (e.g., 20).
- Premium: higher limits (e.g., 200); cloud sync enabled.
