## TTS/VC RT Build Snapshot (15-min pass)

Sources: `baseline-tts.md`, `high-fidelity-vc.md`, `vocoder.md`, `rt-tips.md`, `controls.md`, `data.md`, `safety.md`.

### Recommended stack
- **TTS**: FastPitch/FastSpeech2 + HiFi-GAN (22.05–24k) with phonemizer + MFA durations; ONNX/TensorRT FP16.
- **VC**: GPT-SoVITS/CosyVoice for quality; AutoVC/FragmentVC for lighter RT; ECAPA/x-vector speaker encoder, 3–8 s prompt.
- **Vocoder**: HiFi-GAN V1/V2 (RT) or BigVGAN chunked; 20–40 ms chunks, 50% overlap-add.
- **Serving**: Worker thread for model; RT thread for callback only. gRPC/WebRTC for cloud; AudioUnit/CoreAudio for mac/iOS. Warm-up once per model.
- **Controls/API**: `{text, voice_id, style_token?, emotion:{arousal,valence}, prosody:{f0_scale,duration_scale,energy_scale}, vc:{strength,formant_shift}}`; expose bypass/reset and safe ranges.
- **Data**: LibriTTS-R, VCTK, HiFiTTS + ESD/EmoV-DB; manifest with license/consent; clean (MFA, trim, -23 LUFS, SNR>20).
- **Safety**: blocklist, consent for cloning, watermark/tag per utterance, rate limits, local-first voiceprints with optional encrypted sync.

### Memory budget (Mac GPU, 22.05 kHz, AutoVC + HiFi-GAN)
- System: ~4.8 GB (macOS/headroom).
- Content encoder (HuBERT/WavLM base): ~0.6 GB total (weights + runtime).
- Speaker encoder (ECAPA): ~0.06 GB.
- AutoVC core: ~0.55–0.6 GB.
- HiFi-GAN (22.05 kHz): ~0.33 GB.
- FastPitch fallback (idle): ~0.2 GB — unload after warm-up if unused.
- RT infra (rings/threads/metrics): ~0.1–0.13 GB.
- Safety/meta (consent/blocklist/watermark): ~0.02 GB.
- **Total stack**: ~6.7 GB; **Total resident** ≈ 11.5 GB (incl. system) leaving ~4.5 GB headroom (no swap).
- Avoid on-device: BigVGAN, large HuBERT/WavLM, second concurrent voice (+1.2–1.6 GB), heavy IDE/browser/Docker during RT tests.

### Build skeleton (proposed)
- `services/tts/` — FastPitch/FastSpeech2 inference wrapper (ONNX), text norm + phonemizer, prosody controls.
- `services/vc/` — VC pipeline (GPT-SoVITS or lightweight VC), speaker encoder cache, formant/denoise guard.
- `services/vocoder/` — HiFi-GAN/BigVGAN runner with chunked streaming and overlap-add.
- `rt/` — lock-free ring buffer, double-buffer, warm-up routines, metrics hooks.
- `api/` — gRPC/WebRTC server, JSON schema for controls, auth/ratelimit hooks.
- `safety/` — blocklist, consent checks, watermark tagger, logging (non-RT).
- `data/` — manifests, cleaning scripts (MFA, loudnorm), license ledger.
- `tests/` — WER, UTMOS/MOS-proxy, speaker-sim, latency/glitch harness.

### Immediate next steps
1) Pick target sample rate (22.05k or 24k) and platform (on-device vs GPU cloud) to finalize model exports.
2) Decide VC path (quality vs speed) to fix API and caching strategy.
3) Stand up minimal RT harness: text norm → TTS → vocoder with 20–40 ms chunks; measure latency p50/p95.
4) Add safety gates: consent flag for voice cloning, blocklist, watermark tag.
5) Prepare manifests for initial datasets with license/consent columns.

### RT harness checklist (22.05 kHz, AutoVC first)
- Single stream only; assert model sizes on load; unload FastPitch after warm-up.
- Audio callback 64–96 samples @ 22.05 kHz; lock-free ring between RT and worker; no allocs/locks in callback.
- Chunking 20–32 ms into vocoder; 50% overlap-add; carry state across chunks.
- Warm-up once per model; cache speaker/style embeddings.
- Safety inline: consent token required for VC, blocklist, per-utterance watermark/tag, rate limits.
- Metrics: log p50/p95 end-to-end latency, XRuns/glitches, speaker-sim cosine, WER/MOS-proxy snapshots.
