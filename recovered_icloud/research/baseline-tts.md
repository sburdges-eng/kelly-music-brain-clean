## Baseline TTS (fast + controllable)

- **Model**: FastPitch or FastSpeech2 for duration/F0/energy control; phonemizer with MFA lexicon; text normalization pipeline (rule-based + numbers/dates).
- **Vocoder pairing**: HiFi-GAN V1/V2 or UnivNet (see vocoder doc); train/finetune on 22.05k or 24k to reduce RT load; consider 44.1k only if quality demands it.
- **Data**: Multispeaker clean corpora (LibriTTS-R, VCTK, HiFiTTS). Filter with SNR > 20 dB, trim silences, loudness normalize to -23 LUFS. Keep manifest with license per clip.
- **Alignment**: Montreal Forced Aligner for phonemes/durations; cache phoneme durations to speed training.
- **Controls**: predict durations, F0, energy; expose sliders/JSON API for prosody tweaks; optional style tokens (GST) for preset voices.
- **Inference (RT)**: export to ONNX/ TensorRT; FP16; batch size 1; streaming decoder in 80–120 ms windows; pre-warm model once to avoid cold-start stalls.
- **Serving**: gRPC/WebRTC; AudioUnit/CoreAudio for macOS/iOS; double-buffer with overlap-add; keep text normalization and phonemization off the RT thread.
- **Testing**: WER for intelligibility, UTMOS/MOS-proxy for quality, latency p50/p95, glitch/XRun count under buffer stress.
