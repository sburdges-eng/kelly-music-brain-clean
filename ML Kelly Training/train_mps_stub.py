"""
MPS-friendly training stub for a 16 GB M4 MacBook Pro.

Reads a YAML config (e.g., configs/laptop_m4_small.yaml), loads a manifest
of audio + labels, builds a tiny mel classifier, and trains with mixed
precision + gradient accumulation. Swap the dataset/model with your real
pipeline as needed.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

# Experiment Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trainer")

def save_checkpoint(model, optimizer, epoch, step, loss, cfg, path: Path):
    """Robust checkpointing with metadata."""
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": cfg,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    torch.save(ckpt, temp_path)
    os.replace(temp_path, path) # Atomic swap
    logger.info(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path: Path):
    """Resumes training from a checkpoint."""
    if not path.exists():
        return 0, 0, float('inf')
    
    logger.info(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["step"], ckpt["loss"]

try:
    import torchaudio
    from torchaudio import transforms as T
except Exception:  # pragma: no cover - optional dependency
    torchaudio = None
    T = None


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg: Dict[str, Any]) -> torch.device:
    preferred = cfg.get("device", "mps")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    return torch.device("cpu")


class ManifestAudioDataset(torch.utils.data.Dataset):
    """
    Minimal manifest-driven audio dataset.

    Manifest JSONL entries must contain audio path and label fields (keys
    configurable via YAML). Audio is loaded, resampled, normalized, cropped/
    padded to segment length, and converted to mel.
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int,
        segment_seconds: float,
        target_lufs: float,
        audio_key: str = "audio",
        label_key: str = "label",
        n_mels: int = 128,
        hop_length: int = 512,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.target_rms = 10 ** (target_lufs / 20)
        self.audio_key = audio_key
        self.label_key = label_key
        self.n_mels = n_mels
        self.hop_length = hop_length

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

        labels = [str(it[label_key]) for it in self.items if label_key in it]
        self.label_to_idx = label_to_idx or {lab: i for i, lab in enumerate(sorted(set(labels)))}

        if torchaudio:
            self.mel = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                hop_length=hop_length,
            )
            self.to_db = T.AmplitudeToDB()
        else:  # fallback: identity
            self.mel = None
            self.to_db = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        it = self.items[idx]
        audio_path = Path(it[self.audio_key])
        label_str = str(it[self.label_key])
        label = torch.tensor(self.label_to_idx.get(label_str, 0), dtype=torch.long)

        wav = self._load_audio(audio_path)
        wav = self._ensure_length(wav)
        wav = self._normalize_rms(wav)

        if self.mel:
            mel = self.mel(wav)
            mel = self.to_db(mel)
        else:
            mel = wav.unsqueeze(0)  # [1, time]

        # Simple channel-first: [n_mels, time]
        return mel, label

    def _load_audio(self, path: Path) -> torch.Tensor:
        if torchaudio and path.exists():
            wav, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            wav = wav.mean(dim=0, keepdim=False)  # mono
            return wav
        # Fallback: random noise if file missing or torchaudio unavailable
        return torch.randn(self.segment_samples)

    def _ensure_length(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.numel() >= self.segment_samples:
            start = torch.randint(0, wav.numel() - self.segment_samples + 1, (1,)).item()
            return wav[start : start + self.segment_samples]
        pad = self.segment_samples - wav.numel()
        return torch.nn.functional.pad(wav, (0, pad))

    def _normalize_rms(self, wav: torch.Tensor) -> torch.Tensor:
        rms = wav.pow(2).mean().sqrt()
        if rms > 0:
            wav = wav * (self.target_rms / rms)
        return wav


def build_model(n_mels: int, num_classes: int, cfg: Dict[str, Any]) -> torch.nn.Module:
    hidden = cfg["model"].get("embedding_dim", 256)
    dropout = cfg["model"].get("dropout", 0.1)
    return torch.nn.Sequential(
        torch.nn.Conv1d(n_mels, hidden, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1),
        torch.nn.Flatten(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, num_classes),
    )


def train(cfg_path: str = "configs/laptop_m4_small.yaml") -> None:
    cfg = load_config(cfg_path)
    device = get_device(cfg)

    torch.manual_seed(cfg.get("seed", 42))
    torch.set_float32_matmul_precision("high")
    if device.type == "mps":
        torch.backends.mps.allow_tf32 = True

    dcfg = cfg["data"]
    
    # Use StreamingAudioDataset if configured or for very large datasets
    if cfg.get("use_streaming", False):
        from python.penta_core.ml.datasets.streaming import StreamingAudioDataset
        train_ds = StreamingAudioDataset(
            dcfg["train_manifest"],
            sample_rate=dcfg["sample_rate"],
            segment_seconds=dcfg["segment_seconds"],
            audio_key=dcfg.get("manifest_audio_key", "audio"),
            label_key=dcfg.get("manifest_label_key", "label"),
        )
        val_ds = StreamingAudioDataset(
            dcfg["val_manifest"],
            sample_rate=dcfg["sample_rate"],
            segment_seconds=dcfg["segment_seconds"],
            audio_key=dcfg.get("manifest_audio_key", "audio"),
            label_key=dcfg.get("manifest_label_key", "label"),
        )
    else:
        train_ds = ManifestAudioDataset(
            dcfg["train_manifest"],
            sample_rate=dcfg["sample_rate"],
            segment_seconds=dcfg["segment_seconds"],
            target_lufs=dcfg["target_lufs"],
            audio_key=dcfg.get("manifest_audio_key", "audio"),
            label_key=dcfg.get("manifest_label_key", "label"),
            n_mels=dcfg.get("n_mels", 128),
            hop_length=dcfg.get("hop_length", 512),
        )
        val_ds = ManifestAudioDataset(
            dcfg["val_manifest"],
            sample_rate=dcfg["sample_rate"],
            segment_seconds=dcfg["segment_seconds"],
            target_lufs=dcfg["target_lufs"],
            audio_key=dcfg.get("manifest_audio_key", "audio"),
            label_key=dcfg.get("manifest_label_key", "label"),
            n_mels=dcfg.get("n_mels", 128),
            hop_length=dcfg.get("hop_length", 512),
            label_to_idx=train_ds.label_to_idx,
        )

    num_classes = len(train_ds.label_to_idx)
    n_mels = dcfg.get("n_mels", 128)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=dcfg.get("num_workers", 0),
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=dcfg.get("num_workers", 0),
    )

    model = build_model(n_mels, num_classes, cfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )

    # RESUME LOGIC
    ckpt_dir = Path("checkpoints") / cfg.get("model_id", "default")
    latest_ckpt = ckpt_dir / "latest.pt"
    start_epoch, total_steps, best_loss = load_checkpoint(model, opt, latest_ckpt)

    precision = cfg.get("precision", "fp16")
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and precision == "fp16")
    )

    grad_accum = cfg["training"]["grad_accum_steps"]
    log_every = cfg["training"]["log_every"]
    ckpt_every = cfg["training"].get("ckpt_every", 500)
    max_steps = cfg["training"].get("max_steps", 200)  # keep short by default
    spending_limit = cfg.get("spending_limit_usd", 100.0)
    cost_per_step = cfg.get("cost_per_step_sim", 0.001) # Simulated cost

    steps = total_steps
    model.train()
    start_time = time.time()
    
    logger.info(f"Training started. Spending Limit: ${spending_limit}")

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        for i, (x, y) in enumerate(train_dl):
            # Cost Check
            current_cost = steps * cost_per_step
            if current_cost > spending_limit:
                logger.warning(f"STOPPING: Spending limit of ${spending_limit} reached!")
                save_checkpoint(model, opt, epoch, steps, loss.item(), cfg, latest_ckpt)
                return

            x = x.to(device)
            y = y.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
                enabled=(precision in ("fp16", "bf16")),
            ):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % grad_accum == 0:
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
                steps += 1

                if steps % log_every == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Epoch {epoch} | Step {steps} | Loss {loss.item():.4f} | Cost ${current_cost:.2f} | Time {elapsed:.1f}s")

                if steps % ckpt_every == 0:
                    save_checkpoint(model, opt, epoch, steps, loss.item(), cfg, latest_ckpt)

            if steps >= max_steps:
                break
        if steps >= max_steps:
            break

    # Final Save
    save_checkpoint(model, opt, epoch, steps, loss.item(), cfg, latest_ckpt)

    # Quick val pass (optional, single batch)
    model.eval()
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb).argmax(dim=1)
            acc = (preds == yb).float().mean().item()
            print(f"val batch acc: {acc:.3f}")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPS laptop training stub")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/laptop_m4_small.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)

