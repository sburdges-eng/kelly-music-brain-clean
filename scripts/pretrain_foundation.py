#!/usr/bin/env python3
"""
Kelly ML Foundation Model Pre-training Script.

Performs Self-Supervised Learning (SSL) on massive unlabeled audio datasets.
Uses Contrastive Learning (SimCLR style) to learn rich musical representations.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from torch.utils.data import DataLoader

from python.penta_core.ml.datasets.streaming import StreamingAudioDataset
from python.penta_core.ml.training.architectures import MusicFoundationModel, EmotionCNN # Assuming EmotionCNN can be used as backbone
from python.penta_core.ml.training.augmentation import AudioAugmentor
from python.penta_core.ml.training.losses import ContrastiveLoss

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pretrainer")

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_device(cfg: Dict[str, Any]) -> torch.device:
    pref = cfg.get("device", "auto")
    if pref == "auto":
        if torch.backends.mps.is_available(): return torch.device("mps")
        if torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(pref)

def save_checkpoint(model, optimizer, epoch, steps, loss, cfg, path: Path):
    ckpt = {
        "epoch": epoch,
        "steps": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": cfg
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    logger.info(f"Checkpoint saved: {path}")

def pretrain(cfg_path: str):
    cfg = load_config(cfg_path)
    device = get_device(cfg)
    logger.info(f"Pre-training starting on device: {device}")

    # 1. Dataset & Loader (Streaming)
    dcfg = cfg["data"]
    # We use StreamingAudioDataset without a fixed transform, 
    # we'll do the augmentations in the loop for SSL.
    dataset = StreamingAudioDataset(
        manifest_path=dcfg["train_manifest"],
        sample_rate=dcfg["sample_rate"],
        segment_seconds=dcfg["segment_seconds"],
        audio_key=dcfg["manifest_audio_key"],
    )
    
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=dcfg.get("num_workers", 4),
        pin_memory=(device.type == "cuda")
    )

    # 2. Model & Loss
    # For this stub, we'll use a simple 1D backbone for raw waveforms.
    from python.penta_core.ml.training.architectures import ConvBlock
    # We'll use nn.Sequential but with 1D layers manually for the dry run stub
    backbone = torch.nn.Sequential(
        torch.nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool1d(4),
        torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(128, cfg["model"]["embedding_dim"])
    )
    
    model = MusicFoundationModel(
        backbone=backbone,
        embedding_dim=cfg["model"]["embedding_dim"],
        projection_dim=cfg["model"]["projection_dim"]
    ).to(device)

    criterion = ContrastiveLoss(temperature=cfg["ssl"]["temperature"])
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"])
    )

    # 3. Augmentor for SSL positive pairs
    augmentor = AudioAugmentor()

    # 4. Training Loop
    steps = 0
    start_time = time.time()
    spending_limit = cfg.get("spending_limit_usd", 100.0)
    cost_per_step = 0.001 # Simulated cost

    model.train()
    logger.info("Entering pre-training loop...")

    for epoch in range(cfg["training"]["epochs"]):
        for batch_idx, (audio, _) in enumerate(loader):
            # Cost Check
            current_cost = steps * cost_per_step
            if current_cost > spending_limit:
                logger.warning(f"Spending limit ${spending_limit} reached. Stopping.")
                save_checkpoint(model, optimizer, epoch, steps, loss.item(), cfg, Path("checkpoints/foundation_latest.pt"))
                return

            # Create positive pairs (two different augmentations of the same audio)
            # audio: [B, 1, T]
            x_i = audio.clone()
            x_j = audio.clone()
            
            # Apply random augmentations to each branch
            # (In a real implementation, this would be vectorized on GPU)
            # For this script, we'll assume the augmentor can handle tensors or we'll loop
            # x_i = augmentor.augment_batch(x_i) 
            
            x_i, x_j = x_i.to(device), x_j.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            # audio is [B, 1, T]
            z_i = model(x_i)
            z_j = model(x_j)
            
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()

            steps += 1

            if steps % cfg["training"]["log_every"] == 0:
                elapsed = time.time() - start_time
                logger.info(f"Epoch {epoch} | Step {steps} | Loss {loss.item():.4f} | Cost ${current_cost:.2f} | Time {elapsed:.1f}s")

            if steps % cfg["training"]["ckpt_every"] == 0:
                save_checkpoint(model, optimizer, epoch, steps, loss.item(), cfg, Path(f"checkpoints/foundation_step_{steps}.pt"))

            if steps >= cfg["training"]["max_steps"]:
                break
        if steps >= cfg["training"]["max_steps"]:
            break

    logger.info("Pre-training complete.")
    save_checkpoint(model, optimizer, epoch, steps, loss.item(), cfg, Path("checkpoints/foundation_final.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/music_foundation_base.yaml")
    args = parser.parse_args()
    pretrain(args.config)

