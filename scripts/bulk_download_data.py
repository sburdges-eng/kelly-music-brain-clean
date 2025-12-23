#!/usr/bin/env python3
"""
Bulk Audio Downloader and Labeler for Kelly project.

Limits:
- Soft target: ~15 GB total download
- Hard cap: 1000 GB absolute safety stop
- Per-dataset timeout + failure circuit breaker to avoid hangs/loops

Automatically labels downloads by running preprocessing (resampling, mel-spec,
metadata generation) for each dataset.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# pylint: disable=wrong-import-position
from scripts.prepare_datasets import DATASETS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SOFT_TARGET_GB = 2500      # 2.5 TB soft target
HARD_CAP_GB = 3000         # 3 TB absolute safety stop
HUMAN_READABLE_HARD_CAP = f"{HARD_CAP_GB} GB (3 TB)"

AUDIO_DATA_ROOT = Path("/Volumes/Extreme SSD/kelly-audio-data")
MAX_CONSECUTIVE_FAILURES = 5
DOWNLOAD_TIMEOUT_SECONDS = 3600 * 24 # 24 hours per dataset max for TB scale

# Rough dataset size estimates (GB).
DATASET_ESTIMATES_GB = {
    "emotion_ravdess": 1,
    "emotion_cremad": 5,
    "emotion_tess": 1,
    "groove_midi": 1,
    "maestro": 20,
    "lakh_midi": 30,
    "nsynth_full": 30,
    "musdb18": 10,
    "fma_small": 8,
    "fma_medium": 22,
    "fma_full": 900,
    "musicnet": 168,
    "mtg_jamendo": 1000,
    "local_music": 50,
}

# Datasets to download in order of priority
# Balanced for diversity first, then mass scale
TARGET_DATASETS = [
    "emotion_ravdess",
    "emotion_cremad",
    "emotion_tess",
    "groove_midi",
    "maestro",
    "lakh_midi",
    "nsynth_full",
    "musdb18",
    "gtzan",
    "fma_small",
    "fma_medium",
    "musicnet",
    "fma_full",
    "mtg_jamendo",
]


def get_current_size_gb(path: Path) -> float:
    """Calculate current size of directory in GB with safe traversal."""
    if not path.exists():
        return 0.0

    total_size = 0
    try:
        for f in path.rglob('*'):
            if f.is_file():
                total_size += f.stat().st_size
    except Exception as e:
        logger.warning(f"Error calculating size: {e}")

    return total_size / (1024**3)


def download_and_preprocess_dataset(name: str) -> bool:
    """Download and label/preprocess a dataset."""
    logger.info(f"Starting process for dataset: {name}")
    try:
        # 1. Download
        logger.info(f"  Downloading {name}...")
        download_cmd = [
            sys.executable, "scripts/prepare_datasets.py",
            "--dataset", name, "--download"
        ]
        subprocess.run(
            download_cmd, check=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
        )

        # 2. Preprocess (Labeling)
        logger.info(f"  Labeling/Preprocessing {name}...")
        preprocess_cmd = [
            sys.executable, "scripts/prepare_datasets.py",
            "--dataset", name, "--preprocess"
        ]
        subprocess.run(
            preprocess_cmd, check=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
        )

        logger.info(f"Successfully finished processing: {name}")
        return True

    except subprocess.TimeoutExpired:
        logger.error(
            f"Process for {name} timed out after "
            f"{DOWNLOAD_TIMEOUT_SECONDS} seconds."
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to process {name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {name}: {e}")
        return False


def main() -> None:
    """Run the bulk download process."""
    if not AUDIO_DATA_ROOT.parent.exists():
        logger.error("External SSD not mounted at /Volumes/Extreme SSD")
        logger.info("Please ensure the SSD is connected and try again.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(
        "Bulk Audio Downloader "
        f"(soft target {SOFT_TARGET_GB} GB, hard cap {HARD_CAP_GB} GB)"
    )
    logger.info("=" * 60)

    # Safety: ensure we don't start in an infinite loop
    start_time = time.time()
    consecutive_failures = 0

    current_size = get_current_size_gb(AUDIO_DATA_ROOT)
    logger.info(
        f"Initial storage usage: {current_size:.2f} GB "
        f"(soft target {SOFT_TARGET_GB} GB, "
        f"hard cap {HUMAN_READABLE_HARD_CAP})"
    )

    for i, name in enumerate(TARGET_DATASETS):
        # Progress report
        logger.info(
            f"\n[Dataset {i+1}/{len(TARGET_DATASETS)}] Processing: {name}"
        )

        if name not in DATASETS:
            logger.warning(f"Dataset {name} not found in configs. Skipping.")
            continue

        # Stop if we've already hit the soft target
        if current_size >= SOFT_TARGET_GB:
            logger.info(
                f"Reached soft target ({current_size:.2f} GB >= "
                f"{SOFT_TARGET_GB} GB). Stopping."
            )
            break

        # Check against hard cap before attempting download
        if current_size >= HARD_CAP_GB:
            logger.info(
                f"Reached or exceeded hard cap ({current_size:.2f} GB >= "
                f"{HARD_CAP_GB} GB). Stopping."
            )
            break

        est = DATASET_ESTIMATES_GB.get(name)
        if est is not None and current_size + est > HARD_CAP_GB:
            logger.warning(
                f"Skipping {name}: estimated {est} GB would exceed hard cap "
                f"({current_size:.2f} + {est} > {HARD_CAP_GB})."
            )
            continue

        # Safety: check if script has been running for too long
        total_runtime_hours = (time.time() - start_time) / 3600
        if total_runtime_hours > 48:
            logger.warning(
                f"Bulk script has been running for {total_runtime_hours:.1f} "
                "hours. Safety abort."
            )
            break

        success = download_and_preprocess_dataset(name)

        if success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            logger.warning(f"Consecutive failures: {consecutive_failures}")

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.error(
                f"Reached {MAX_CONSECUTIVE_FAILURES} consecutive failures. "
                "Circuit breaker triggered. Aborting."
            )
            break

        # Recalculate size after each download
        current_size = get_current_size_gb(AUDIO_DATA_ROOT)
        logger.info(
            f"Updated total size: {current_size:.2f} GB "
            f"(soft target {SOFT_TARGET_GB} GB, hard cap {HARD_CAP_GB} GB)"
        )

    total_time = (time.time() - start_time) / 3600
    logger.info("=" * 60)
    logger.info("Bulk download run complete.")
    logger.info(f"Final size: {current_size:.2f} GB")
    logger.info(f"Total time elapsed: {total_time:.2f} hours")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
