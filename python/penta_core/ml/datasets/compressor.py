"""Dataset compressor utilities for server-ready archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import json
import tarfile
import time


@dataclass(frozen=True)
class DatasetArchiveManifest:
    """Metadata for a compressed dataset bundle."""
    created_at: float
    datasets: List[Dict[str, str]]
    total_bytes: int


class DatasetCompressor:
    """Create compressed archives of datasets with embedded manifests."""

    def build_manifest(self, dataset_paths: Iterable[Path]) -> DatasetArchiveManifest:
        datasets: List[Dict[str, str]] = []
        total_bytes = 0

        for dataset_path in dataset_paths:
            dataset_path = Path(dataset_path)
            size_bytes = _directory_size(dataset_path)
            total_bytes += size_bytes
            datasets.append({
                "path": str(dataset_path),
                "bytes": str(size_bytes),
            })

        return DatasetArchiveManifest(
            created_at=time.time(),
            datasets=datasets,
            total_bytes=total_bytes,
        )

    def create_archive(self, output_path: Path, dataset_paths: Iterable[Path]) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.build_manifest(dataset_paths)
        manifest_bytes = json.dumps(manifest.__dict__, indent=2).encode("utf-8")

        with tarfile.open(output_path, "w:gz") as tar:
            for dataset_path in dataset_paths:
                dataset_path = Path(dataset_path)
                tar.add(dataset_path, arcname=dataset_path.name)
            manifest_info = tarfile.TarInfo("manifest.json")
            manifest_info.size = len(manifest_bytes)
            tar.addfile(manifest_info, fileobj=_bytes_io(manifest_bytes))

        return output_path


def _directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _bytes_io(payload: bytes):
    import io

    return io.BytesIO(payload)
