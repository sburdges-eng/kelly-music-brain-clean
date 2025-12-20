"""
Model Registry - Unified model discovery and management.

Provides a centralized registry for ML models across different backends.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import json
import os


class ModelBackend(Enum):
    """Supported ML backends."""
    ONNX = "onnx"
    TENSORFLOW_LITE = "tflite"
    COREML = "coreml"
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"


class ModelTask(Enum):
    """Supported ML tasks."""
    CHORD_PREDICTION = "chord_prediction"
    CHORD_DETECTION = "chord_detection"
    KEY_DETECTION = "key_detection"
    TEMPO_ESTIMATION = "tempo_estimation"
    STYLE_TRANSFER = "style_transfer"
    EMOTION_CLASSIFICATION = "emotion_classification"
    AUDIO_GENERATION = "audio_generation"
    ONSET_DETECTION = "onset_detection"
    BEAT_TRACKING = "beat_tracking"


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    task: ModelTask
    backend: ModelBackend
    path: str
    version: str = "1.0.0"

    # Model metadata
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    sample_rate: Optional[int] = None

    # Performance info
    latency_ms: Optional[float] = None
    memory_mb: Optional[float] = None

    # Additional metadata
    description: str = ""
    author: str = ""
    license: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "task": self.task.value,
            "backend": self.backend.value,
            "path": self.path,
            "version": self.version,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "sample_rate": self.sample_rate,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            task=ModelTask(data["task"]),
            backend=ModelBackend(data["backend"]),
            path=data["path"],
            version=data.get("version", "1.0.0"),
            input_shape=data.get("input_shape"),
            output_shape=data.get("output_shape"),
            sample_rate=data.get("sample_rate"),
            latency_ms=data.get("latency_ms"),
            memory_mb=data.get("memory_mb"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", ""),
            tags=data.get("tags", []),
        )


class ModelRegistry:
    """
    Centralized registry for ML models.

    Manages model discovery, loading, and caching.
    """

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls) -> "ModelRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: Dict[str, ModelInfo] = {}
        self._model_dirs: List[Path] = []
        self._cache: Dict[str, Any] = {}
        self._initialized = True

        # Add default model directories
        self._add_default_dirs()

    def _add_default_dirs(self) -> None:
        """Add default model search directories."""
        # Project model directory
        project_dir = Path(__file__).parent.parent.parent.parent / "Data_Files" / "models"
        if project_dir.exists():
            self._model_dirs.append(project_dir)

        # User model directory
        user_dir = Path.home() / ".idaw" / "models"
        if user_dir.exists():
            self._model_dirs.append(user_dir)

    def add_model_dir(self, path: str) -> None:
        """Add a directory to search for models."""
        model_dir = Path(path)
        if model_dir.exists() and model_dir not in self._model_dirs:
            self._model_dirs.append(model_dir)

    def register(self, model: ModelInfo) -> None:
        """Register a model."""
        self._models[model.name] = model

    def unregister(self, name: str) -> bool:
        """Unregister a model."""
        if name in self._models:
            del self._models[name]
            return True
        return False

    def get(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self._models.get(name)

    def list(self, task: Optional[ModelTask] = None) -> List[ModelInfo]:
        """List all registered models, optionally filtered by task."""
        models = list(self._models.values())
        if task:
            models = [m for m in models if m.task == task]
        return models

    def discover(self) -> int:
        """
        Discover models in registered directories.

        Returns:
            Number of models discovered
        """
        count = 0

        for model_dir in self._model_dirs:
            # Look for model manifest files
            for manifest_path in model_dir.glob("**/model_info.json"):
                try:
                    with open(manifest_path) as f:
                        data = json.load(f)

                    # Update path to be absolute
                    if not Path(data["path"]).is_absolute():
                        data["path"] = str(manifest_path.parent / data["path"])

                    model = ModelInfo.from_dict(data)
                    self.register(model)
                    count += 1
                except Exception:
                    continue

            # Auto-discover by file extension
            for ext, backend in [
                (".onnx", ModelBackend.ONNX),
                (".tflite", ModelBackend.TENSORFLOW_LITE),
                (".mlmodel", ModelBackend.COREML),
                (".pt", ModelBackend.PYTORCH),
                (".pth", ModelBackend.PYTORCH),
            ]:
                for model_path in model_dir.glob(f"**/*{ext}"):
                    name = model_path.stem
                    if name not in self._models:
                        # Infer task from directory name or file name
                        task = self._infer_task(model_path)
                        model = ModelInfo(
                            name=name,
                            task=task,
                            backend=backend,
                            path=str(model_path),
                        )
                        self.register(model)
                        count += 1

        return count

    def _infer_task(self, path: Path) -> ModelTask:
        """Infer model task from path."""
        path_str = str(path).lower()

        if "chord" in path_str:
            if "detect" in path_str:
                return ModelTask.CHORD_DETECTION
            return ModelTask.CHORD_PREDICTION
        elif "key" in path_str:
            return ModelTask.KEY_DETECTION
        elif "tempo" in path_str or "bpm" in path_str:
            return ModelTask.TEMPO_ESTIMATION
        elif "style" in path_str or "transfer" in path_str:
            return ModelTask.STYLE_TRANSFER
        elif "emotion" in path_str or "mood" in path_str:
            return ModelTask.EMOTION_CLASSIFICATION
        elif "onset" in path_str:
            return ModelTask.ONSET_DETECTION
        elif "beat" in path_str:
            return ModelTask.BEAT_TRACKING

        return ModelTask.CHORD_PREDICTION  # Default

    def save_registry(self, path: str) -> None:
        """Save registry to JSON file."""
        data = {
            "models": [m.to_dict() for m in self._models.values()],
            "model_dirs": [str(d) for d in self._model_dirs],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_registry(self, path: str) -> None:
        """Load registry from JSON file."""
        with open(path) as f:
            data = json.load(f)

        for model_data in data.get("models", []):
            model = ModelInfo.from_dict(model_data)
            self.register(model)

        for dir_path in data.get("model_dirs", []):
            self.add_model_dir(dir_path)


# Singleton access functions
def get_registry() -> ModelRegistry:
    """Get the model registry singleton."""
    return ModelRegistry()


def register_model(model: ModelInfo) -> None:
    """Register a model in the global registry."""
    get_registry().register(model)


def get_model(name: str) -> Optional[ModelInfo]:
    """Get a model from the global registry."""
    return get_registry().get(name)


def list_models(task: Optional[ModelTask] = None) -> List[ModelInfo]:
    """List models in the global registry."""
    return get_registry().list(task)
