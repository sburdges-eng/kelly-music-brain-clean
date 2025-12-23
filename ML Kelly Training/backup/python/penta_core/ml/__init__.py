# ruff: noqa: I001
"""
ML Model Integration - Machine Learning inference for iDAW.

Provides unified interfaces for:
- ONNX Runtime (cross-platform)
- TensorFlow Lite (mobile/edge)
- CoreML (macOS/iOS)
- PyTorch (training and inference)

Supports:
- Chord prediction models
- Style transfer for groove
- Emotion classification
- Audio feature extraction
- Mac-optimized training (MPS/CPU)
- Model export (ONNX, Core ML, RTNeural)

See docs/MK_TRAINING_GUIDELINES.md for training workflow.
"""

from python.penta_core.ml.model_registry import (
    ModelRegistry,
    ModelInfo,
    ModelBackend,
    load_registry_manifest,
    register_model,
    get_model,
    list_models,
)

from python.penta_core.ml.inference import (
    InferenceEngine,
    InferenceResult,
    create_engine,
)

from python.penta_core.ml.chord_predictor import (
    ChordPredictor,
    ChordPrediction,
    predict_next_chord,
    predict_progression,
)

from python.penta_core.ml.style_transfer import (
    GrooveStyleTransfer,
    StyleTransferResult,
    transfer_groove_style,
)

from python.penta_core.ml.gpu_utils import (
    get_available_devices,
    select_best_device,
    GPUDevice,
    DeviceType,
)

# Training and export utilities (optional imports)
try:
    from python.penta_core.ml.audio_dataset import (  # noqa: F401
        AudioDataset,
        AudioDatasetTorch,
        AudioSample,
        create_dataloaders,
        create_metadata_template,
    )
    _HAS_DATASET = True
except ImportError:
    _HAS_DATASET = False

try:
    from python.penta_core.ml.export import (  # noqa: F401
        ModelExporter,
        ExportConfig,
        verify_onnx_model,
        verify_coreml_model,
    )
    _HAS_EXPORT = True
except ImportError:
    _HAS_EXPORT = False

__all__ = [
    # Registry
    "ModelRegistry",
    "ModelInfo",
    "ModelBackend",
    "load_registry_manifest",
    "register_model",
    "get_model",
    "list_models",
    # Inference
    "InferenceEngine",
    "InferenceResult",
    "create_engine",
    # Chord Prediction
    "ChordPredictor",
    "ChordPrediction",
    "predict_next_chord",
    "predict_progression",
    # Style Transfer
    "GrooveStyleTransfer",
    "StyleTransferResult",
    "transfer_groove_style",
    # GPU
    "get_available_devices",
    "select_best_device",
    "GPUDevice",
    "DeviceType",
]

# Add dataset exports if available
if _HAS_DATASET:
    __all__.extend([
        "AudioDataset",
        "AudioDatasetTorch",
        "AudioSample",
        "create_dataloaders",
        "create_metadata_template",
    ])

# Add export utilities if available
if _HAS_EXPORT:
    __all__.extend([
        "ModelExporter",
        "ExportConfig",
        "verify_onnx_model",
        "verify_coreml_model",
    ])
