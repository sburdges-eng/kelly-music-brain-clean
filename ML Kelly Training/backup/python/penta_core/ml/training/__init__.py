"""
Kelly ML Training Utilities.

Provides comprehensive tools for music ML training:

Modules:
- augmentation: Advanced audio data augmentation
- losses: Music-aware custom loss functions  
- architectures: Attention, residual, and multi-task models
- evaluation: Metrics and validation algorithms

Usage:
    from python.penta_core.ml.training import (
        AudioAugmentor,
        MusicAwareLoss,
        EmotionCNN,
        evaluate_model,
    )
"""

# Augmentation
from .augmentation import (
    AudioAugmentor,
    AugmentationConfig,
    AugmentationPipeline,
    augment_audio,
    create_augmentation_pipeline,
)

# Losses
from .losses import (
    MusicAwareLoss,
    EmotionContrastiveLoss,
    HarmonyAwareLoss,
    GrooveConsistencyLoss,
    MultiTaskLoss,
    FocalLoss,
    LabelSmoothingLoss,
)

# Architectures
from .architectures import (
    AttentionBlock,
    MultiHeadAttention,
    ConvBlock,
    ResidualBlock,
    EmotionCNN,
    MelodyLSTM,
    HarmonyMLP,
    MultiTaskModel,
)

# Evaluation
from .evaluation import (
    BaseMetrics,
    MetricResult,
    MusicMetrics,
    EmotionMetrics,
    GrooveMetrics,
    ModelValidator,
    evaluate_model,
    cross_validate,
    compute_confusion_matrix,
    plot_confusion_matrix,
)

__all__ = [
    # Augmentation
    "AudioAugmentor",
    "AugmentationConfig",
    "AugmentationPipeline",
    "augment_audio",
    "create_augmentation_pipeline",
    # Losses
    "MusicAwareLoss",
    "EmotionContrastiveLoss",
    "HarmonyAwareLoss",
    "GrooveConsistencyLoss",
    "MultiTaskLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    # Architectures
    "AttentionBlock",
    "MultiHeadAttention",
    "ConvBlock",
    "ResidualBlock",
    "EmotionCNN",
    "MelodyLSTM",
    "HarmonyMLP",
    "MultiTaskModel",
    # Evaluation
    "BaseMetrics",
    "MetricResult",
    "MusicMetrics",
    "EmotionMetrics",
    "GrooveMetrics",
    "ModelValidator",
    "evaluate_model",
    "cross_validate",
    "compute_confusion_matrix",
    "plot_confusion_matrix",
]
