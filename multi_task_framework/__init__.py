"""Multi-task Learning Framework

A modular, extensible framework for multi-task learning with:
- Shared multi-modal encoder
- Independent task heads
- Loss balancing
- Progressive migration support
"""

from .base import (
    TaskConfig,
    SharedEncoder,
    TaskHead,
    LossBalancer,
    MultiTaskModel
)

from .encoders import (
    MultiModalEncoder,
    MultiHeadAttentionFusion,
    GatingFusion,
    HierarchicalEncoder
)

from .heads import (
    ClassificationHead,
    RegressionHead,
    SequenceHead,
    MultiLabelHead,
    ContrastiveHead,
    RankingHead,
    CustomHead
)

from .factory import (
    MultiTaskModelFactory,
    ProgressiveMigrationBuilder,
    BackwardsCompatibilityWrapper
)

from .trainer import MultiTaskTrainer

__version__ = "0.1.0"

__all__ = [
    # Base classes
    "TaskConfig",
    "SharedEncoder",
    "TaskHead",
    "LossBalancer",
    "MultiTaskModel",
    # Encoders
    "MultiModalEncoder",
    "MultiHeadAttentionFusion",
    "GatingFusion",
    "HierarchicalEncoder",
    # Heads
    "ClassificationHead",
    "RegressionHead",
    "SequenceHead",
    "MultiLabelHead",
    "ContrastiveHead",
    "RankingHead",
    "CustomHead",
    # Factory
    "MultiTaskModelFactory",
    "ProgressiveMigrationBuilder",
    "BackwardsCompatibilityWrapper",
    # Training
    "MultiTaskTrainer",
]
