"""Factory for building multi-task models with extensibility"""

from typing import Dict, List, Optional, Type, Callable
import torch.nn as nn
from .base import TaskConfig, SharedEncoder, TaskHead, MultiTaskModel, LossBalancer
from .encoders import MultiModalEncoder, HierarchicalEncoder
from .heads import (
    ClassificationHead,
    RegressionHead,
    SequenceHead,
    MultiLabelHead,
    ContrastiveHead,
    RankingHead,
    CustomHead
)


class MultiTaskModelFactory:
    """Factory for building extensible multi-task models"""

    # Registry of available task head types
    _head_registry: Dict[str, Type[TaskHead]] = {
        "classification": ClassificationHead,
        "regression": RegressionHead,
        "sequence": SequenceHead,
        "multilabel": MultiLabelHead,
        "contrastive": ContrastiveHead,
        "ranking": RankingHead,
    }

    # Registry of available encoders
    _encoder_registry: Dict[str, Type[SharedEncoder]] = {
        "multimodal": MultiModalEncoder,
        "hierarchical": HierarchicalEncoder,
    }

    @classmethod
    def register_head(
        cls,
        task_type: str,
        head_class: Type[TaskHead]
    ):
        """Register custom task head"""
        cls._head_registry[task_type] = head_class

    @classmethod
    def register_encoder(
        cls,
        encoder_type: str,
        encoder_class: Type[SharedEncoder]
    ):
        """Register custom encoder"""
        cls._encoder_registry[encoder_type] = encoder_class

    @classmethod
    def create_encoder(
        cls,
        encoder_type: str,
        **kwargs
    ) -> SharedEncoder:
        """Create encoder instance"""
        if encoder_type not in cls._encoder_registry:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. "
                f"Available: {list(cls._encoder_registry.keys())}"
            )
        return cls._encoder_registry[encoder_type](**kwargs)

    @classmethod
    def create_head(
        cls,
        task_config: TaskConfig,
        input_dim: int,
        **kwargs
    ) -> TaskHead:
        """Create task head"""
        task_type = task_config.task_type

        if task_type not in cls._head_registry:
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Available: {list(cls._head_registry.keys())}"
            )

        head_class = cls._head_registry[task_type]
        return head_class(input_dim, task_config, **kwargs)

    @classmethod
    def build_model(
        cls,
        task_configs: List[TaskConfig],
        encoder_type: str = "multimodal",
        encoder_kwargs: Optional[Dict] = None,
        head_kwargs: Optional[Dict[str, Dict]] = None,
        balance_method: str = "static"
    ) -> MultiTaskModel:
        """
        Build complete multi-task model

        Args:
            task_configs: List of task configurations
            encoder_type: Type of encoder to use
            encoder_kwargs: Kwargs for encoder
            head_kwargs: Per-task kwargs for heads
            balance_method: Loss balancing method

        Returns:
            Configured MultiTaskModel
        """
        # Default kwargs
        encoder_kwargs = encoder_kwargs or {}
        head_kwargs = head_kwargs or {}

        # Create encoder
        encoder = cls.create_encoder(encoder_type, **encoder_kwargs)

        # Create task heads
        heads = {}
        for config in task_configs:
            kwargs = head_kwargs.get(config.name, {})
            head = cls.create_head(config, encoder.output_dim, **kwargs)
            heads[config.name] = head

        # Create loss balancer
        loss_balancer = LossBalancer(task_configs)

        # Create model
        model = MultiTaskModel(encoder, heads, loss_balancer)

        return model

    @classmethod
    def build_from_config(
        cls,
        config: Dict
    ) -> MultiTaskModel:
        """
        Build model from configuration dictionary

        Example config:
        {
            "tasks": [
                {
                    "name": "sentiment",
                    "task_type": "classification",
                    "output_dim": 3,
                    "weight": 1.0
                },
                ...
            ],
            "encoder": {
                "type": "multimodal",
                "modality_dims": {"text": 768, "audio": 128},
                "output_dim": 256
            },
            "loss_balance_method": "static"
        }
        """
        # Parse tasks
        task_configs = [
            TaskConfig(
                name=t["name"],
                task_type=t["task_type"],
                input_dim=t.get("input_dim", 768),
                output_dim=t["output_dim"],
                weight=t.get("weight", 1.0),
                enabled=t.get("enabled", True)
            )
            for t in config["tasks"]
        ]

        # Parse encoder
        encoder_config = config["encoder"]
        encoder_type = encoder_config.pop("type", "multimodal")
        encoder_kwargs = encoder_config.copy()

        # Build model
        return cls.build_model(
            task_configs,
            encoder_type=encoder_type,
            encoder_kwargs=encoder_kwargs
        )


class ProgressiveMigrationBuilder:
    """
    Build models with backwards compatibility and progressive migration

    Allows gradual transition from single-task to multi-task models
    """

    @staticmethod
    def from_single_task(
        original_model: nn.Module,
        additional_tasks: List[TaskConfig],
        shared_encoder_dim: int = 256
    ) -> MultiTaskModel:
        """
        Create multi-task model from existing single-task model

        Args:
            original_model: Existing trained single-task model
            additional_tasks: New tasks to add
            shared_encoder_dim: Dimension for shared encoder

        Returns:
            Multi-task model preserving original task
        """
        # TODO: Extract original task from model
        # TODO: Create shared encoder
        # TODO: Adapt original model to use shared representation
        # TODO: Add new task heads
        pass

    @staticmethod
    def create_backwards_compatible_model(
        task_configs: List[TaskConfig],
        encoder_type: str = "multimodal",
        **kwargs
    ) -> Tuple[MultiTaskModel, nn.Module]:
        """
        Create multi-task model with backwards-compatible wrapper

        Returns:
            Tuple of (MultiTaskModel, CompatibilityWrapper)
        """
        model = MultiTaskModelFactory.build_model(
            task_configs,
            encoder_type=encoder_type,
            **kwargs
        )

        wrapper = BackwardsCompatibilityWrapper(model)
        return model, wrapper


class BackwardsCompatibilityWrapper(nn.Module):
    """
    Wrapper providing backwards-compatible interface

    Allows existing code to work with multi-task models
    """

    def __init__(self, multi_task_model: MultiTaskModel):
        super().__init__()
        self.model = multi_task_model
        self.primary_task = None

    def set_primary_task(self, task_name: str):
        """Set which task output to return by default"""
        if task_name not in self.model.heads:
            raise ValueError(f"Unknown task: {task_name}")
        self.primary_task = task_name

    def forward(self, x):
        """
        Backwards-compatible forward pass

        Returns output of primary task only
        """
        outputs = self.model(x)

        if self.primary_task:
            return outputs[self.primary_task]
        else:
            # Return first task output if no primary task set
            return next(iter(outputs.values()))

    def get_all_outputs(self, x):
        """Get outputs from all tasks"""
        return self.model(x)

    def enable_task(self, task_name: str):
        """Enable additional task"""
        self.model.enable_task(task_name)

    def disable_task(self, task_name: str):
        """Disable task"""
        self.model.disable_task(task_name)
