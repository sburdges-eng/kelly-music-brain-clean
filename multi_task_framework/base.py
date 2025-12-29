"""Base classes for multi-task learning framework"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Configuration for a single task"""
    name: str
    task_type: str  # 'classification', 'regression', 'sequence', etc.
    input_dim: int
    output_dim: int
    weight: float = 1.0
    enabled: bool = True
    loss_fn: Optional[str] = None


class SharedEncoder(nn.Module, ABC):
    """Base class for shared multi-modal encoder"""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self._frozen = False

    @abstractmethod
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process multi-modal input and return unified representation

        Args:
            x: Dictionary of modal inputs {modality: tensor}

        Returns:
            Unified representation tensor
        """
        pass

    def freeze(self):
        """Freeze encoder parameters"""
        self._frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze encoder parameters"""
        self._frozen = False
        for param in self.parameters():
            param.requires_grad = True

    @property
    def frozen(self) -> bool:
        return self._frozen


class TaskHead(nn.Module, ABC):
    """Base class for task-specific head"""

    def __init__(self, input_dim: int, task_config: TaskConfig):
        super().__init__()
        self.input_dim = input_dim
        self.config = task_config
        self.enabled = task_config.enabled

    @abstractmethod
    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """
        Process shared representation for task-specific output

        Args:
            shared_repr: Output from shared encoder

        Returns:
            Task-specific predictions
        """
        pass

    def enable(self):
        """Enable task"""
        self.enabled = True

    def disable(self):
        """Disable task"""
        self.enabled = False

    @abstractmethod
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute task-specific loss"""
        pass


class LossBalancer(nn.Module):
    """Weighted multi-task loss balancing"""

    def __init__(self, task_configs: List[TaskConfig]):
        super().__init__()
        self.task_configs = {cfg.name: cfg for cfg in task_configs}
        self.num_tasks = len(task_configs)

        # Learnable loss weights (optional dynamic weighting)
        self.log_vars = nn.ParameterDict({
            cfg.name: nn.Parameter(torch.zeros(1))
            for cfg in task_configs
        })

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        method: str = "static"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Balance losses across tasks

        Args:
            losses: Dictionary of task losses
            method: 'static' (fixed weights) or 'dynamic' (learnable)

        Returns:
            Tuple of (weighted_loss, loss_dict)
        """
        if method == "static":
            return self._static_balance(losses)
        elif method == "dynamic":
            return self._dynamic_balance(losses)
        else:
            raise ValueError(f"Unknown balancing method: {method}")

    def _static_balance(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Static weighting based on task config"""
        weighted_losses = {}
        total_loss = 0.0

        for task_name, loss in losses.items():
            weight = self.task_configs[task_name].weight
            weighted_loss = weight * loss
            weighted_losses[task_name] = weighted_loss.item()
            total_loss = total_loss + weighted_loss

        return total_loss, weighted_losses

    def _dynamic_balance(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Dynamic weighting using learnable uncertainty"""
        weighted_losses = {}
        total_loss = 0.0

        for task_name, loss in losses.items():
            log_var = self.log_vars[task_name]
            precision = torch.exp(-log_var)
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            weighted_losses[task_name] = weighted_loss.item()
            total_loss = total_loss + weighted_loss

        return total_loss, weighted_losses

    def update_weight(self, task_name: str, new_weight: float):
        """Update static weight for a task"""
        if task_name in self.task_configs:
            self.task_configs[task_name].weight = new_weight


class MultiTaskModel(nn.Module, ABC):
    """Base multi-task learning model"""

    def __init__(
        self,
        encoder: SharedEncoder,
        heads: Dict[str, TaskHead],
        loss_balancer: LossBalancer
    ):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)
        self.loss_balancer = loss_balancer
        self.active_tasks = set(heads.keys())

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder and all active task heads

        Args:
            x: Dictionary of multi-modal inputs

        Returns:
            Dictionary of task outputs
        """
        # Shared encoding
        shared_repr = self.encoder(x)

        # Task-specific outputs
        outputs = {}
        for task_name, head in self.heads.items():
            if head.enabled:
                outputs[task_name] = head(shared_repr)

        return outputs

    def compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute losses for all active tasks"""
        losses = {}
        for task_name, pred in predictions.items():
            if task_name in self.heads and self.heads[task_name].enabled:
                losses[task_name] = self.heads[task_name].compute_loss(
                    pred, targets[task_name]
                )
        return losses

    def enable_task(self, task_name: str):
        """Enable a task"""
        if task_name in self.heads:
            self.heads[task_name].enable()
            self.active_tasks.add(task_name)

    def disable_task(self, task_name: str):
        """Disable a task"""
        if task_name in self.heads:
            self.heads[task_name].disable()
            self.active_tasks.discard(task_name)

    def get_active_tasks(self) -> List[str]:
        """Get list of active tasks"""
        return list(self.active_tasks)
