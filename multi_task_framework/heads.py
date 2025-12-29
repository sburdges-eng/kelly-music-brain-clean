"""Task-specific head implementations"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TaskHead, TaskConfig


class ClassificationHead(TaskHead):
    """Classification task head"""

    def __init__(
        self,
        input_dim: int,
        task_config: TaskConfig,
        hidden_dim: int = 128
    ):
        super().__init__(input_dim, task_config)
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, task_config.output_dim)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """Return logits"""
        return self.network(shared_repr)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(predictions, targets)


class RegressionHead(TaskHead):
    """Regression task head"""

    def __init__(
        self,
        input_dim: int,
        task_config: TaskConfig,
        hidden_dim: int = 128
    ):
        super().__init__(input_dim, task_config)
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, task_config.output_dim)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """Return predictions"""
        return self.network(shared_repr)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(predictions, targets)


class SequenceHead(TaskHead):
    """Sequence labeling task head"""

    def __init__(
        self,
        input_dim: int,
        task_config: TaskConfig,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__(input_dim, task_config)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_dim * 2, task_config.output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """Process sequence through LSTM"""
        # Handle 2D or 3D input
        if shared_repr.dim() == 2:
            shared_repr = shared_repr.unsqueeze(1)

        lstm_out, _ = self.lstm(shared_repr)
        logits = self.classifier(lstm_out)
        return logits

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Flatten for loss computation
        batch_size, seq_len, num_classes = predictions.shape
        return self.loss_fn(
            predictions.view(-1, num_classes),
            targets.view(-1)
        )


class MultiLabelHead(TaskHead):
    """Multi-label classification head"""

    def __init__(
        self,
        input_dim: int,
        task_config: TaskConfig,
        hidden_dim: int = 128
    ):
        super().__init__(input_dim, task_config)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, task_config.output_dim)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """Return logits for multi-label classification"""
        return self.network(shared_repr)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(predictions, targets.float())


class ContrastiveHead(TaskHead):
    """Contrastive learning head"""

    def __init__(
        self,
        input_dim: int,
        task_config: TaskConfig,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__(input_dim, task_config)
        self.projection_dim = projection_dim
        self.temperature = temperature

        self.projector = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """Return normalized projections"""
        proj = self.projector(shared_repr)
        return F.normalize(proj, dim=-1)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """NT-Xent loss for contrastive learning"""
        # Simple cosine similarity-based loss
        logits = torch.mm(predictions, predictions.t()) / self.temperature
        labels = torch.eye(logits.shape[0], device=logits.device)
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels.argmax(dim=1))


class RankingHead(TaskHead):
    """Ranking/metric learning head"""

    def __init__(
        self,
        input_dim: int,
        task_config: TaskConfig,
        margin: float = 0.5
    ):
        super().__init__(input_dim, task_config)
        self.margin = margin
        self.projector = nn.Linear(input_dim, task_config.output_dim)

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """Return embeddings for ranking"""
        return F.normalize(self.projector(shared_repr), dim=-1)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Triplet loss for ranking"""
        # predictions: [batch_size, embedding_dim]
        # targets: [batch_size] indicating positive/negative pairs
        anchor = predictions[::3]
        positive = predictions[1::3]
        negative = predictions[2::3]

        pos_dist = torch.norm(anchor - positive, dim=-1)
        neg_dist = torch.norm(anchor - negative, dim=-1)

        loss = torch.clamp(self.margin + pos_dist - neg_dist, min=0)
        return loss.mean()


class CustomHead(TaskHead):
    """Custom task head with user-defined architecture"""

    def __init__(
        self,
        input_dim: int,
        task_config: TaskConfig,
        network: nn.Module,
        loss_fn: nn.Module
    ):
        super().__init__(input_dim, task_config)
        self.network = network
        self.loss_fn = loss_fn

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        return self.network(shared_repr)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(predictions, targets)
