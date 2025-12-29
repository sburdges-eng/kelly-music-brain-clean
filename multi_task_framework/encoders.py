"""Multi-modal encoder implementations"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import SharedEncoder


class MultiModalEncoder(SharedEncoder):
    """Generic multi-modal encoder with fusion"""

    def __init__(
        self,
        modality_dims: Dict[str, int],
        output_dim: int,
        hidden_dim: int = 256,
        fusion_method: str = "concat"
    ):
        super().__init__(output_dim)
        self.modality_dims = modality_dims
        self.fusion_method = fusion_method

        # Modality-specific projection layers
        self.projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            for modality, dim in modality_dims.items()
        })

        # Fusion module
        if fusion_method == "concat":
            fusion_input_dim = hidden_dim * len(modality_dims)
        elif fusion_method == "attention":
            fusion_input_dim = hidden_dim
        else:
            fusion_input_dim = hidden_dim

        self.fusion_module = self._build_fusion(
            fusion_method, hidden_dim, fusion_input_dim
        )

        # Final projection to output dimension
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def _build_fusion(
        self,
        method: str,
        hidden_dim: int,
        fusion_input_dim: int
    ) -> nn.Module:
        """Build fusion module based on method"""
        if method == "concat":
            return nn.Identity()  # Concatenation happens in forward
        elif method == "attention":
            return MultiHeadAttentionFusion(hidden_dim)
        elif method == "gating":
            return GatingFusion(hidden_dim)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode multi-modal input to unified representation

        Args:
            x: Dictionary of {modality_name: tensor}

        Returns:
            Unified representation tensor
        """
        # Project each modality
        projections = {}
        for modality, tensor in x.items():
            if modality in self.projections:
                projections[modality] = self.projections[modality](tensor)

        # Fuse representations
        if self.fusion_method == "concat":
            fused = torch.cat(list(projections.values()), dim=-1)
        elif self.fusion_method in ["attention", "gating"]:
            fused = self.fusion_module(list(projections.values()))
        else:
            fused = torch.mean(torch.stack(list(projections.values())), dim=0)

        # Project to output dimension
        output = self.output_projection(fused)
        return output


class MultiHeadAttentionFusion(nn.Module):
    """Attention-based fusion of modalities"""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, modality_reps: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse modality representations using attention

        Args:
            modality_reps: List of modality representations

        Returns:
            Fused representation
        """
        # Stack as sequence
        stacked = torch.stack(modality_reps, dim=1)  # [batch, num_modalities, hidden_dim]

        # Self-attention across modalities
        attended, _ = self.attention(stacked, stacked, stacked)
        attended = self.norm(attended)

        # Average across modalities
        fused = attended.mean(dim=1)
        return fused


class GatingFusion(nn.Module):
    """Gating-based fusion of modalities"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for _ in range(2)  # For each modality pair
        ])

    def forward(self, modality_reps: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse using learnable gating

        Args:
            modality_reps: List of modality representations

        Returns:
            Fused representation
        """
        if len(modality_reps) == 0:
            raise ValueError("No modality representations provided")

        fused = modality_reps[0]
        for i, rep in enumerate(modality_reps[1:]):
            combined = torch.cat([fused, rep], dim=-1)
            gate = self.gate_layers[i % len(self.gate_layers)](combined)
            fused = gate * fused + (1 - gate) * rep

        return fused


class HierarchicalEncoder(SharedEncoder):
    """Hierarchical encoder with modality grouping"""

    def __init__(
        self,
        modality_groups: Dict[str, List[str]],
        modality_dims: Dict[str, int],
        output_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__(output_dim)
        self.modality_groups = modality_groups
        self.modality_dims = modality_dims

        # Group-level encoders
        self.group_encoders = nn.ModuleDict()
        for group_name, modalities in modality_groups.items():
            group_dims = {m: modality_dims[m] for m in modalities}
            self.group_encoders[group_name] = MultiModalEncoder(
                group_dims, hidden_dim, hidden_dim, fusion_method="attention"
            )

        # Cross-group fusion
        num_groups = len(modality_groups)
        self.group_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_groups, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode through hierarchy"""
        group_outputs = {}
        for group_name, modalities in self.modality_groups.items():
            group_input = {m: x[m] for m in modalities if m in x}
            if group_input:
                group_outputs[group_name] = self.group_encoders[group_name](group_input)

        # Fuse group outputs
        fused = torch.cat(list(group_outputs.values()), dim=-1)
        output = self.group_fusion(fused)
        return output
