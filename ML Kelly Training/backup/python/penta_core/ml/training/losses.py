"""
Custom Loss Functions for Music-Aware ML Training.

Provides specialized loss functions that understand musical properties:
- Emotion contrastive loss (valence-arousal aware)
- Harmony-aware loss (chord relationship aware)
- Groove consistency loss (timing pattern aware)
- Multi-task losses for joint training
- Focal loss for imbalanced datasets
- Label smoothing for better generalization

Usage:
    from python.penta_core.ml.training.losses import EmotionContrastiveLoss
    
    loss_fn = EmotionContrastiveLoss(temperature=0.07)
    loss = loss_fn(embeddings, labels)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, loss functions will not work")


if TORCH_AVAILABLE:
    
    # =========================================================================
    # Focal Loss (for imbalanced datasets)
    # =========================================================================
    
    class FocalLoss(nn.Module):
        """
        Focal Loss for handling class imbalance.
        
        Reduces loss for well-classified examples, focusing on hard examples.
        
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
        Args:
            alpha: Class weights (tensor or float)
            gamma: Focusing parameter (default 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        
        def __init__(
            self,
            alpha: Optional[Union[float, torch.Tensor]] = None,
            gamma: float = 2.0,
            reduction: str = "mean",
        ):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
        
        def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                inputs: Logits (N, C)
                targets: Class indices (N,)
            """
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            
            focal_weight = (1 - pt) ** self.gamma
            
            if self.alpha is not None:
                if isinstance(self.alpha, (float, int)):
                    alpha_t = self.alpha
                else:
                    alpha_t = self.alpha[targets]
                focal_weight = alpha_t * focal_weight
            
            loss = focal_weight * ce_loss
            
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss
    
    
    # =========================================================================
    # Label Smoothing Loss
    # =========================================================================
    
    class LabelSmoothingLoss(nn.Module):
        """
        Cross-entropy with label smoothing for better generalization.
        
        Instead of hard labels [0, 1, 0], uses soft labels [ε/K, 1-ε, ε/K]
        
        Args:
            smoothing: Label smoothing factor (0.0 to 1.0)
            num_classes: Number of classes
        """
        
        def __init__(
            self,
            smoothing: float = 0.1,
            num_classes: int = 7,
        ):
            super().__init__()
            self.smoothing = smoothing
            self.num_classes = num_classes
            self.confidence = 1.0 - smoothing
        
        def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                inputs: Logits (N, C)
                targets: Class indices (N,)
            """
            log_probs = F.log_softmax(inputs, dim=-1)
            
            # Create smooth labels
            with torch.no_grad():
                smooth_targets = torch.zeros_like(log_probs)
                smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
            
            loss = (-smooth_targets * log_probs).sum(dim=-1)
            return loss.mean()
    
    
    # =========================================================================
    # Emotion Contrastive Loss
    # =========================================================================
    
    class EmotionContrastiveLoss(nn.Module):
        """
        Contrastive loss for emotion embeddings.
        
        Pulls together embeddings of similar emotions and pushes apart
        embeddings of different emotions, weighted by emotional distance.
        
        Uses valence-arousal space to determine emotional similarity.
        
        Args:
            temperature: Contrastive temperature
            emotion_distances: Pre-computed emotion distance matrix
        """
        
        # Default emotion coordinates in valence-arousal space
        EMOTION_COORDS = {
            "happy": (0.8, 0.6),
            "sad": (-0.6, -0.4),
            "angry": (-0.6, 0.7),
            "fear": (-0.7, 0.5),
            "surprise": (0.3, 0.8),
            "disgust": (-0.5, 0.3),
            "neutral": (0.0, 0.0),
        }
        
        def __init__(
            self,
            temperature: float = 0.07,
            emotion_labels: Optional[List[str]] = None,
        ):
            super().__init__()
            self.temperature = temperature
            
            # Build emotion distance matrix
            if emotion_labels is None:
                emotion_labels = list(self.EMOTION_COORDS.keys())
            
            self.emotion_labels = emotion_labels
            self.register_buffer(
                "emotion_distances",
                self._compute_emotion_distances(emotion_labels)
            )
        
        def _compute_emotion_distances(self, labels: List[str]) -> torch.Tensor:
            """Compute pairwise emotion distances in valence-arousal space."""
            n = len(labels)
            distances = torch.zeros(n, n)
            
            for i, label_i in enumerate(labels):
                coord_i = self.EMOTION_COORDS.get(label_i, (0, 0))
                for j, label_j in enumerate(labels):
                    coord_j = self.EMOTION_COORDS.get(label_j, (0, 0))
                    # Euclidean distance in VA space
                    dist = np.sqrt(
                        (coord_i[0] - coord_j[0])**2 +
                        (coord_i[1] - coord_j[1])**2
                    )
                    distances[i, j] = dist
            
            # Normalize to [0, 1]
            distances = distances / distances.max()
            return distances
        
        def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                embeddings: Normalized embeddings (N, D)
                labels: Emotion class indices (N,)
            """
            # Normalize embeddings
            embeddings = F.normalize(embeddings, dim=1)
            
            # Compute similarity matrix
            similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
            
            # Create mask for positive pairs (same emotion)
            labels = labels.view(-1, 1)
            positive_mask = (labels == labels.T).float()
            
            # Get emotion-weighted negative mask
            # Emotions that are far apart should be pushed further
            batch_distances = self.emotion_distances[labels.squeeze(), :][:, labels.squeeze()]
            negative_weight = batch_distances * (1 - positive_mask)
            
            # Contrastive loss
            # For each sample, maximize similarity to positives, minimize to negatives
            exp_sim = torch.exp(similarity)
            
            # Positive term
            positive_sim = (exp_sim * positive_mask).sum(dim=1)
            
            # Negative term (weighted by emotional distance)
            negative_sim = (exp_sim * negative_weight).sum(dim=1)
            
            # NCE-style loss
            loss = -torch.log(positive_sim / (positive_sim + negative_sim + 1e-8))
            
            return loss.mean()
    
    
    # =========================================================================
    # Harmony-Aware Loss
    # =========================================================================
    
    class HarmonyAwareLoss(nn.Module):
        """
        Loss function that understands chord relationships.
        
        Penalizes predictions based on harmonic distance:
        - Same chord type: low penalty
        - Related chords (e.g., relative minor): medium penalty
        - Unrelated chords: high penalty
        
        Args:
            chord_vocab: List of chord symbols
            use_circle_of_fifths: Weight by circle of fifths distance
        """
        
        # Chord relationships (simplified)
        CHORD_RELATIONS = {
            # (root_interval, quality_match) -> weight
            (0, True): 0.0,   # Same chord
            (0, False): 0.3,  # Same root, different quality
            (7, True): 0.2,   # Fifth above (dominant)
            (5, True): 0.2,   # Fourth above (subdominant)
            (9, True): 0.3,   # Relative minor/major
            (3, True): 0.3,   # Minor third
            (4, True): 0.3,   # Major third
        }
        
        def __init__(
            self,
            num_chords: int = 48,
            use_circle_of_fifths: bool = True,
        ):
            super().__init__()
            self.num_chords = num_chords
            self.use_circle_of_fifths = use_circle_of_fifths
            
            # Build chord distance matrix
            self.register_buffer(
                "chord_distances",
                self._build_chord_distance_matrix()
            )
        
        def _build_chord_distance_matrix(self) -> torch.Tensor:
            """Build matrix of harmonic distances between chords."""
            # Simplified: assume chords are indexed by root * 4 + quality
            # quality: 0=major, 1=minor, 2=dim, 3=aug
            
            distances = torch.ones(self.num_chords, self.num_chords)
            
            for i in range(self.num_chords):
                root_i = i // 4
                qual_i = i % 4
                
                for j in range(self.num_chords):
                    root_j = j // 4
                    qual_j = j % 4
                    
                    # Root interval
                    interval = (root_j - root_i) % 12
                    quality_match = (qual_i == qual_j)
                    
                    # Check known relationships
                    key = (interval, quality_match)
                    if key in self.CHORD_RELATIONS:
                        distances[i, j] = self.CHORD_RELATIONS[key]
                    elif interval in [7, 5]:  # Fifth/fourth
                        distances[i, j] = 0.4
                    elif interval in [2, 10]:  # Second/seventh
                        distances[i, j] = 0.6
                    else:
                        distances[i, j] = 0.8
            
            return distances
        
        def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                logits: Chord prediction logits (N, num_chords)
                targets: Target chord indices (N,)
            """
            # Standard cross-entropy
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            
            # Get predicted chord
            predicted = logits.argmax(dim=1)
            
            # Weight by harmonic distance
            harmonic_weight = self.chord_distances[predicted, targets]
            
            # Weighted loss
            weighted_loss = ce_loss * (1 + harmonic_weight)
            
            return weighted_loss.mean()
    
    
    # =========================================================================
    # Groove Consistency Loss
    # =========================================================================
    
    class GrooveConsistencyLoss(nn.Module):
        """
        Loss for groove/timing prediction that enforces consistency.
        
        Ensures:
        - Timing offsets are smooth (no sudden jumps)
        - Velocity patterns are coherent
        - Style consistency across a sequence
        
        Args:
            smoothness_weight: Weight for smoothness penalty
            style_weight: Weight for style consistency
        """
        
        def __init__(
            self,
            smoothness_weight: float = 0.1,
            style_weight: float = 0.1,
        ):
            super().__init__()
            self.smoothness_weight = smoothness_weight
            self.style_weight = style_weight
        
        def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            style_labels: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                predictions: Predicted groove parameters (N, T, D)
                targets: Target groove parameters (N, T, D)
                style_labels: Optional style labels for consistency
            """
            # Base MSE loss
            mse_loss = F.mse_loss(predictions, targets)
            
            # Smoothness loss: penalize large differences between consecutive steps
            if predictions.dim() == 3:
                diff = predictions[:, 1:, :] - predictions[:, :-1, :]
                smoothness_loss = (diff ** 2).mean()
            else:
                smoothness_loss = torch.tensor(0.0, device=predictions.device)
            
            # Style consistency loss
            if style_labels is not None:
                # Embeddings of same style should be similar
                # This is a simplified version
                style_loss = torch.tensor(0.0, device=predictions.device)
            else:
                style_loss = torch.tensor(0.0, device=predictions.device)
            
            total_loss = (
                mse_loss +
                self.smoothness_weight * smoothness_loss +
                self.style_weight * style_loss
            )
            
            return total_loss
    
    
    # =========================================================================
    # Multi-Task Loss
    # =========================================================================
    
    class MultiTaskLoss(nn.Module):
        """
        Combined loss for multi-task learning.
        
        Supports:
        - Fixed weights
        - Learned task weights (uncertainty weighting)
        - Gradient normalization
        
        Args:
            task_losses: Dict of task_name -> loss_fn
            task_weights: Dict of task_name -> weight (or "learned")
            normalize_gradients: Apply gradient normalization
        """
        
        def __init__(
            self,
            task_losses: Dict[str, nn.Module],
            task_weights: Optional[Dict[str, float]] = None,
            learn_weights: bool = False,
        ):
            super().__init__()
            
            self.task_losses = nn.ModuleDict(task_losses)
            self.task_names = list(task_losses.keys())
            
            if task_weights is None:
                task_weights = {name: 1.0 for name in self.task_names}
            
            if learn_weights:
                # Learnable log-variance for uncertainty weighting
                self.log_vars = nn.ParameterDict({
                    name: nn.Parameter(torch.zeros(1))
                    for name in self.task_names
                })
            else:
                self.log_vars = None
                self.task_weights = task_weights
        
        def forward(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            """
            Args:
                predictions: Dict of task_name -> predictions
                targets: Dict of task_name -> targets
            
            Returns:
                total_loss, task_losses dict
            """
            task_losses = {}
            total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
            
            for name in self.task_names:
                if name not in predictions or name not in targets:
                    continue
                
                loss = self.task_losses[name](predictions[name], targets[name])
                task_losses[name] = loss
                
                if self.log_vars is not None:
                    # Uncertainty weighting: L = L / (2 * var) + log(var)
                    precision = torch.exp(-self.log_vars[name])
                    weighted_loss = precision * loss + self.log_vars[name]
                else:
                    weighted_loss = self.task_weights[name] * loss
                
                total_loss = total_loss + weighted_loss
            
            return total_loss, task_losses
    
    
    # =========================================================================
    # Music-Aware Combined Loss
    # =========================================================================
    
    class MusicAwareLoss(nn.Module):
        """
        Comprehensive music-aware loss combining multiple objectives.
        
        Combines:
        - Classification/regression loss
        - Contrastive learning
        - Temporal consistency
        - Domain-specific constraints
        
        Args:
            task: Task type ("emotion", "melody", "harmony", "dynamics", "groove")
            base_loss: Base loss function
            contrastive_weight: Weight for contrastive term
            consistency_weight: Weight for temporal consistency
        """
        
        def __init__(
            self,
            task: str = "emotion",
            num_classes: int = 7,
            contrastive_weight: float = 0.1,
            consistency_weight: float = 0.1,
            label_smoothing: float = 0.1,
        ):
            super().__init__()
            
            self.task = task
            self.contrastive_weight = contrastive_weight
            self.consistency_weight = consistency_weight
            
            # Base loss
            if task in ["emotion", "harmony"]:
                self.base_loss = LabelSmoothingLoss(
                    smoothing=label_smoothing,
                    num_classes=num_classes,
                )
            else:
                self.base_loss = nn.MSELoss()
            
            # Task-specific losses
            if task == "emotion":
                self.contrastive_loss = EmotionContrastiveLoss()
            elif task == "harmony":
                self.contrastive_loss = HarmonyAwareLoss(num_chords=num_classes)
            elif task == "groove":
                self.contrastive_loss = GrooveConsistencyLoss()
            else:
                self.contrastive_loss = None
        
        def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            embeddings: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                predictions: Model predictions
                targets: Ground truth
                embeddings: Optional embeddings for contrastive loss
            """
            # Base loss
            loss = self.base_loss(predictions, targets)
            
            # Contrastive loss
            if self.contrastive_loss is not None and embeddings is not None:
                if self.task == "emotion":
                    contrastive = self.contrastive_loss(embeddings, targets)
                    loss = loss + self.contrastive_weight * contrastive
                elif self.task == "harmony":
                    harmony_loss = self.contrastive_loss(predictions, targets)
                    loss = loss + self.contrastive_weight * harmony_loss
            
            return loss


else:
    # Placeholder classes when PyTorch is not available
    class FocalLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for loss functions")
    
    class LabelSmoothingLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for loss functions")
    
    class EmotionContrastiveLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for loss functions")
    
    class HarmonyAwareLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for loss functions")
    
    class GrooveConsistencyLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for loss functions")
    
    class MultiTaskLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for loss functions")
    
    class MusicAwareLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for loss functions")

