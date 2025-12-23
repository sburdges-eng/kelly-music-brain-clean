"""
Model Architecture Improvements for Kelly ML.

Provides advanced neural network components:
- Attention mechanisms (self-attention, multi-head)
- Residual blocks
- Multi-task learning architectures
- Task-specific models (emotion, melody, harmony, dynamics, groove)

Usage:
    from python.penta_core.ml.training.architectures import EmotionCNN, MultiTaskModel
    
    model = EmotionCNN(num_classes=7)
    multi_model = MultiTaskModel(tasks=["emotion", "genre"])
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, architectures will not work")


if TORCH_AVAILABLE:
    
    # =========================================================================
    # Basic Building Blocks
    # =========================================================================
    
    class ConvBlock(nn.Module):
        """
        Convolutional block with batch norm and activation.
        
        Conv -> BatchNorm -> Activation -> (optional) Dropout
        """
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            activation: str = "relu",
            dropout: float = 0.0,
            use_bn: bool = True,
        ):
            super().__init__()
            
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            ]
            
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            
            self.block = nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)
    
    
    class ResidualBlock(nn.Module):
        """
        Residual block with skip connection.
        
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
        """
        
        def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            dropout: float = 0.0,
        ):
            super().__init__()
            
            padding = kernel_size // 2
            
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
            self.bn2 = nn.BatchNorm2d(channels)
            self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            
            out = out + residual
            out = F.relu(out)
            
            return out
    
    
    # =========================================================================
    # Attention Mechanisms
    # =========================================================================
    
    class AttentionBlock(nn.Module):
        """
        Self-attention block for sequence or spatial data.
        
        Computes attention weights and applies to values.
        """
        
        def __init__(
            self,
            dim: int,
            num_heads: int = 1,
            dropout: float = 0.0,
        ):
            super().__init__()
            
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.proj = nn.Linear(dim, dim)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = self.head_dim ** -0.5
        
        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                x: Input tensor (batch, seq_len, dim)
                mask: Optional attention mask
            
            Returns:
                Attended output (batch, seq_len, dim)
            """
            B, N, C = x.shape
            
            # Compute Q, K, V
            q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention scores
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention to values
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            
            return out
    
    
    class MultiHeadAttention(nn.Module):
        """
        Multi-head attention with optional positional encoding.
        """
        
        def __init__(
            self,
            d_model: int,
            num_heads: int = 8,
            dropout: float = 0.1,
            use_positional: bool = True,
            max_len: int = 512,
        ):
            super().__init__()
            
            self.attention = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
            
            self.use_positional = use_positional
            if use_positional:
                self.positional = self._create_positional_encoding(max_len, d_model)
            
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        
        def _create_positional_encoding(
            self,
            max_len: int,
            d_model: int,
        ) -> nn.Parameter:
            """Create sinusoidal positional encoding."""
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        
        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                x: Input (batch, seq_len, d_model)
                mask: Optional attention mask
            
            Returns:
                Output (batch, seq_len, d_model)
            """
            # Add positional encoding
            if self.use_positional:
                x = x + self.positional[:, :x.size(1), :]
            
            # Self-attention with residual
            attn_out, _ = self.attention(x, x, x, attn_mask=mask)
            x = self.norm(x + self.dropout(attn_out))
            
            return x
    
    
    class TemporalAttention(nn.Module):
        """
        Temporal attention for audio/music sequences.
        
        Learns to attend to important time steps.
        """
        
        def __init__(self, input_dim: int, hidden_dim: int = 64):
            super().__init__()
            
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: Input (batch, seq_len, input_dim)
            
            Returns:
                Weighted output (batch, input_dim), attention weights (batch, seq_len)
            """
            # Compute attention weights
            attn_weights = self.attention(x).squeeze(-1)  # (batch, seq_len)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Weighted sum
            weighted = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
            
            return weighted, attn_weights
    
    
    # =========================================================================
    # Task-Specific Architectures
    # =========================================================================
    
    class EmotionCNN(nn.Module):
        """
        CNN for emotion recognition from mel spectrograms.
        
        Architecture:
        - Conv blocks with increasing channels
        - Attention pooling
        - Classification head
        """
        
        def __init__(
            self,
            num_classes: int = 7,
            in_channels: int = 1,
            base_channels: int = 32,
            dropout: float = 0.3,
            use_attention: bool = True,
        ):
            super().__init__()
            
            self.use_attention = use_attention
            
            # Convolutional backbone
            self.conv1 = ConvBlock(in_channels, base_channels, 3, padding=1)
            self.pool1 = nn.MaxPool2d(2)
            
            self.conv2 = ConvBlock(base_channels, base_channels * 2, 3, padding=1)
            self.pool2 = nn.MaxPool2d(2)
            
            self.conv3 = ConvBlock(base_channels * 2, base_channels * 4, 3, padding=1)
            self.res3 = ResidualBlock(base_channels * 4)
            self.pool3 = nn.MaxPool2d(2)
            
            self.conv4 = ConvBlock(base_channels * 4, base_channels * 8, 3, padding=1)
            self.res4 = ResidualBlock(base_channels * 8)
            
            # Global pooling
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Attention (optional)
            if use_attention:
                self.channel_attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(base_channels * 8, base_channels * 2),
                    nn.ReLU(),
                    nn.Linear(base_channels * 2, base_channels * 8),
                    nn.Sigmoid(),
                )
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_channels * 8, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
            
            # Embedding output (before classifier)
            self.embedding_dim = 128
        
        def forward(
            self,
            x: torch.Tensor,
            return_embedding: bool = False,
        ) -> torch.Tensor:
            """
            Args:
                x: Mel spectrogram (batch, 1, n_mels, time)
                return_embedding: Return embedding instead of logits
            
            Returns:
                Logits (batch, num_classes) or embedding (batch, 128)
            """
            # Convolutional layers
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.res3(self.conv3(x)))
            x = self.res4(self.conv4(x))
            
            # Channel attention
            if self.use_attention:
                attn = self.channel_attention(x).unsqueeze(-1).unsqueeze(-1)
                x = x * attn
            
            # Global pooling
            x = self.global_pool(x)
            x = x.flatten(1)
            
            if return_embedding:
                # Return intermediate embedding
                x = self.classifier[0](x)  # Flatten
                x = self.classifier[1](x)  # Linear 256
                x = self.classifier[2](x)  # ReLU
                x = self.classifier[3](x)  # Dropout
                x = self.classifier[4](x)  # Linear 128
                return x
            
            return self.classifier(x)
    
    
    class MelodyLSTM(nn.Module):
        """
        LSTM with attention for melody generation.
        
        Architecture:
        - Embedding layer for note tokens
        - Bidirectional LSTM
        - Attention mechanism
        - Output head for next-note prediction
        """
        
        def __init__(
            self,
            vocab_size: int = 128,  # MIDI notes
            embedding_dim: int = 64,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_classes: int = 128,
            dropout: float = 0.2,
            bidirectional: bool = False,
        ):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            
            lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
            
            self.attention = TemporalAttention(lstm_output_dim, hidden_dim // 2)
            
            self.classifier = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        
        def forward(
            self,
            x: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                x: Note indices (batch, seq_len)
                lengths: Optional sequence lengths for packing
            
            Returns:
                Logits for next note (batch, num_classes)
            """
            # Embed notes
            x = self.embedding(x)
            
            # LSTM
            if lengths is not None:
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
            
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            if lengths is not None:
                lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            
            # Attention
            attended, attn_weights = self.attention(lstm_out)
            
            # Classify
            logits = self.classifier(attended)
            
            return logits
    
    
    class HarmonyMLP(nn.Module):
        """
        MLP for chord prediction from melodic context.
        
        Fast inference for real-time suggestions.
        """
        
        def __init__(
            self,
            input_dim: int = 128,
            hidden_dims: List[int] = [256, 128, 64],
            num_chords: int = 48,
            dropout: float = 0.2,
            use_residual: bool = True,
        ):
            super().__init__()
            
            self.use_residual = use_residual
            
            layers = []
            in_dim = input_dim
            
            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            
            self.backbone = nn.Sequential(*layers)
            self.classifier = nn.Linear(hidden_dims[-1], num_chords)
            
            # Residual projection if dimensions don't match
            if use_residual and input_dim != hidden_dims[-1]:
                self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])
            else:
                self.residual_proj = None
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Melodic context features (batch, input_dim)
            
            Returns:
                Chord logits (batch, num_chords)
            """
            out = self.backbone(x)
            
            if self.use_residual:
                if self.residual_proj is not None:
                    residual = self.residual_proj(x)
                else:
                    residual = x
                out = out + residual
            
            return self.classifier(out)
    
    
    # =========================================================================
    # Multi-Task Architecture
    # =========================================================================
    
    class MultiTaskModel(nn.Module):
        """
        Multi-task learning model with shared backbone.
        
        Supports:
        - Shared feature extraction
        - Task-specific heads
        - Gradient balancing
        
        Args:
            backbone: Shared feature extractor
            task_heads: Dict of task_name -> head module
            feature_dim: Dimension of shared features
        """
        
        def __init__(
            self,
            tasks: List[str],
            input_channels: int = 1,
            feature_dim: int = 256,
            task_configs: Optional[Dict[str, Dict]] = None,
        ):
            super().__init__()
            
            self.tasks = tasks
            self.feature_dim = feature_dim
            
            # Shared CNN backbone
            self.backbone = nn.Sequential(
                ConvBlock(input_channels, 32, 3, padding=1),
                nn.MaxPool2d(2),
                ConvBlock(32, 64, 3, padding=1),
                nn.MaxPool2d(2),
                ConvBlock(64, 128, 3, padding=1),
                ResidualBlock(128),
                nn.MaxPool2d(2),
                ConvBlock(128, 256, 3, padding=1),
                ResidualBlock(256),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            
            # Task-specific heads
            self.task_heads = nn.ModuleDict()
            
            default_configs = {
                "emotion": {"num_classes": 7},
                "genre": {"num_classes": 10},
                "valence": {"output_dim": 1},
                "arousal": {"output_dim": 1},
                "tempo": {"output_dim": 1},
            }
            
            if task_configs:
                default_configs.update(task_configs)
            
            for task in tasks:
                config = default_configs.get(task, {"num_classes": 10})
                
                if "num_classes" in config:
                    # Classification head
                    self.task_heads[task] = nn.Sequential(
                        nn.Linear(feature_dim, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, config["num_classes"]),
                    )
                else:
                    # Regression head
                    self.task_heads[task] = nn.Sequential(
                        nn.Linear(feature_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, config.get("output_dim", 1)),
                    )
        
        def forward(
            self,
            x: torch.Tensor,
            tasks: Optional[List[str]] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                x: Input spectrogram (batch, 1, n_mels, time)
                tasks: Optional list of tasks to compute (default: all)
            
            Returns:
                Dict of task_name -> predictions
            """
            if tasks is None:
                tasks = self.tasks
            
            # Shared features
            features = self.backbone(x)
            
            # Task-specific outputs
            outputs = {}
            for task in tasks:
                if task in self.task_heads:
                    outputs[task] = self.task_heads[task](features)
            
            return outputs
        
        def get_shared_features(self, x: torch.Tensor) -> torch.Tensor:
            """Get shared backbone features."""
            return self.backbone(x)


    class MusicFoundationModel(nn.Module):
        """
        Self-Supervised Foundation Model for Music.
        Uses an encoder-projector setup for Contrastive Learning.
        Designed to be pre-trained on the full 3 TB unlabeled dataset.
        """
        def __init__(self, backbone: nn.Module, embedding_dim: int = 512, projection_dim: int = 128):
            super().__init__()
            self.encoder = backbone
            self.projector = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, projection_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Returns embeddings for SSL loss
            features = self.encoder(x)
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            return self.projector(features)

        def get_features(self, x: torch.Tensor) -> torch.Tensor:
            # Returns raw features for fine-tuning downstream tasks
            return self.encoder(x)


else:
    # Placeholder classes when PyTorch is not available
    class ConvBlock:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    class ResidualBlock:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    class AttentionBlock:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    class MultiHeadAttention:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    class EmotionCNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    class MelodyLSTM:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    class HarmonyMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
    
    class MultiTaskModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")

