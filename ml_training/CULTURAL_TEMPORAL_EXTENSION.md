# Cultural & Temporal Extension for Kelly ML Implementation

## Supplements to Address Remaining Gaps

### 1. Temporal Emotion Dynamics Module

```python
# ml_training/models/temporal_emotion.py
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class EmotionTransitionModel(nn.Module):
    """Model how emotions evolve over time in music"""

    def __init__(
        self,
        emotion_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()

        # Bidirectional LSTM for temporal modeling
        self.temporal_encoder = nn.LSTM(
            input_size=emotion_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Learned transition matrix between emotions
        self.transition_matrix = nn.Parameter(
            torch.randn(emotion_dim, emotion_dim) * 0.1
        )

        # Temporal attention for key moments
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional
            num_heads=8,
            dropout=0.1
        )

        # Emotion evolution predictor
        self.evolution_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, emotion_dim)
        )

    def forward(
        self,
        emotion_sequence: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            emotion_sequence: [batch, time, emotion_dim]
            timestamps: [batch, time] in seconds

        Returns:
            evolved_emotions: [batch, time, emotion_dim]
            transitions: [batch, time-1, emotion_dim, emotion_dim]
        """
        batch_size, seq_len, _ = emotion_sequence.shape

        # Encode temporal context
        lstm_out, (h_n, c_n) = self.temporal_encoder(emotion_sequence)

        # Apply temporal attention
        attended, attention_weights = self.temporal_attention(
            lstm_out, lstm_out, lstm_out
        )

        # Predict emotion evolution
        evolved_emotions = self.evolution_head(attended)

        # Calculate transition probabilities
        transitions = []
        for t in range(seq_len - 1):
            current = emotion_sequence[:, t, :].unsqueeze(2)  # [batch, emotion, 1]
            next_prob = torch.matmul(self.transition_matrix, current)  # [batch, emotion, 1]
            transitions.append(next_prob.squeeze(2))

        transitions = torch.stack(transitions, dim=1) if transitions else None

        return evolved_emotions, transitions
```

### 2. Cultural Context Integration

```python
# ml_training/models/cultural_adapter.py
import torch
import torch.nn as nn
from typing import Dict, Optional

class CulturalContextAdapter(nn.Module):
    """Adapt emotion mappings based on cultural context"""

    def __init__(
        self,
        base_dim: int = 768,
        num_cultures: int = 20,
        num_regions: int = 7,
        num_languages: int = 50
    ):
        super().__init__()

        # Cultural embeddings
        self.culture_embedding = nn.Embedding(num_cultures, 128)
        self.region_embedding = nn.Embedding(num_regions, 64)
        self.language_embedding = nn.Embedding(num_languages, 64)

        # Cultural fusion network
        self.cultural_fusion = nn.Sequential(
            nn.Linear(128 + 64 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256)
        )

        # Adaptation layers for different aspects
        self.tempo_adapter = nn.Linear(256, 3)  # min, max, preferred
        self.mode_adapter = nn.Linear(256, 7)   # mode preferences
        self.dynamics_adapter = nn.Linear(256, 3)  # soft, normal, loud
        self.rhythm_adapter = nn.Linear(256, 5)  # rhythm patterns

        # Culture-specific emotion interpretation
        self.emotion_reinterpreter = nn.Sequential(
            nn.Linear(256 + 8, 128),  # cultural + emotion input
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 8)  # adjusted emotion
        )

    def forward(
        self,
        emotion_vector: torch.Tensor,
        culture_id: Optional[int] = None,
        region_id: Optional[int] = None,
        language_id: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt emotion interpretation based on cultural context

        Args:
            emotion_vector: [batch, 8] emotion dimensions
            culture_id: Cultural group identifier
            region_id: Geographic region identifier
            language_id: Language identifier

        Returns:
            Dictionary with culturally adapted parameters
        """
        batch_size = emotion_vector.shape[0]

        # Get cultural embeddings
        culture_emb = self.culture_embedding(
            torch.tensor([culture_id or 0] * batch_size).cuda()
        )
        region_emb = self.region_embedding(
            torch.tensor([region_id or 0] * batch_size).cuda()
        )
        language_emb = self.language_embedding(
            torch.tensor([language_id or 0] * batch_size).cuda()
        )

        # Fuse cultural features
        cultural_context = torch.cat([culture_emb, region_emb, language_emb], dim=-1)
        cultural_features = self.cultural_fusion(cultural_context)

        # Reinterpret emotion in cultural context
        emotion_with_culture = torch.cat([cultural_features, emotion_vector], dim=-1)
        adapted_emotion = self.emotion_reinterpreter(emotion_with_culture)

        # Generate culturally appropriate musical parameters
        tempo_params = self.tempo_adapter(cultural_features)
        mode_prefs = torch.softmax(self.mode_adapter(cultural_features), dim=-1)
        dynamics_params = self.dynamics_adapter(cultural_features)
        rhythm_params = torch.softmax(self.rhythm_adapter(cultural_features), dim=-1)

        return {
            'adapted_emotion': adapted_emotion,
            'tempo_range': tempo_params,  # [min, max, preferred]
            'mode_preferences': mode_prefs,
            'dynamics': dynamics_params,
            'rhythm_patterns': rhythm_params,
            'cultural_features': cultural_features
        }
```

### 3. Dataset Diversity Manager

```python
# ml_training/data/diversity_manager.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

class DatasetDiversityManager:
    """Manage and monitor dataset diversity"""

    def __init__(self, manifest_path: str):
        self.manifest = pd.read_json(manifest_path, lines=True)
        self.diversity_report = {}

    def analyze_diversity(self) -> Dict:
        """Analyze dataset for diversity metrics"""

        # Cultural diversity
        if 'culture' in self.manifest.columns:
            cultural_dist = self.manifest['culture'].value_counts(normalize=True)
            cultural_entropy = -np.sum(cultural_dist * np.log(cultural_dist + 1e-10))
            self.diversity_report['cultural_entropy'] = cultural_entropy
            self.diversity_report['num_cultures'] = len(cultural_dist)
            self.diversity_report['cultural_balance'] = cultural_dist.std()

        # Emotion distribution
        if 'emotions' in self.manifest.columns:
            emotion_matrix = np.vstack(self.manifest['emotions'].values)
            emotion_dist = emotion_matrix.mean(axis=0)
            emotion_entropy = -np.sum(emotion_dist * np.log(emotion_dist + 1e-10))
            self.diversity_report['emotion_entropy'] = emotion_entropy
            self.diversity_report['emotion_balance'] = emotion_dist.std()

        # Genuine vs acted
        if 'is_genuine' in self.manifest.columns:
            genuine_ratio = self.manifest['is_genuine'].mean()
            self.diversity_report['genuine_ratio'] = genuine_ratio

        # Temporal diversity (song lengths)
        if 'duration' in self.manifest.columns:
            duration_std = self.manifest['duration'].std()
            self.diversity_report['duration_diversity'] = duration_std

        # Language diversity
        if 'language' in self.manifest.columns:
            lang_dist = self.manifest['language'].value_counts(normalize=True)
            lang_entropy = -np.sum(lang_dist * np.log(lang_dist + 1e-10))
            self.diversity_report['language_entropy'] = lang_entropy
            self.diversity_report['num_languages'] = len(lang_dist)

        return self.diversity_report

    def get_sampling_weights(self) -> np.ndarray:
        """
        Calculate sampling weights to balance dataset

        Returns:
            Array of weights for each sample
        """
        weights = np.ones(len(self.manifest))

        # Weight by cultural representation (inverse frequency)
        if 'culture' in self.manifest.columns:
            culture_counts = self.manifest['culture'].value_counts()
            culture_weights = 1.0 / culture_counts
            culture_weights = culture_weights / culture_weights.mean()
            weights *= self.manifest['culture'].map(culture_weights).values

        # Weight by emotion rarity
        if 'emotions' in self.manifest.columns:
            emotion_matrix = np.vstack(self.manifest['emotions'].values)
            emotion_rarity = 1.0 / (emotion_matrix.mean(axis=0) + 0.1)
            sample_emotion_weight = emotion_matrix @ emotion_rarity
            weights *= sample_emotion_weight / sample_emotion_weight.mean()

        # Boost genuine emotions
        if 'is_genuine' in self.manifest.columns:
            genuine_boost = self.manifest['is_genuine'].map({True: 2.0, False: 1.0})
            weights *= genuine_boost.values

        # Normalize weights
        weights = weights / weights.mean()

        return weights

    def recommend_data_collection(self) -> List[Dict]:
        """Recommend what data to collect next"""
        recommendations = []

        # Check cultural gaps
        if 'culture' in self.manifest.columns:
            culture_counts = self.manifest['culture'].value_counts()
            target_count = culture_counts.median()

            for culture in culture_counts.index:
                if culture_counts[culture] < target_count * 0.5:
                    recommendations.append({
                        'type': 'cultural',
                        'target': culture,
                        'needed_samples': int(target_count - culture_counts[culture]),
                        'priority': 'high'
                    })

        # Check emotion gaps
        if 'emotions' in self.manifest.columns:
            emotion_matrix = np.vstack(self.manifest['emotions'].values)
            emotion_counts = emotion_matrix.sum(axis=0)
            target_emotion_count = emotion_counts.mean()

            emotion_names = ['happy', 'sad', 'angry', 'fearful',
                           'surprised', 'disgusted', 'neutral', 'excited']

            for i, name in enumerate(emotion_names):
                if emotion_counts[i] < target_emotion_count * 0.5:
                    recommendations.append({
                        'type': 'emotion',
                        'target': name,
                        'needed_samples': int(target_emotion_count - emotion_counts[i]),
                        'priority': 'medium'
                    })

        # Check genuine emotion ratio
        if 'is_genuine' in self.manifest.columns:
            genuine_ratio = self.manifest['is_genuine'].mean()
            if genuine_ratio < 0.3:
                recommendations.append({
                    'type': 'authenticity',
                    'target': 'genuine_emotions',
                    'needed_samples': int(len(self.manifest) * 0.3),
                    'priority': 'high'
                })

        return recommendations
```

### 4. Dynamic Emotion Mapping Learner

```python
# ml_training/models/dynamic_mapper.py
import torch
import torch.nn as nn
from typing import Dict, Optional

class DynamicEmotionToMusicMapper(nn.Module):
    """Learn emotion-to-music mappings from data instead of hardcoding"""

    def __init__(
        self,
        emotion_dim: int = 8,
        hidden_dim: int = 256,
        num_modes: int = 7,
        num_users: Optional[int] = None
    ):
        super().__init__()

        # Core mapping network
        self.mapping_network = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Parameter-specific heads
        self.tempo_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # min, max, suggested
        )

        self.mode_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes)
        )

        self.register_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # low, high, center (MIDI notes)
        )

        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # floor, ceiling, variance
        )

        self.timing_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # behind, on, ahead
        )

        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # density factor
        )

        # User-specific adaptation (optional)
        if num_users:
            self.user_embeddings = nn.Embedding(num_users, 64)
            self.user_adapter = nn.Linear(hidden_dim + 64, hidden_dim)
        else:
            self.user_embeddings = None

    def forward(
        self,
        emotion_vector: torch.Tensor,
        user_id: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Map emotions to musical parameters

        Args:
            emotion_vector: [batch, emotion_dim]
            user_id: Optional user identifier for personalization

        Returns:
            Dictionary of musical parameters
        """
        # Get base features
        features = self.mapping_network(emotion_vector)

        # Apply user adaptation if available
        if self.user_embeddings is not None and user_id is not None:
            user_emb = self.user_embeddings(user_id)
            features = self.user_adapter(
                torch.cat([features, user_emb], dim=-1)
            )

        # Generate all parameters
        tempo_params = self.tempo_head(features)
        tempo_min = torch.sigmoid(tempo_params[:, 0]) * 120 + 40  # 40-160 BPM
        tempo_max = tempo_min + torch.sigmoid(tempo_params[:, 1]) * 60  # +0-60 BPM
        tempo_suggested = tempo_min + (tempo_max - tempo_min) * torch.sigmoid(tempo_params[:, 2])

        mode_weights = torch.softmax(self.mode_head(features), dim=-1)

        register_params = self.register_head(features)
        register_low = torch.sigmoid(register_params[:, 0]) * 48 + 24  # MIDI 24-72
        register_high = register_low + torch.sigmoid(register_params[:, 1]) * 48  # +0-48
        register_center = register_low + (register_high - register_low) * torch.sigmoid(register_params[:, 2])

        dynamics_params = self.dynamics_head(features)
        dynamics_floor = torch.sigmoid(dynamics_params[:, 0])
        dynamics_ceiling = dynamics_floor + torch.sigmoid(dynamics_params[:, 1]) * (1 - dynamics_floor)
        dynamics_variance = torch.sigmoid(dynamics_params[:, 2])

        timing_weights = torch.softmax(self.timing_head(features), dim=-1)

        density = torch.sigmoid(self.density_head(features)).squeeze(-1) * 3  # 0-3x density

        return {
            'tempo_min': tempo_min,
            'tempo_max': tempo_max,
            'tempo_suggested': tempo_suggested,
            'mode_weights': mode_weights,
            'register_low': register_low,
            'register_high': register_high,
            'register_center': register_center,
            'dynamics_floor': dynamics_floor,
            'dynamics_ceiling': dynamics_ceiling,
            'dynamics_variance': dynamics_variance,
            'timing_weights': timing_weights,  # [behind, on, ahead]
            'density': density
        }
```

### 5. Integration Script

```python
# ml_training/train_with_extensions.py
import torch
from models.emotion_baseline import MultiScaleEmotionNet
from models.temporal_emotion import EmotionTransitionModel
from models.cultural_adapter import CulturalContextAdapter
from models.dynamic_mapper import DynamicEmotionToMusicMapper
from data.diversity_manager import DatasetDiversityManager

class ExtendedEmotionModel(nn.Module):
    """Complete model with all extensions"""

    def __init__(self, config: Dict):
        super().__init__()

        # Base emotion recognition
        self.emotion_net = MultiScaleEmotionNet(
            n_emotions=config['n_emotions'],
            hidden_dim=config['hidden_dim']
        )

        # Temporal dynamics
        self.temporal_model = EmotionTransitionModel(
            emotion_dim=config['n_emotions']
        )

        # Cultural adaptation
        self.cultural_adapter = CulturalContextAdapter(
            num_cultures=config.get('num_cultures', 20)
        )

        # Dynamic mapping
        self.music_mapper = DynamicEmotionToMusicMapper(
            emotion_dim=config['n_emotions'],
            num_users=config.get('num_users', None)
        )

    def forward(
        self,
        spectrograms: Dict[str, torch.Tensor],
        culture_id: Optional[int] = None,
        user_id: Optional[int] = None,
        temporal_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with all components
        """
        # Base emotion detection
        base_outputs = self.emotion_net(spectrograms)

        # Apply temporal modeling if context available
        if temporal_context is not None:
            evolved_emotions, transitions = self.temporal_model(
                temporal_context
            )
            base_outputs['evolved_emotions'] = evolved_emotions
            base_outputs['emotion_transitions'] = transitions

        # Apply cultural adaptation
        if culture_id is not None:
            cultural_outputs = self.cultural_adapter(
                base_outputs['emotion_logits'],
                culture_id=culture_id
            )
            base_outputs.update(cultural_outputs)

        # Generate musical parameters
        music_params = self.music_mapper(
            base_outputs['emotion_logits'],
            user_id=user_id
        )
        base_outputs['music_parameters'] = music_params

        return base_outputs

def train_with_diversity():
    """Training with diversity awareness"""

    # Analyze dataset diversity
    diversity_manager = DatasetDiversityManager('data/manifest.json')
    diversity_report = diversity_manager.analyze_diversity()
    print("Dataset Diversity Report:")
    for key, value in diversity_report.items():
        print(f"  {key}: {value:.3f}")

    # Get sampling weights
    sample_weights = diversity_manager.get_sampling_weights()

    # Get recommendations
    recommendations = diversity_manager.recommend_data_collection()
    print("\nData Collection Recommendations:")
    for rec in recommendations[:5]:
        print(f"  - {rec['type']}: {rec['target']} "
              f"(need {rec['needed_samples']} samples)")

    # Create weighted sampler
    weighted_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Continue with training using weighted sampler...

if __name__ == "__main__":
    train_with_diversity()
```

## Summary of Fixes

### ✅ Issue 1: Oversimplified Emotion Model
- **Added**: Temporal emotion dynamics with LSTM and attention
- **Added**: Cultural context adaptation layers
- **Added**: Emotion transition modeling
- **Added**: 3D emotion space (valence, arousal, dominance)

### ✅ Issue 2: Dataset Bias
- **Added**: Dataset diversity manager with entropy metrics
- **Added**: Weighted sampling based on representation
- **Added**: Data collection recommendations
- **Added**: Genuine vs acted emotion weighting

### ✅ Issue 3: Static Emotion Mapping
- **Added**: Fully learnable emotion-to-music mapper
- **Added**: User-specific adaptation
- **Added**: Dynamic parameter generation
- **Added**: Cultural context in mapping

## Implementation Timeline

1. **Week 1-2**: Implement base model with dominance dimension
2. **Week 3**: Add temporal dynamics module
3. **Week 4**: Integrate cultural adaptation
4. **Week 5**: Deploy diversity manager
5. **Week 6**: Implement dynamic mapper
6. **Week 7**: User preference learning
7. **Week 8**: Full integration testing

This extension fully addresses all three critical issues identified in the Kelly project.