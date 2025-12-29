# Multi-Task Learning Framework

A comprehensive, production-ready framework for multi-task learning with encoder generalization and modular extensibility.

## Key Features

### 1. **Encoder Generalization: Multi-modal → Single Representation**
- Unified `SharedEncoder` base class for all modality fusion approaches
- Pre-built encoders: `MultiModalEncoder`, `HierarchicalEncoder`
- Support for multiple fusion strategies:
  - Concatenation
  - Attention-based fusion
  - Gating-based fusion
- Easy to extend with custom encoders

### 2. **Head Independence: Task Evaluation/Update Separately**
- Each task has its own independent `TaskHead`
- Tasks can be independently:
  - Enabled/disabled at runtime
  - Updated with separate learning rates
  - Evaluated in isolation
- No coupling between task heads

### 3. **Loss Balancing: Weighted Multi-Task Learning**
- `LossBalancer` for intelligent loss combination
- Two strategies:
  - **Static**: Fixed weights per task
  - **Dynamic**: Learnable uncertainty weights
- Prevents one task from dominating training
- Configurable per-task weights

### 4. **Extension Modularity: Optional Components Without Core Dependency**
- Register custom heads and encoders without modifying core
- Task types are registered dynamically
- Plugin architecture via `MultiTaskModelFactory`
- Optional components don't break existing code

### 5. **Backwards Compatibility: Gradual Migration Path**
- `BackwardsCompatibilityWrapper` for existing code
- Seamless transition from single-task to multi-task
- Can gradually add new tasks to existing models
- Primary task concept for single-output scenarios

### 6. **Extensibility: Easy to Add New Tasks/Modalities**
- Configuration-based model building
- Pre-built task heads for common types:
  - Classification
  - Regression
  - Sequence labeling
  - Multi-label classification
  - Contrastive learning
  - Ranking/metric learning
- Custom head implementation straightforward

## Architecture

```
MultiTaskModel
├── SharedEncoder (Multi-modal → Unified Representation)
│   ├── Modality Projections
│   ├── Fusion Module
│   └── Output Projection
├── TaskHeads (Independent, Task-Specific)
│   ├── Head 1 (Classification)
│   ├── Head 2 (Regression)
│   └── Head N (Custom)
└── LossBalancer (Weighted Combination)
    ├── Static Weighting
    └── Dynamic Weighting
```

## Quick Start

### Basic Usage

```python
from multi_task_framework import (
    TaskConfig,
    MultiTaskModelFactory,
    MultiTaskTrainer
)

# Define tasks
task_configs = [
    TaskConfig(
        name="sentiment",
        task_type="classification",
        output_dim=3,
        weight=1.0
    ),
    TaskConfig(
        name="toxicity",
        task_type="classification",
        output_dim=2,
        weight=1.2
    ),
]

# Build model
model = MultiTaskModelFactory.build_model(
    task_configs=task_configs,
    encoder_type="multimodal",
    encoder_kwargs={
        "modality_dims": {"text": 512, "audio": 128},
        "output_dim": 256,
        "fusion_method": "attention"
    }
)

# Train
optimizer = torch.optim.Adam(model.parameters())
trainer = MultiTaskTrainer(model, optimizer, model.loss_balancer)

# Forward pass
outputs = model({
    "text": text_tensor,
    "audio": audio_tensor
})
```

### Configuration-Based Building

```python
config = {
    "tasks": [
        {"name": "sentiment", "task_type": "classification", "output_dim": 3},
        {"name": "emotion", "task_type": "classification", "output_dim": 6},
    ],
    "encoder": {
        "type": "multimodal",
        "modality_dims": {"text": 512, "audio": 128},
        "output_dim": 256
    }
}

model = MultiTaskModelFactory.build_from_config(config)
```

### Task Management

```python
# Enable/disable tasks at runtime
model.disable_task("sentiment")
model.enable_task("sentiment")

# Get active tasks
active = model.get_active_tasks()

# Selective training - only compute losses for enabled tasks
outputs = model(inputs)  # Only forward through enabled heads
losses = model.compute_losses(outputs, targets)
```

### Loss Balancing

```python
# Static weighting
total_loss, weighted_losses = model.loss_balancer(
    losses, method="static"
)

# Dynamic weighting with learnable uncertainty
total_loss, weighted_losses = model.loss_balancer(
    losses, method="dynamic"
)

# Update weights
model.loss_balancer.update_weight("sentiment", 1.5)
```

### Custom Extensions

```python
# Custom encoder
class MyCustomEncoder(SharedEncoder):
    def forward(self, x):
        # Custom logic
        return encoded_representation

MultiTaskModelFactory.register_encoder("custom", MyCustomEncoder)

# Custom head
class MyCustomHead(TaskHead):
    def forward(self, shared_repr):
        # Custom logic
        return predictions

    def compute_loss(self, predictions, targets):
        # Custom loss
        return loss

MultiTaskModelFactory.register_head("custom_task", MyCustomHead)

# Now use in config
model = MultiTaskModelFactory.build_from_config({
    "tasks": [{"name": "x", "task_type": "custom_task", ...}],
    "encoder": {"type": "custom", ...}
})
```

### Backwards Compatibility

```python
from multi_task_framework import BackwardsCompatibilityWrapper

# Wrap multi-task model
model = MultiTaskModelFactory.build_model(...)
wrapper = BackwardsCompatibilityWrapper(model)

# Set primary task for backward-compatible interface
wrapper.set_primary_task("sentiment")

# Old code still works - returns only sentiment output
output = wrapper(input_data)

# New code can access all outputs
all_outputs = wrapper.get_all_outputs(input_data)

# Add new tasks progressively
wrapper.enable_task("toxicity")
```

## Advanced Features

### Encoder Freezing

```python
# Freeze encoder (e.g., after pretraining)
model.encoder.freeze()

# Only update task heads
for param in model.encoder.parameters():
    param.requires_grad = False

# Unfreeze when needed
model.encoder.unfreeze()
```

### Hierarchical Multi-Modal Encoding

```python
encoder = HierarchicalEncoder(
    modality_groups={
        "vision": ["image", "video"],
        "audio": ["speech", "music"],
        "text": ["transcript"]
    },
    modality_dims={"image": 2048, "video": 1024, ...},
    output_dim=512
)
```

### Different Fusion Strategies

```python
# Concatenation-based (preserves all info)
model1 = MultiTaskModelFactory.build_model(
    ...,
    encoder_kwargs={"fusion_method": "concat", ...}
)

# Attention-based (learns importance weighting)
model2 = MultiTaskModelFactory.build_model(
    ...,
    encoder_kwargs={"fusion_method": "attention", ...}
)

# Gating-based (learned interpolation)
model3 = MultiTaskModelFactory.build_model(
    ...,
    encoder_kwargs={"fusion_method": "gating", ...}
)
```

## Framework Components

### Core Classes

- **TaskConfig**: Task definition with name, type, dimensions, weight
- **SharedEncoder**: Base for multi-modal encoders
- **TaskHead**: Base for task-specific heads
- **LossBalancer**: Intelligent loss combination
- **MultiTaskModel**: Main model combining encoder, heads, balancer

### Pre-built Encoders

- **MultiModalEncoder**: General-purpose multi-modal fusion
- **HierarchicalEncoder**: Grouped modality processing
- **MultiHeadAttentionFusion**: Attention-based fusion
- **GatingFusion**: Gating-based fusion

### Pre-built Heads

- **ClassificationHead**: Single-label classification
- **RegressionHead**: Regression tasks
- **SequenceHead**: Sequence labeling with LSTM
- **MultiLabelHead**: Multi-label classification
- **ContrastiveHead**: Contrastive learning (NT-Xent loss)
- **RankingHead**: Metric learning with triplet loss
- **CustomHead**: Wrapper for user-defined architectures

### Utilities

- **MultiTaskModelFactory**: Build models from configs
- **MultiTaskTrainer**: Training loop with logging
- **BackwardsCompatibilityWrapper**: Single-task interface for multi-task models
- **ProgressiveMigrationBuilder**: Migrate from single to multi-task

## File Structure

```
multi_task_framework/
├── __init__.py              # Package exports
├── base.py                  # Core classes and interfaces
├── encoders.py              # Encoder implementations
├── heads.py                 # Task head implementations
├── factory.py               # Factory pattern & registry
├── trainer.py               # Training utilities
├── examples.py              # Comprehensive examples
└── README.md                # This file
```

## Examples

See `examples.py` for comprehensive examples including:

1. Basic multi-task model creation
2. Task independence and selective training
3. Static and dynamic loss balancing
4. Custom head implementation and registration
5. Configuration-based model building
6. Full training loop

Run examples:
```bash
python -m multi_task_framework.examples
```

## Best Practices

### 1. Weight Balancing
- Start with equal weights (1.0) for all tasks
- Adjust based on task importance and loss scales
- Use dynamic weighting for fine-tuning

### 2. Modality Fusion
- Use attention fusion for interpretability
- Use concatenation for maximum information preservation
- Use gating fusion for learned, adaptive fusion

### 3. Task Independence
- Keep task-specific heads simple
- Use shared encoder for generalization
- Freeze encoder after pretraining individual tasks

### 4. Extensibility
- Register custom components via factory methods
- Keep custom heads compatible with TaskHead interface
- Document custom implementations well

### 5. Training
- Start with all tasks enabled
- Monitor per-task loss trends
- Adjust weights if one task dominates
- Use early stopping on validation loss

## Performance Considerations

- **Shared Encoder**: Reduces parameter count vs. separate models
- **Task Heads**: Minimal overhead (single linear layer + activation)
- **Batch Size**: Can be same as single-task model
- **Memory**: Scales with number of active tasks during backward pass
- **Speed**: Forward pass slower due to multiple heads (but still efficient)

## Debugging

```python
# Monitor active tasks
print(model.get_active_tasks())

# Check frozen status
print(model.encoder.frozen)

# Inspect loss weights
for task in task_configs:
    weight = model.loss_balancer.task_configs[task.name].weight
    print(f"{task.name}: {weight}")

# Verify gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.mean().item():.6f}")
```

## Contributing

To add new components:

1. Inherit from base class (SharedEncoder, TaskHead)
2. Implement required methods
3. Register via factory if extending framework
4. Add tests and documentation

## License

MIT License
