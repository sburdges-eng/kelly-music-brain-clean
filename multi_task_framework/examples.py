"""
Comprehensive examples demonstrating multi-task framework usage

Shows:
1. Basic multi-task model creation
2. Task independence and selective training
3. Loss balancing strategies
4. Modular extension with custom components
5. Backwards compatibility patterns
6. Progressive migration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import TaskConfig
from .factory import MultiTaskModelFactory
from .encoders import MultiModalEncoder
from .heads import ClassificationHead, CustomHead
from .trainer import MultiTaskTrainer


# ============================================================================
# Example 1: Basic Multi-Task Model
# ============================================================================

def example_basic_multi_task():
    """Basic multi-task model with predefined tasks"""
    print("\n" + "="*60)
    print("Example 1: Basic Multi-Task Model")
    print("="*60)

    # Define tasks
    task_configs = [
        TaskConfig(
            name="sentiment",
            task_type="classification",
            input_dim=768,
            output_dim=3,
            weight=1.0
        ),
        TaskConfig(
            name="emotion",
            task_type="classification",
            input_dim=768,
            output_dim=6,
            weight=0.8
        ),
        TaskConfig(
            name="toxicity",
            task_type="classification",
            input_dim=768,
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

    print(f"Model created with {len(task_configs)} tasks")
    print(f"Active tasks: {model.get_active_tasks()}")

    # Dummy forward pass
    batch_input = {
        "text": torch.randn(8, 512),
        "audio": torch.randn(8, 128)
    }
    outputs = model(batch_input)
    print(f"Output keys: {list(outputs.keys())}")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")


# ============================================================================
# Example 2: Task Independence and Selective Training
# ============================================================================

def example_task_independence():
    """Demonstrate task independence - enable/disable tasks"""
    print("\n" + "="*60)
    print("Example 2: Task Independence & Selective Training")
    print("="*60)

    task_configs = [
        TaskConfig(
            name="primary",
            task_type="classification",
            input_dim=768,
            output_dim=10,
            weight=1.0
        ),
        TaskConfig(
            name="auxiliary",
            task_type="regression",
            input_dim=768,
            output_dim=1,
            weight=0.5
        ),
    ]

    model = MultiTaskModelFactory.build_model(
        task_configs=task_configs,
        encoder_type="multimodal",
        encoder_kwargs={
            "modality_dims": {"text": 256},
            "output_dim": 128
        }
    )

    print(f"Initial active tasks: {model.get_active_tasks()}")

    # Disable auxiliary task
    model.disable_task("auxiliary")
    print(f"After disabling 'auxiliary': {model.get_active_tasks()}")

    # Re-enable auxiliary task
    model.enable_task("auxiliary")
    print(f"After re-enabling 'auxiliary': {model.get_active_tasks()}")

    # Forward pass only computes enabled tasks
    batch_input = {"text": torch.randn(4, 256)}

    model.disable_task("auxiliary")
    outputs = model(batch_input)
    print(f"Outputs with only primary task: {list(outputs.keys())}")

    model.enable_task("auxiliary")
    outputs = model(batch_input)
    print(f"Outputs with both tasks: {list(outputs.keys())}")


# ============================================================================
# Example 3: Loss Balancing Strategies
# ============================================================================

def example_loss_balancing():
    """Demonstrate static and dynamic loss balancing"""
    print("\n" + "="*60)
    print("Example 3: Loss Balancing Strategies")
    print("="*60)

    task_configs = [
        TaskConfig(
            name="task1",
            task_type="classification",
            input_dim=768,
            output_dim=5,
            weight=1.0
        ),
        TaskConfig(
            name="task2",
            task_type="classification",
            input_dim=768,
            output_dim=3,
            weight=2.0
        ),
    ]

    model = MultiTaskModelFactory.build_model(
        task_configs=task_configs,
        encoder_type="multimodal",
        encoder_kwargs={
            "modality_dims": {"input": 256},
            "output_dim": 128
        }
    )

    # Create dummy data
    batch_input = {"input": torch.randn(4, 256)}
    targets = {
        "task1": torch.randint(0, 5, (4,)),
        "task2": torch.randint(0, 3, (4,))
    }

    outputs = model(batch_input)
    losses = model.compute_losses(outputs, targets)

    print("\nStatic Loss Balancing:")
    total_loss, weighted = model.loss_balancer(losses, method="static")
    for task, loss in weighted.items():
        print(f"  {task}: {loss:.4f}")
    print(f"  Total: {total_loss:.4f}")

    print("\nDynamic Loss Balancing (with uncertainty):")
    total_loss, weighted = model.loss_balancer(losses, method="dynamic")
    for task, loss in weighted.items():
        print(f"  {task}: {loss:.4f}")
    print(f"  Total: {total_loss:.4f}")

    # Update static weights
    print("\nUpdating weights: task1=1.5, task2=0.5")
    model.loss_balancer.update_weight("task1", 1.5)
    model.loss_balancer.update_weight("task2", 0.5)
    total_loss, weighted = model.loss_balancer(losses, method="static")
    for task, loss in weighted.items():
        print(f"  {task}: {loss:.4f}")


# ============================================================================
# Example 4: Modular Extension with Custom Head
# ============================================================================

def example_custom_extension():
    """Demonstrate how to add custom task head"""
    print("\n" + "="*60)
    print("Example 4: Modular Extension with Custom Head")
    print("="*60)

    # Define custom head
    class CustomMetricLearningHead(nn.Module):
        """Custom head for metric learning"""
        def __init__(self, input_dim, output_dim=128):
            super().__init__()
            self.projector = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )

        def forward(self, x):
            return torch.nn.functional.normalize(self.projector(x), dim=-1)

    # Register custom head type
    class CustomMetricHead:
        """Wrapper to make custom head compatible with framework"""
        def __init__(self, input_dim, task_config):
            self.head = CustomMetricLearningHead(input_dim)
            self.config = task_config

        def forward(self, x):
            return self.head(x)

        def compute_loss(self, predictions, targets):
            # Simple cosine similarity loss
            return 1.0 - torch.nn.functional.cosine_similarity(
                predictions, targets, dim=-1
            ).mean()

    # Register the custom head
    MultiTaskModelFactory.register_head("custom_metric", CustomMetricHead)

    print("Custom head type registered: 'custom_metric'")

    # Now use it in a task config
    task_configs = [
        TaskConfig(
            name="metric",
            task_type="custom_metric",
            input_dim=768,
            output_dim=128
        )
    ]

    print("Custom task created and added to model")


# ============================================================================
# Example 5: Configuration-Based Model Building
# ============================================================================

def example_config_based():
    """Build model from configuration dictionary"""
    print("\n" + "="*60)
    print("Example 5: Configuration-Based Model Building")
    print("="*60)

    config = {
        "tasks": [
            {
                "name": "sentiment",
                "task_type": "classification",
                "output_dim": 3,
                "weight": 1.0
            },
            {
                "name": "intent",
                "task_type": "classification",
                "output_dim": 10,
                "weight": 0.8
            }
        ],
        "encoder": {
            "type": "multimodal",
            "modality_dims": {"text": 512, "audio": 64},
            "output_dim": 256,
            "fusion_method": "concat"
        }
    }

    model = MultiTaskModelFactory.build_from_config(config)
    print(f"Model built from config with tasks: {model.get_active_tasks()}")

    # Test forward pass
    batch_input = {
        "text": torch.randn(2, 512),
        "audio": torch.randn(2, 64)
    }
    outputs = model(batch_input)
    print(f"Output shapes: {[(k, v.shape) for k, v in outputs.items()]}")


# ============================================================================
# Example 6: Full Training Loop
# ============================================================================

def example_full_training():
    """Complete training example with evaluation"""
    print("\n" + "="*60)
    print("Example 6: Full Training Loop")
    print("="*60)

    # Setup
    task_configs = [
        TaskConfig(
            name="classification",
            task_type="classification",
            input_dim=768,
            output_dim=3,
            weight=1.0
        )
    ]

    model = MultiTaskModelFactory.build_model(
        task_configs=task_configs,
        encoder_type="multimodal",
        encoder_kwargs={
            "modality_dims": {"input": 64},
            "output_dim": 128
        }
    )

    # Create dummy data
    X = torch.randn(100, 64)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=8)

    # Optimizer and trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = MultiTaskTrainer(
        model=model,
        optimizer=optimizer,
        loss_balancer=model.loss_balancer,
        device="cpu"
    )

    # Training
    print("Training for 3 epochs...")
    for epoch in range(3):
        total_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            inputs = {"input": X_batch}
            targets = {"classification": y_batch}

            # Manual training step for this example
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = model.compute_losses(outputs, targets)
            loss, _ = model.loss_balancer(losses)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")


# ============================================================================
# Run all examples
# ============================================================================

if __name__ == "__main__":
    example_basic_multi_task()
    example_task_independence()
    example_loss_balancing()
    example_custom_extension()
    example_config_based()
    example_full_training()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
