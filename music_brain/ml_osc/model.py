"""
ML Models for OSC Pattern Learning

Implements LSTM-based models for learning and predicting OSC sequences.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide fallback for when PyTorch is not installed


class OSCPatternLearner(nn.Module if TORCH_AVAILABLE else object):
    """
    LSTM-based model for learning OSC message patterns.
    
    Architecture:
        - Input: (batch, context_length, feature_dim)
        - LSTM layers for temporal modeling
        - Output: (batch, feature_dim) - predicted next message
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        """
        Initialize the model.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for OSCPatternLearner. Install with: pip install torch")
        
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, context_length, feature_dim)
            
        Returns:
            Predicted next message of shape (batch, feature_dim)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        
        # Predict next message
        output = self.fc(last_out)
        return output


class OSCPredictor:
    """
    High-level interface for training and using OSC pattern models.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize predictor.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Number of hidden units
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for optimizer
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.model = OSCPatternLearner(feature_dim, hidden_dim, num_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Perform one training step.
        
        Args:
            X: Context sequences of shape (batch, context_length, feature_dim)
            y: Target messages of shape (batch, feature_dim)
            
        Returns:
            Loss value
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Forward pass
        self.model.train()
        predictions = self.model(X_tensor)
        
        # Compute loss
        loss = self.criterion(predictions, y_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1
    ):
        """
        Train the model.
        
        Args:
            X: Training contexts of shape (num_samples, context_length, feature_dim)
            y: Training targets of shape (num_samples, feature_dim)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data to use for validation
        """
        # Split into train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        for epoch in range(epochs):
            # Training
            train_losses = []
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                loss = self.train_step(batch_X, batch_y)
                train_losses.append(loss)
            
            # Validation
            if len(X_val) > 0:
                val_loss = self.evaluate(X_val, y_val)
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {np.mean(train_losses):.4f}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model on data.
        
        Args:
            X: Context sequences
            y: Target messages
            
        Returns:
            Average loss
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            predictions = self.model(X_tensor)
            loss = self.criterion(predictions, y_tensor)
        return loss.item()
    
    def predict(self, context: np.ndarray) -> np.ndarray:
        """
        Predict next message given context.
        
        Args:
            context: Context of shape (context_length, feature_dim) or (batch, context_length, feature_dim)
            
        Returns:
            Predicted next message of shape (feature_dim,) or (batch, feature_dim)
        """
        self.model.eval()
        with torch.no_grad():
            # Handle single sample
            if context.ndim == 2:
                context = context[np.newaxis, ...]
                squeeze = True
            else:
                squeeze = False
            
            context_tensor = torch.FloatTensor(context).to(self.device)
            prediction = self.model(context_tensor).cpu().numpy()
            
            if squeeze:
                prediction = prediction[0]
        
        return prediction
    
    def save(self, path: Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Model loaded from {path}")
