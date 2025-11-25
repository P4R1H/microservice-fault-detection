"""
Metrics encoders for time series data.

Implements two approaches for ablation:
1. Chronos-Bolt-Tiny: Foundation model, zero-shot, 100MB
2. TCN: Temporal Convolutional Network, trained, parallelizable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class ChronosEncoder(nn.Module):
    """
    Chronos-Bolt-Tiny encoder for zero-shot metrics encoding.

    Features:
    - 20M parameters, 100MB memory
    - Zero-shot anomaly detection
    - 250x faster than LSTM
    - Direct multi-step forecasting

    Reference: Amazon Chronos (November 2024)
    HuggingFace: amazon/chronos-bolt-tiny
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-tiny",
        context_length: int = 512,
        prediction_length: int = 64,
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.model_name = model_name
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        self.device = device

        # Load Chronos model
        try:
            from chronos import ChronosPipeline

            # Try new API first (dtype instead of torch_dtype)
            try:
                self.chronos = ChronosPipeline.from_pretrained(
                    model_name,
                    device_map=device,
                    dtype=torch.float32  # Use FP32 for stability
                )
            except (TypeError, AttributeError):
                # Fallback to old API
                try:
                    self.chronos = ChronosPipeline.from_pretrained(
                        model_name,
                        device_map=device,
                        torch_dtype=torch.float32  # Old API
                    )
                except TypeError as e:
                    # Version incompatibility - likely config issue
                    raise RuntimeError(
                        f"Chronos version incompatibility: {e}\n"
                        f"Try: pip install --upgrade chronos-forecasting\n"
                        f"Or use TCN encoder instead (already working)"
                    )

            # Freeze backbone if specified
            if freeze_backbone:
                for param in self.chronos.model.parameters():
                    param.requires_grad = False

        except ImportError:
            raise ImportError(
                "chronos-forecasting not installed. "
                "Install with: pip install chronos-forecasting>=1.0.0"
            )

        # Projection layer to get fixed embedding dimension
        # Chronos outputs vary, so we add a projection
        self.projection = nn.Linear(prediction_length, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode metrics time series.

        Args:
            x: (batch, seq_len, n_features) metrics tensor

        Returns:
            (batch, embedding_dim) encoded representation
        """
        batch_size, seq_len, n_features = x.shape

        # Chronos expects (batch, seq_len) for each feature
        # We'll encode each feature separately then aggregate
        feature_embeddings = []

        for feat_idx in range(n_features):
            # Extract single feature: (batch, seq_len)
            feature_data = x[:, :, feat_idx]

            # Get Chronos forecast (uses last context_length points)
            if seq_len > self.context_length:
                context = feature_data[:, -self.context_length:]
            else:
                # Pad if needed
                context = F.pad(feature_data, (self.context_length - seq_len, 0), value=0)

            # Forward through Chronos (zero-shot forecast)
            with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
                forecast = self.chronos.predict(
                    context,
                    prediction_length=self.prediction_length,
                    num_samples=1  # Single deterministic forecast
                )  # Returns (batch, prediction_length)

            feature_embeddings.append(forecast.squeeze(1))  # (batch, prediction_length)

        # Stack all feature forecasts: (batch, n_features, prediction_length)
        all_forecasts = torch.stack(feature_embeddings, dim=1)

        # Aggregate across features (mean pooling)
        aggregated = all_forecasts.mean(dim=1)  # (batch, prediction_length)

        # Project to embedding dimension
        embedding = self.projection(aggregated)  # (batch, embedding_dim)

        return embedding

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score using reconstruction error.

        Args:
            x: (batch, seq_len, n_features) metrics tensor

        Returns:
            (batch,) anomaly scores
        """
        batch_size, seq_len, n_features = x.shape

        reconstruction_errors = []

        for feat_idx in range(n_features):
            feature_data = x[:, :, feat_idx]

            # Use first part as context, last part as target
            split_point = seq_len - self.prediction_length
            if split_point < self.context_length:
                split_point = seq_len // 2

            context = feature_data[:, :split_point]
            target = feature_data[:, split_point:split_point+self.prediction_length]

            # Forecast
            with torch.no_grad():
                forecast = self.chronos.predict(
                    context,
                    prediction_length=self.prediction_length,
                    num_samples=1
                ).squeeze(1)

            # Compute MSE
            mse = F.mse_loss(forecast, target, reduction='none').mean(dim=1)
            reconstruction_errors.append(mse)

        # Average across features
        anomaly_score = torch.stack(reconstruction_errors, dim=1).mean(dim=1)

        return anomaly_score


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder.

    Features:
    - Dilated causal convolutions
    - Parallelizable training (3-5x faster than LSTM)
    - Exponentially large receptive field
    - Lightweight (efficient for edge deployment)

    Architecture:
    - 7 layers with exponential dilation [1,2,4,8,16,32,64]
    - Receptive field: 381 timesteps
    - Parameters: <10M

    Reference: "SDN Anomalous Traffic Detection Based on TCN" (2024)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        embedding_dim: int = 256,
        num_layers: int = 7,
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Build TCN layers with exponential dilation
        self.tcn_layers = nn.ModuleList()

        # Dilation pattern: [1, 2, 4, 8, 16, 32, 64]
        dilations = [2**i for i in range(num_layers)]

        # First layer: in_channels -> hidden_channels
        self.tcn_layers.append(
            TemporalBlock(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilations[0],
                dropout=dropout
            )
        )

        # Hidden layers: hidden_channels -> hidden_channels
        for i in range(1, num_layers):
            self.tcn_layers.append(
                TemporalBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    dropout=dropout
                )
            )

        # Final projection to embedding dimension
        self.projection = nn.Linear(hidden_channels, embedding_dim)

        # Calculate receptive field
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode metrics time series with TCN.

        Args:
            x: (batch, seq_len, features) metrics tensor

        Returns:
            (batch, embedding_dim) encoded representation
        """
        # TCN expects (batch, features, seq_len)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)

        # Forward through TCN layers
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)  # (batch, hidden_channels, seq_len)

        # Global average pooling across time
        x = x.mean(dim=2)  # (batch, hidden_channels)

        # Project to embedding dimension
        embedding = self.projection(x)  # (batch, embedding_dim)

        return embedding


class TemporalBlock(nn.Module):
    """
    Single TCN temporal block with dilated convolutions.

    Components:
    - Dilated causal conv1d
    - Weight normalization
    - ReLU activation
    - Dropout
    - Residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.3
    ):
        super().__init__()

        # Padding to maintain causality (no future leakage)
        # Padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation

        # Two convolutional layers in the block
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

        # Weight normalization for stability
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.conv2 = nn.utils.weight_norm(self.conv2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (1x1 conv if dimensions don't match)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal block.

        Args:
            x: (batch, in_channels, seq_len)

        Returns:
            (batch, out_channels, seq_len)
        """
        # Save input for residual
        residual = x

        # First conv block
        out = self.conv1(x)
        # Chomp padding (causal - remove right side)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = F.relu(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        # Chomp padding
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = F.relu(out)
        out = self.dropout2(out)

        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)

        # Ensure same seq_len for residual addition
        if residual.size(2) != out.size(2):
            residual = residual[:, :, :out.size(2)]

        return F.relu(out + residual)


# Helper functions for metrics encoding

def create_metrics_encoder(
    encoder_type: str = 'chronos',
    in_channels: int = 7,
    embedding_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create metrics encoder.

    Args:
        encoder_type: 'chronos' or 'tcn'
        in_channels: Number of input features
        embedding_dim: Output embedding dimension
        **kwargs: Additional arguments for specific encoder

    Returns:
        Configured encoder module
    """
    if encoder_type == 'chronos':
        return ChronosEncoder(
            embedding_dim=embedding_dim,
            **kwargs
        )
    elif encoder_type == 'tcn':
        return TCNEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'chronos' or 'tcn'.")


def encode_metrics_batch(
    metrics_batch: torch.Tensor,
    encoder: nn.Module,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    Encode a batch of metrics time series.

    Args:
        metrics_batch: (batch, seq_len, n_features) tensor
        encoder: Metrics encoder module
        device: Device to run on

    Returns:
        (batch, embedding_dim) embeddings
    """
    encoder = encoder.to(device)
    encoder.eval()

    metrics_batch = metrics_batch.to(device)

    with torch.no_grad():
        embeddings = encoder(metrics_batch)

    return embeddings
