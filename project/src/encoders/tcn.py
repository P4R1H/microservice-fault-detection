"""
TCN (Temporal Convolutional Network) encoder for time series.
"""

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Single TCN block with dilated causal convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        # Causal trimming
        self.trim = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, channels, seq_len)"""
        residual = x
        
        out = self.conv1(x)
        out = out[:, :, :-self.trim] if self.trim > 0 else out
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.trim] if self.trim > 0 else out
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # Match lengths
        if residual.size(2) > out.size(2):
            residual = residual[:, :, -out.size(2):]
        
        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder.
    
    Features:
    - Dilated causal convolutions for long-range dependencies
    - Parallelizable (unlike LSTM/GRU)
    - Residual connections for gradient flow
    """
    
    def __init__(self, 
                 in_channels: int = 64,
                 hidden_channels: int = 128,
                 out_channels: int = 256,
                 num_layers: int = 4,
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        dilations = [2**i for i in range(num_layers)]
        
        for i, dilation in enumerate(dilations):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            (batch, out_channels)
        """
        # (batch, features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        return x
