import torch
import torch.nn as nn
from typing import List
from src.models.base import BaseModel

class PixelMLP(BaseModel):
    """
    Multilayer Perceptron designed for raw pixel input.
    Demonstrates the inefficiency of FC layers for spatial data.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_layers: List[int], 
        num_classes: int,
        dropout_rate: float
    ):
        super().__init__()
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = h_dim
            
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flattening: (Batch, Channel, Height, Width) -> (Batch, C*H*W)
        x = x.view(x.size(0), -1)
        return self.network(x)