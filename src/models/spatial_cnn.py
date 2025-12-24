import torch
import torch.nn as nn
from typing import List
from src.models.base import BaseModel

class SpatialCNN(BaseModel):
    """
    Convolutional Neural Network that leverages spatial correlation 
    to extract a latent feature vector before classification.
    """
    def __init__(
        self, 
        channels: List[int], 
        latent_dim: int, 
        num_classes: int
    ):
        super().__init__()
        
        # Feature Extraction Path
        modules = []
        for i in range(len(channels) - 1):
            modules.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
        
        self.encoder = nn.Sequential(*modules)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head (The MLP part that works on vectors)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.gap(x)
        x = torch.flatten(x, 1) # Extract feature vector
        return self.head(x)