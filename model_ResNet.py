import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNetDiffusion(nn.Module):
    def __init__(self, T, num_channels=3):
        super().__init__()
        self.T = T
        
        # Base ResNet model (removing final classification layer)
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Identity()  # Remove FC layer
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(T, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        # Final noise prediction head
        self.output_layer = nn.Conv2d(512, num_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, t):
        batch_size = x.shape[0]
        
        # Extract ResNet features
        features = self.resnet(x)
        
        # Expand time embedding to match feature dimensions
        t_emb = self.time_embedding(t.float()).view(batch_size, -1, 1, 1)
        
        # Fuse time embedding with ResNet features
        features = features.view(batch_size, 512, 1, 1) + t_emb
        
        # Predict noise
        noise = self.output_layer(features)
        return noise
