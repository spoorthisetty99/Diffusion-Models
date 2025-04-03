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
        # Replace first conv to support different number of input channels (e.g., 3 for RGB)
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Identity()  # Remove final FC layer
        
        # Time embedding network: expects a one-hot vector of dimension T
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
        
        # Extract ResNet features (shape: [B, 512])
        features = self.resnet(x)
        
        # Convert t (shape: [B]) to one-hot encoding (shape: [B, T])
        t_onehot = F.one_hot(t, num_classes=self.T).float()
        
        # Compute time embedding and reshape to [B, 512, 1, 1]
        t_emb = self.time_embedding(t_onehot).view(batch_size, -1, 1, 1)
        
        # Add time embedding to ResNet features (after reshaping features to [B, 512, 1, 1])
        features = features.view(batch_size, 512, 1, 1) + t_emb
        
        # Predict noise from fused features
        noise = self.output_layer(features)
        return noise

# Example usage:
if __name__ == '__main__':
    batch_size = 8
    img_size = 32
    T = 1000  # Total diffusion timesteps
    model = ResNetDiffusion(T=T, num_channels=3)
    
    # Create dummy image batch and timestep indices
    x = torch.randn(batch_size, 3, img_size, img_size)
    t = torch.randint(0, T, (batch_size,))
    
    # Forward pass: returns the predicted noise
    predicted_noise = model(x, t)
    print("Predicted noise shape:", predicted_noise.shape)  # Expected: [8, 3, 1, 1]
