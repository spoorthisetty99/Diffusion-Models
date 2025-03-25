import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import conv3x3, BasicBlock  # import basic building blocks from resnet.py

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        """
        Create a sinusoidal time embedding and then process it through an MLP.
        T: total diffusion steps.
        d_model: embedding dimension before MLP (should be even).
        dim: output dimension (e.g. ch*4).
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        # Create sinusoidal embeddings similar to positional encoding
        emb = torch.arange(0, d_model, step=2).float() / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]  # (T, d_model/2)
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)  # (T, d_model/2, 2)
        emb = emb.view(T, d_model)  # (T, d_model)
        
        self.embedding = nn.Embedding.from_pretrained(emb, freeze=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        # t should be a LongTensor of shape (B,)
        emb = self.embedding(t)
        emb = self.mlp(emb)
        return emb

class ResNetDiffusion(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        """
        Parameters (as passed from main.py):
          T: total diffusion steps.
          ch: base channel count (used as the starting number of channels).
          ch_mult, attn, num_res_blocks: parameters from the original UNet (here not directly used)
                                      since we fix an encoder similar to ResNet18.
          dropout: dropout rate.
        """
        super().__init__()
        self.T = T
        base_channels = ch  # use ch as the starting number of channels
        time_emb_dim = base_channels * 4  # as in the original UNet
        
        # Time embedding network
        self.time_embedding = TimeEmbedding(T, d_model=base_channels, dim=time_emb_dim)
        
        # Build a ResNet-like encoder.
        # For a 32x32 input we follow a ResNet-18 style with layers = [2,2,2,2].
        self.inplanes = base_channels
        self.conv1 = conv3x3(3, base_channels)   # output: (B, ch, 32, 32)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, base_channels, 2, stride=1)      # (B, ch, 32,32)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, 2, stride=2)  # (B, ch*2, 16,16)
        self.layer3 = self._make_layer(BasicBlock, base_channels * 4, 2, stride=2)  # (B, ch*4, 8,8)
        self.layer4 = self._make_layer(BasicBlock, base_channels * 8, 2, stride=2)  # (B, ch*8, 4,4)
        
        # Project the time embedding to match the channel dimension of the encoder’s output.
        self.time_proj = nn.Linear(time_emb_dim, base_channels * 8)
        
        # Decoder: Upsample from (B, ch*8, 4,4) back to (B, 3, 32,32).
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),      # 16x16 -> 32x32
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1)  # map to 3 channels (RGB)
        )
        self.dropout = nn.Dropout(dropout)
        
    def _make_layer(self, block, planes, blocks, stride):
        """Mimic the ResNet _make_layer function."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
        
    def forward(self, x, t):
        """
        x: input image tensor of shape (B, 3, 32, 32)
        t: diffusion timestep tensor of shape (B,)
        """
        # Compute time embedding.
        temb = self.time_embedding(t)  # (B, time_emb_dim)
        
        # Encoder forward pass.
        x = self.conv1(x)     # (B, ch, 32,32)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)    # (B, ch, 32,32)
        x = self.layer2(x)    # (B, ch*2, 16,16)
        x = self.layer3(x)    # (B, ch*4, 8,8)
        x = self.layer4(x)    # (B, ch*8, 4,4)
        
        # Condition on the time embedding.
        temb_proj = self.time_proj(temb)  # (B, ch*8)
        temb_proj = temb_proj.unsqueeze(-1).unsqueeze(-1)  # (B, ch*8, 1,1)
        x = x + temb_proj
        x = self.dropout(x)
        
        # Decoder to predict the noise (or denoised image) with the same resolution as the input.
        x = self.decoder(x)   # (B, 3, 32,32)
        return x

if __name__ == '__main__':
    # Example usage with dummy data.
    batch_size = 8
    # The parameters below come from main.py:
    # T (total diffusion steps), ch (base channel count), ch_mult, attn, num_res_blocks, dropout.
    model = ResNetDiffusion(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    y = model(x, t)
    print("Output shape:", y.shape)  # Expected: (8, 3, 32, 32)
