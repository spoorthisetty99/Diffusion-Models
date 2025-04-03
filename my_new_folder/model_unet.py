import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, emb_dim):
        super().__init__()
        assert d_model % 2 == 0

        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=True),
            nn.Linear(d_model, emb_dim),
            Swish(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t):
        return self.time_embedding(t)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),  # Ensures the channel increase
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)  # Concatenate with skip connection

    def forward(self, x, skip_x):

        
        if x.shape[1] != self.upconv.in_channels:
            raise ValueError(f"Expected {self.upconv.in_channels} channels, got {x.shape[1]} instead!")

        x = self.upconv(x)


        if x.isnan().any() or x.isinf().any():
            raise ValueError("NaN or Inf detected in upconv output!")
        x = torch.cat([x, skip_x], dim=1)  # Concatenate along channels
  

        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()

        base_channels = ch  
        time_emb_dim = base_channels * 4  
        self.time_embedding = TimeEmbedding(T, base_channels, time_emb_dim)

        # Define channels based on multipliers
        channels = [base_channels * m for m in ch_mult]

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, channels[-1]),
            nn.ReLU(),
            nn.Linear(channels[-1], channels[-1])
        )

        self.input_block = ConvBlock(3, channels[0])
        self.downs = nn.ModuleList()
        for i in range(len(ch_mult) - 1):
            self.downs.append(DownSample(channels[i], channels[i + 1]))

        self.bottleneck = ConvBlock(channels[-1], channels[-1])  

        self.ups = nn.ModuleList()
        for i in range(len(ch_mult) - 1, 0, -1):

            self.ups.append(UpSample(channels[i], channels[i - 1]))

        self.output_block = nn.Conv2d(channels[0], 3, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embedding(t))
        t_emb = t_emb[:, :, None, None]  # Reshape for broadcasting

        skips = []
        x = self.input_block(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x) + t_emb  


        # Remove the deepest skip connection since its spatial size doesn't match the upsampled feature map.
        removed_skip = skips.pop()

        # Iterate over the upsampling layers in the original order.
        for up in self.ups:
 
            x = up(x, skips.pop())

        return self.output_block(x)
