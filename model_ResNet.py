import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Time embedding: creates sinusoidal embeddings then passes through an MLP.
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, emb_dim):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        emb = torch.arange(0, d_model, step=2).float() / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]  # shape: (T, d_model/2)
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)  # (T, d_model/2, 2)
        emb = emb.view(T, d_model)  # (T, d_model)
        self.embedding = nn.Embedding.from_pretrained(emb, freeze=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, emb_dim),
            Swish(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t):
        emb = self.embedding(t)
        return self.mlp(emb)

# A simple 3x3 convolutional block used in the ResNet.
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Basic residual block (as in ResNet) for feature extraction.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ResNet encoder backbone. It uses several layers of BasicBlocks to extract features.
class ResNetEncoder(nn.Module):
    def __init__(self, block, layers, in_channels=3, base_channels=64):
        super().__init__()
        self.inplanes = base_channels
        self.conv1 = conv3x3(in_channels, base_channels)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        # Create layers similar to a ResNet-18.
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        skip1 = out  # Skip connection 1.
        out = self.layer2(out)
        skip2 = out  # Skip connection 2.
        out = self.layer3(out)
        skip3 = out  # Skip connection 3.
        out = self.layer4(out)  # Bottleneck features.
        return out, (skip1, skip2, skip3)

# Decoder that upsamples the bottleneck features back to image resolution.
class ResNetDecoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        # Upsample using transposed convolutions.
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4 * 2, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2 * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(base_channels * 2, 3, kernel_size=3, padding=1)

    def forward(self, x, skips):
        skip1, skip2, skip3 = skips
        d1 = self.up1(x)             # Upsample bottleneck to match spatial dims of skip3.
        d1 = torch.cat([d1, skip3], dim=1)
        d2 = self.up2(d1)            # Upsample to match skip2.
        d2 = torch.cat([d2, skip2], dim=1)
        d3 = self.up3(d2)            # Upsample to match skip1.
        d3 = torch.cat([d3, skip1], dim=1)
        out = self.final_conv(d3)
        return out

# The overall denoising network that uses a ResNet encoder backbone.
class ResNetDenoiser(nn.Module):
    def __init__(self, T, base_channels=64):
        super().__init__()
        # Create a time embedding module.
        self.time_embedding = TimeEmbedding(T, d_model=base_channels, emb_dim=base_channels * 4)
        # Use a ResNet encoder (here, similar to ResNet-18 with layers [2, 2, 2, 2]).
        self.encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2], in_channels=3, base_channels=base_channels)
        # A decoder that upsamples the features back to image space.
        self.decoder = ResNetDecoder(base_channels=base_channels)
        # Project the time embedding to the same number of channels as the bottleneck.
        self.time_proj = nn.Linear(base_channels * 4, base_channels * 8)

    def forward(self, x, t):
        # Compute time embedding.
        t_emb = self.time_embedding(t)  # (B, base_channels*4)
        # Encode the input image.
        bottleneck, skips = self.encoder(x)  # bottleneck: (B, base_channels*8, H, W)
        # Condition the bottleneck features on the time embedding.
        t_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, base_channels*8, 1, 1)
        bottleneck = bottleneck + t_proj
        # Decode back to an image.
        out = self.decoder(bottleneck, skips)
        return out

# Example usage:
if __name__ == '__main__':
    batch_size = 8
    img_size = 32
    T = 1000  # Total diffusion timesteps
    model = ResNetDenoiser(T=T, base_channels=64)
    x = torch.randn(batch_size, 3, img_size, img_size)
    t = torch.randint(0, T, (batch_size,))
    y = model(x, t)
    print("Output shape:", y.shape)  # Expected: (8, 3, 32, 32)
