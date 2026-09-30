import torch
import torch.nn as nn

##############
### Models ###
##############

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()

        layers = []
        input_dim = in_dim

        for hidden_dim in hidden:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FourierEmbedding(nn.Module):
    def __init__(self, dim=64, scale=10.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        n_freqs = dim // 2
        self.register_buffer("freqs", torch.randn(n_freqs) * scale)

    def forward(self, x):
        # x: (N, 1) -> (N, dim)
        proj = x * self.freqs[None, :]   # (N, dim // 2)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
        
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, output_layer=None):
        super().__init__()

        self.encoder1 = DoubleConv(in_channels, 32)
        self.encoder2 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        
        self.decoder2 = DoubleConv(128 + 64, 64)
        self.decoder1 = DoubleConv(64 + 32, 32)

        if output_layer is None:
            output_layer = nn.Conv2d(32, out_channels, kernel_size=1)

        self.output = output_layer

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool(x1))

        x = self.bottleneck(self.pool(x2))

        x = self.up(x)
        x = self.decoder2(torch.cat([x, x2], dim=1))

        x = self.up(x)
        x = self.decoder1(torch.cat([x, x1], dim=1))

        return self.output(x)
        
class SuperResolutionUNet(UNet):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 scale_factor=(4, 4)):

        output_layer = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Softplus(),
        )

        super().__init__(
            in_channels,
            out_channels,
            output_layer=output_layer,
        )