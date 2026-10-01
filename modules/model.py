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

class ResidualBlock(nn.Module):
    """Two 3x3 convs with a skip connection: out = ReLU(F(x) + shortcut(x))."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 1x1 conv on the skip path only when channels change, so shapes match.
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))
        
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, output_layer=None):
        """
        output_layer: module applied to the last decoder features
            (shape B x base_channels x H x W).
            - None (default): a 1x1 conv mapping base_channels -> out_channels.
            - nn.Identity(): return the raw decoder features (out_channels is ignored).
            - any custom nn.Module that accepts base_channels input channels.
        """
        super().__init__()
        c = base_channels

        # ---- Encoders ----
        self.enc1 = ResidualBlock(in_channels, c)                                # H   x W
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(c, 2 * c))      # H/2 x W/2

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(2 * c, 4 * c))  # H/4 x W/4

        # ---- Decoders ----
        # Bilinear upsampling has no parameters and keeps the channel count, so
        # after concatenating with the skip connection the input to each
        # decoder block has (previous channels + encoder channels).
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ResidualBlock(4 * c + 2 * c, 2 * c)   # upsampled bottleneck + enc2
        self.dec1 = ResidualBlock(2 * c + c, c)           # upsampled dec2 + enc1

        # ---- Optional output layer ----
        if output_layer is None:
            output_layer = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.output_layer = output_layer

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)           # (B, c,  H,   W)
        e2 = self.enc2(e1)          # (B, 2c, H/2, W/2)
        b = self.bottleneck(e2)     # (B, 4c, H/4, W/4)

        # Decoder path with skip connections
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1))    # (B, 2c, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))   # (B, c,  H,   W)

        return self.output_layer(d1)

class SuperResolutionUNetWithPreUpsampling(nn.Module):
    """Nearest-neighbour upsampling, then a UNet that predicts the missing detail.

        x_up   = nearest_upsample(x)
        output = ReLU(x_up + UNet(x_up))
    """

    def __init__(self, channels=1, base_channels=32, scale_factor=(4, 4)):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.unet = UNet(in_channels=channels, out_channels=channels, base_channels=base_channels)
        self.activation = nn.Softplus()  # keeps the output >= 0

    def forward(self, x):
        x_up = self.upsample(x)
        return self.activation(x_up + self.unet(x_up))

class SuperResolutionUNetOLD(UNet):
    """UNet at low resolution; the output layer does the upsampling.

        x (low-res) -> UNet features -> 1x1 conv -> bilinear upsample -> ReLU
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, scale_factor=(4, 4)):
        output_layer = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
            nn.Softplus(),  # keeps the output >= 0
        )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            output_layer=output_layer,
        )

class SuperResolutionUNet(UNet):
    """UNet at low resolution; the output layer learns the upsampling."""

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, scale_factor=(4, 4)):
        output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels, out_channels,
                kernel_size=scale_factor, stride=scale_factor,
            ),
            nn.Softplus(),  # keeps the output >= 0
        )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            output_layer=output_layer,
        )