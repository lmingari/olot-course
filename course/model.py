import torch
import torch.nn as nn

##############
### Models ###
##############

class MultiLayerPerceptron(nn.Module):
    """Plain MLP: (B, in_dim) -> (B, out_dim)."""
    
    def __init__(self, in_dim, hidden, out_dim, activation=nn.ReLU):
        super().__init__()

        layers = []
        input_dim = in_dim
        for hidden_dim in hidden:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FourierEmbedding(nn.Module):
    """Random Fourier features: (B, 1) -> (B, dim).

    Maps a scalar to [sin(f_i * x), cos(f_i * x)] with fixed random frequencies
    f_i. This lets the network react to small changes of t.
    """
    
    def __init__(self, dim=64, scale=10.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x):
        proj = x * self.freqs[None, :]                         # (B, dim/2)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)     # (B, dim)

class TimeEmbedding(nn.Module):
    """Time t: (B,) -> embedding: (B, t_dim)."""

    def __init__(self, t_dim=128, fourier_dim=64):
        super().__init__()
        self.fourier = FourierEmbedding(fourier_dim)
        self.mlp = MultiLayerPerceptron(fourier_dim, [t_dim], t_dim, activation=nn.SiLU)
        self.act = nn.SiLU()

    def forward(self, t):
        t = t[:, None]                    # (B,)    -> (B, 1)
        h = self.fourier(t)               # (B, 1)  -> (B, fourier_dim)
        return self.act(self.mlp(h))      # (B, fourier_dim) -> (B, t_dim)

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
    """Two 3x3 convs with BatchNorm and a skip connection (no time input).

        h   = ReLU(BN(conv1(x)))
        h   = BN(conv2(h))
        out = ReLU(h + shortcut(x))

    Shapes:
        x:   (B, in_channels,  H, W)
        out: (B, out_channels, H, W)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        # 1x1 conv on the skip path only when the number of channels changes.
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))        # (B, out, H, W)
        h = self.bn2(self.conv2(h))                  # (B, out, H, W)
        return self.act(h + self.shortcut(x))        # (B, out, H, W)

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

class TimeResidualBlock(nn.Module):
    """Two 3x3 convs with a skip connection, conditioned on time.

        h   = SiLU(conv1(x))
        h   = h + time_bias(t_emb)        <- the only place where t enters
        h   = SiLU(conv2(h))
        out = h + shortcut(x)

    Shapes:
        x:     (B, in_channels,  H, W)
        t_emb: (B, t_dim)
        out:   (B, out_channels, H, W)
    """

    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        # Linear map: time embedding -> one bias value per output channel.
        self.time_bias = nn.Linear(t_dim, out_channels)

        # 1x1 conv on the skip path only when the number of channels changes.
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.conv1(x))                  # (B, out, H, W)

        bias = self.time_bias(t_emb)                 # (B, out)
        h = h + bias[:, :, None, None]               # (B, out, 1, 1) broadcast over H, W

        h = self.act(self.conv2(h))                  # (B, out, H, W)
        return h + self.shortcut(x)                  # (B, out, H, W)

class FlowUNet(nn.Module):
    """Time-conditioned UNet that predicts the velocity field v(x_t, t).

    Inputs:
        x: (B, C, H, W)   current sample x_t
        t: (B,)           time in [0, 1]
    Output:
        v: (B, C, H, W)   predicted velocity (no output activation: it can be negative)

    n_levels is the number of downsampling steps (max-pools). H and W must be
    divisible by 2 ** n_levels. The channels double at each level (c = base_channels):

        level 0         (B,  c,         H,            W)           encoder
        level 1         (B,  2c,        H/2,          W/2)         encoder
        ...
        level n_levels  (B,  2^n_levels * c, H/2^n_levels, W/2^n_levels)   bottleneck

    The decoder goes back up, concatenating the encoder output of each level
    (skip connection) after upsampling. n_levels=2 is the original model.
    """

    def __init__(self, channels=1, base_channels=32, n_levels=2, t_dim=128):
        super().__init__()
        self.n_levels = n_levels
        chs = [base_channels * 2**i for i in range(n_levels + 1)]    # channels at each level

        self.time_embed = TimeEmbedding(t_dim)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Encoder: levels 0 .. n_levels-1 (input channels: data, then previous level)
        self.encoders = nn.ModuleList()
        in_ch = channels
        for i in range(n_levels):
            self.encoders.append(ResidualBlock(in_ch, chs[i], t_dim))
            in_ch = chs[i]

        # Bottleneck: level n_levels
        self.bottleneck = ResidualBlock(chs[-2], chs[-1], t_dim)

        # Decoder: from level n_levels-1 back to 0
        # (input channels = upsampled features from the level below + skip connection)
        self.decoders = nn.ModuleList(
            [ResidualBlock(chs[i + 1] + chs[i], chs[i], t_dim) for i in reversed(range(n_levels))]
        )

        # Output: map features back to the data channels
        self.out = nn.Conv2d(chs[0], channels, kernel_size=1)

    def forward(self, x, t):
        assert x.shape[-1] % 2**self.n_levels == 0 and x.shape[-2] % 2**self.n_levels == 0, \
            f"H and W must be divisible by {2**self.n_levels}"

        t_emb = self.time_embed(t)                       # (B, t_dim)

        # Encoder: keep the output of each level for the skip connections
        skips = []
        h = x
        for i, enc in enumerate(self.encoders):
            if i > 0:
                h = self.pool(h)                         # (B, ., H/2^i, W/2^i)
            h = enc(h, t_emb)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck(self.pool(h), t_emb)         # (B, 2^n * c, H/2^n, W/2^n)

        # Decoder: upsample, concatenate the skip of the same level, convolve
        for dec, skip in zip(self.decoders, reversed(skips)):
            h = dec(torch.cat([self.up(h), skip], dim=1), t_emb)

        return self.out(h)                               # (B, C, H, W)
