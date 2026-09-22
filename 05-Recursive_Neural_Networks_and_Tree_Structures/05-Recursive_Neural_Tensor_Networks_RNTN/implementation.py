"""
models.py

Encoder / Decoder / ProjectionHead for CNN-based image watermarking.

All three are plain nn.Module subclasses so they can be trained, checkpointed,
and swapped independently. Shapes below use N=batch size, L=payload_bits,
H=W=resolution.

IMPORTANT DESIGN NOTE (found via actual training, not assumed): an earlier
version of this Encoder expanded the payload through a single shared
1-channel spatial map. In practice this created a "winner-take-all"
bottleneck -- diagnosing a stalled training run by checking *per-bit*
accuracy showed one bit reaching ~98% while every other bit sat at chance
(~50%), because all L bits were forced to compete for the same channel's
capacity and gradient competition let only one direction win. Giving the
payload its own channel per bit (below) fixed this: accuracy became a
smooth, monotonically improving curve across all bits instead of one bit
solved and the rest stuck. This is documented here so future edits don't
accidentally reintroduce the single-channel bottleneck.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, kernel_size=3, stride=1):
    """Conv -> BatchNorm -> ReLU, the shared building block for the
    encoder/decoder conv stacks (per the spec: 4-6 conv layers, BN, ReLU)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Embeds an L-bit payload into a cover image.

    Input:
        cover:   FloatTensor [N, 3, H, W] in [-1, 1]
        payload: FloatTensor [N, L] of 0/1 bits
    Output:
        watermarked: FloatTensor [N, 3, H, W] in [-1, 1]
                     (= cover + bounded residual, then clamped)

    Mechanism: the payload is linearly expanded and spatially replicated
    into `pm_channels` feature-map channels (one per bit by default -- see
    the module docstring for why this matters), concatenated channel-wise
    with the cover image, then a 4-layer conv stack (BN + ReLU) predicts a
    bounded (tanh) residual added back to the cover image.
    """

    def __init__(self, resolution: int = 128, payload_bits: int = 32,
                 base_channels: int = 32, residual_scale: float = 0.2,
                 pm_channels: int = None):
        super().__init__()
        self.resolution = resolution
        self.payload_bits = payload_bits
        self.residual_scale = residual_scale  # bounds how visible the watermark can be
        # One payload channel per bit by default (capped at 32 so very large
        # payloads don't blow up the conv input width); see module docstring.
        self.pm_channels = pm_channels or min(payload_bits, 32)

        self.payload_fc = nn.Linear(payload_bits, self.pm_channels * resolution * resolution)

        c = base_channels
        self.conv_stack = nn.Sequential(
            conv_block(3 + self.pm_channels, c),
            conv_block(c, c * 2),
            conv_block(c * 2, c * 2),
            conv_block(c * 2, c),
            nn.Conv2d(c, 3, kernel_size=3, padding=1),  # -> residual, no BN/ReLU on output
        )

    def forward(self, cover: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        n = cover.shape[0]
        payload_map = self.payload_fc(payload)                       # [N, pm_channels*H*W]
        payload_map = payload_map.view(n, self.pm_channels, self.resolution, self.resolution)
        payload_map = torch.tanh(payload_map)  # keep in a bounded range like the image

        x = torch.cat([cover, payload_map], dim=1)                   # [N, 3+pm_channels, H, W]
        residual = torch.tanh(self.conv_stack(x)) * self.residual_scale
        watermarked = torch.clamp(cover + residual, -1.0, 1.0)
        return watermarked


class Decoder(nn.Module):
    """Recovers the L-bit payload from a (possibly attacked) image.

    Input:
        image: FloatTensor [N, 3, H, W] in [-1, 1]
    Output:
        bits:  FloatTensor [N, L] in [0, 1] (post-sigmoid, threshold at 0.5)
        features: FloatTensor [N, feat_dim] penultimate feature map
                  (global-average-pooled), exposed for the projection head.
    """

    def __init__(self, resolution: int = 128, payload_bits: int = 32,
                 base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.conv_stack = nn.Sequential(
            conv_block(3, c),
            conv_block(c, c * 2, stride=2),
            conv_block(c * 2, c * 2, stride=2),
            conv_block(c * 2, c * 4, stride=2),
            conv_block(c * 4, c * 4),
        )
        self.feat_dim = c * 4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.feat_dim, payload_bits)

    def forward(self, image: torch.Tensor):
        feat_map = self.conv_stack(image)          # [N, feat_dim, H', W']
        feat = self.pool(feat_map).flatten(1)       # [N, feat_dim]  (penultimate features)
        bits = torch.sigmoid(self.fc(feat))         # [N, L]
        return bits, feat


class ProjectionHead(nn.Module):
    """Small MLP mapping decoder penultimate features to a normalized
    embedding, used only for the contrastive (InfoNCE) loss during training.

    Input:  features [N, feat_dim]
    Output: embedding [N, proj_dim], L2-normalized
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        z = self.net(features)
        return F.normalize(z, dim=1)


if __name__ == "__main__":
    # Quick shape smoke test.
    res, bits = 64, 32
    enc = Encoder(res, bits)
    dec = Decoder(res, bits)
    proj = ProjectionHead(dec.feat_dim)

    cover = torch.randn(4, 3, res, res).clamp(-1, 1)
    payload = torch.randint(0, 2, (4, bits)).float()

    wm = enc(cover, payload)
    pred_bits, feat = dec(wm)
    emb = proj(feat)

    print("watermarked:", wm.shape, wm.min().item(), wm.max().item())
    print("pred_bits:", pred_bits.shape)
    print("embedding:", emb.shape, emb.norm(dim=1))
