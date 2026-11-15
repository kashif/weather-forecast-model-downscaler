"""Joint space-time decoding with warm-startable spatial weights."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .downscaler import UNet
from .rain_temporal import segments


class TimeMix(nn.Module):
    """Learn time/channel interactions at each spatial scale; initially identity."""
    def __init__(self, channels, sequence_hours):
        super().__init__()
        self.hours = sequence_hours
        self.conv = nn.Conv3d(channels, channels, (3, 1, 1))
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, members):
        n, c, h, w = x.shape
        if n % (self.hours * members):
            raise ValueError("joint decoder requires complete hour/member sequences")
        groups = n // (self.hours * members)
        volume = x.reshape(groups, self.hours, members, c, h, w).permute(0, 2, 3, 1, 4, 5).reshape(groups * members, c, self.hours, h, w)
        volume = volume + self.conv(F.pad(volume, (0, 0, 0, 0, 1, 1), mode="replicate"))
        return volume.reshape(groups, members, c, self.hours, h, w).permute(0, 3, 1, 2, 4, 5).reshape(n, c, h, w)


class TrajectoryUNet(UNet):
    """Spatial U-Net plus temporal mixing throughout encoder and decoder."""
    def __init__(self, channels_in, base=16, depth=3, sequence_hours=12):
        super().__init__(channels_in, 1, base, depth, "bilinear")
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.time_down = nn.ModuleList([TimeMix(c, sequence_hours) for c in widths])
        self.time_up = nn.ModuleList([TimeMix(c, sequence_hours) for c in reversed(widths[:-1])])

    def forward(self, x, members=1):
        skips = [self.time_down[0](self.inc(x), members)]
        for block, mix in zip(self.down, self.time_down[1:]):
            skips.append(mix(block(F.avg_pool2d(skips[-1], 2)), members))
        h = skips.pop()
        for up, dec, mix in zip(self.up, self.dec, self.time_up):
            h = mix(dec(torch.cat([up(h), skips.pop()], 1)), members)
        return self.out(h)


def predict_overlapping(net, x, baseline, noise, times, device, members):
    """Blend overlapping estimates of the same hour, never adjacent hourly maps.

    Each window uses the same globally sampled latent for a given hour/member.
    Half-window overlaps and positive Hann weights suppress decoder edge effects.
    Only unavailable context outside each contiguous forecast segment is padded.
    """
    length = net.cfg.sequence_hours
    stride = length // 2
    output = np.zeros((len(x), members, *baseline.shape[-2:]), dtype="float32")
    weights = np.zeros(len(x), dtype="float32")
    blend = np.sin(np.pi * (np.arange(length) + .5) / length) ** 2
    for part in segments(times, net.cfg.window_hours):
        n = part.stop - part.start
        for start in range(-stride, n, stride):
            positions = start + np.arange(length)
            indices = np.clip(positions, 0, n - 1) + part.start
            valid = (positions >= 0) & (positions < n)
            values = net(torch.tensor(x[indices], device=device),
                         torch.tensor(baseline[indices], dtype=torch.float32, device=device), members,
                         noise=noise[indices]).cpu().numpy()
            chosen = positions[valid] + part.start
            output[chosen] += values[valid] * blend[valid, None, None, None]
            weights[chosen] += blend[valid]
    if np.any(weights == 0):
        raise ValueError("uncovered forecast hour")
    return output / weights[:, None, None, None]
