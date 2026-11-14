"""Regional probabilistic rainfall downscaler with physical-unit multiscale CRPS."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch import nn

from .downscaler import UNet, multiscale_fair_crps, time_features


@dataclass
class RainConfig:
    base: int = 16
    depth: int = 3
    noise_factor: int = 8
    members: int = 4
    window_hours: int = 6


def check_pack(ds: xr.Dataset, target=True):
    if ds.predictors.dims != ("time", "variable", "lat", "lon"):
        raise ValueError("predictors must have dimensions time, variable, lat, lon")
    if str(ds.variable.values[0]) != "tp":
        raise ValueError("the first predictor must be tp, hourly or interval-mean depth in mm")
    if ds.attrs.get("precipitation_units") != "mm":
        raise ValueError("pack must declare precipitation_units: mm")
    if target and ds.target.dims != ("time", "lat", "lon"):
        raise ValueError("target must have dimensions time, lat, lon")
    times = ds.time.values
    if len(times) == 0 or len(np.unique(times)) != len(times) or np.any(np.diff(times) <= np.timedelta64(0, "s")):
        raise ValueError("times must be nonempty, unique and increasing")
    for coord in ("lat", "lon"):
        if ds.sizes[coord] < 2 or not np.all(np.diff(ds[coord].values) > 0):
            raise ValueError(f"{coord} must be strictly increasing with at least two cells")
    if not np.isfinite(ds.predictors.values).all():
        raise ValueError("predictors contain missing/nonfinite values")
    if np.any(ds.predictors.sel(variable="tp").values < 0):
        raise ValueError("rainfall predictors must be nonnegative")
    if target and (not np.isfinite(ds.target.values).all() or np.any(ds.target.values < 0)):
        raise ValueError("rainfall targets must be finite and nonnegative")


def features(ds: xr.Dataset, mean=None, std=None):
    """Normalise using training-only statistics, append geography and calendar."""
    data = ds.predictors.values.astype("float32").copy()
    data[:, 0] = np.log1p(data[:, 0])
    if mean is None:
        mean = data.mean((0, 2, 3), keepdims=True)
        std = np.maximum(data.std((0, 2, 3), keepdims=True), 1e-4)
    data = (data - mean) / std
    n, _, h, w = data.shape
    lat, lon = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
    geo = np.broadcast_to(np.stack([lat / 90, lon / 180])[None], (n, 2, h, w))
    calendar = np.broadcast_to(time_features(ds.time.values)[:, :, None, None], (n, 4, h, w))
    return np.concatenate([data, geo, calendar], 1).astype("float32"), mean, std


class RainNet(nn.Module):
    def __init__(self, channels: int, cfg: RainConfig):
        super().__init__()
        if cfg.noise_factor < 1 or cfg.members < 2 or cfg.depth < 1 or cfg.window_hours < 1:
            raise ValueError("positive depth/noise_factor and at least two members required")
        self.cfg = cfg
        self.unet = UNet(channels + 1, 1, cfg.base, cfg.depth, "bilinear")

    def forward(self, x, baseline, members=None):
        """Return nonnegative hourly depths (batch, members, lat, lon), in mm."""
        m = self.cfg.members if members is None else members
        if m < 1:
            raise ValueError("members must be positive")
        b, _, h, w = x.shape
        if b % self.cfg.window_hours:
            raise ValueError("batch must contain complete forecast windows")
        noise = torch.randn(b // self.cfg.window_hours, m, 1, max(1, h // self.cfg.noise_factor),
                            max(1, w // self.cfg.noise_factor), device=x.device)
        noise = noise[:, None].expand(-1, self.cfg.window_hours, -1, -1, -1, -1).reshape(b * m, 1, *noise.shape[-2:])
        noise = F.interpolate(noise, (h, w), mode="bilinear", align_corners=False)
        inputs = torch.cat([x.repeat_interleave(m, 0), noise], 1)
        multiple = 2 ** self.cfg.depth
        inputs = F.pad(inputs, (0, (-w) % multiple, 0, (-h) % multiple), mode="replicate")
        correction = self.unet(inputs)[:, 0, :h, :w].reshape(b, m, h, w)
        # A small offset gives initially dry pixels a gradient. Upper bound prevents overflow.
        log_depth = (torch.log1p(baseline[:, None]) + 0.01 + correction).clamp(0, np.log1p(500.0))
        return torch.expm1(log_depth)


def rain_loss(pred, truth, window_hours=1):
    factors = [f for f in (2, 4, 8) if f <= min(truth.shape[-2:])]
    from .downscaler import fair_crps
    # WN3 A.1.2 explicitly adds domain-mean CRPS at weight 0.3.
    loss = multiscale_fair_crps(pred, truth, factors=factors)
    loss = loss + 0.3 * fair_crps(pred.mean((-2, -1), keepdim=True), truth.mean((-2, -1), keepdim=True))
    if window_hours > 1:
        if len(truth) % window_hours:
            raise ValueError("loss requires complete forecast windows")
        p = pred.reshape(-1, window_hours, *pred.shape[1:]).sum(1)
        t = truth.reshape(-1, window_hours, *truth.shape[1:]).sum(1)
        loss = loss + 0.3 * fair_crps(p, t) / window_hours
    return loss


def score(pred, truth):
    """Physical-unit scores against the supplied target; no independent-skill claim."""
    from .evaluation import fair_crps
    pred, truth = np.asarray(pred), np.asarray(truth)
    mean = pred.mean(1)
    result = {"rmse_mm": float(np.sqrt(np.mean((mean - truth) ** 2))),
              "bias_mm": float(np.mean(mean - truth)),
              "crps_mm": float(fair_crps(pred, truth, axis=1).mean()) if pred.shape[1] > 1 else float(np.abs(mean - truth).mean())}
    for threshold in (0.1, 1.0, 5.0, 10.0):
        observed = truth >= threshold
        prob = (pred >= threshold).mean(1)
        event = prob >= 0.5
        union = np.logical_or(event, observed).sum()
        result[f"brier_{threshold:g}mm"] = float(np.mean((prob - observed) ** 2))
        result[f"csi_{threshold:g}mm"] = float(np.logical_and(event, observed).sum() / union) if union else None
        result[f"observed_events_{threshold:g}mm"] = int(observed.sum())
        # Reliability of event probabilities, including empty-bin counts.
        reliability = []
        for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
            mask = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
            reliability.append({"n": int(mask.sum()), "mean_probability": float(prob[mask].mean()) if mask.any() else None,
                                "observed_frequency": float(observed[mask].mean()) if mask.any() else None})
        result[f"reliability_{threshold:g}mm"] = reliability
        # Neighbourhood FSS for the ensemble-median event footprint. Padding is
        # excluded, so edge neighbourhoods only use points within the domain.
        for size in (1, 3, 9):
            if size > min(truth.shape[-2:]):
                continue
            a = F.avg_pool2d(torch.tensor(event[:, None], dtype=torch.float32), size, stride=1, padding=size // 2, count_include_pad=False).numpy()
            b = F.avg_pool2d(torch.tensor(observed[:, None], dtype=torch.float32), size, stride=1, padding=size // 2, count_include_pad=False).numpy()
            denom = np.mean(a * a + b * b)
            result[f"fss_{threshold:g}mm_{size}px"] = float(1 - np.mean((a - b) ** 2) / denom) if denom else None
    return result


def save_model(path, net, ds, mean, std):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "france-rain-v1", "config": asdict(net.cfg),
                "variables": ds.variable.values.tolist(), "mean": torch.tensor(mean), "std": torch.tensor(std),
                "lat": torch.tensor(ds.lat.values), "lon": torch.tensor(ds.lon.values),
                "data_attrs": dict(ds.attrs), "state_dict": net.state_dict()}, path)


def predict(path, ds, device="cpu", members=8, batch=4):
    check_pack(ds, target=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != "france-rain-v1" or ds.variable.values.tolist() != checkpoint["variables"]:
        raise ValueError("checkpoint format or predictor channels differ")
    for name in ("lat", "lon"):
        if not np.array_equal(ds[name].values, checkpoint[name].numpy()):
            raise ValueError(f"{name} differs from the training grid; regrid predictors first")
    for name in ("forecast_interval_hours", "interval"):
        if ds.attrs.get(name) != checkpoint["data_attrs"].get(name):
            raise ValueError(f"{name} differs from the training convention")
    x, _, _ = features(ds, checkpoint["mean"].numpy(), checkpoint["std"].numpy())
    net = RainNet(x.shape[1], RainConfig(**checkpoint["config"])).to(device).eval()
    net.load_state_dict(checkpoint["state_dict"])
    window = net.cfg.window_hours
    if len(x) % window:
        raise ValueError("inference requires complete forecast windows")
    batch = max(window, (batch // window) * window)
    outputs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outputs.append(net(torch.tensor(x[i:i + batch], device=device),
                               torch.tensor(ds.predictors.values[i:i + batch, 0], dtype=torch.float32, device=device), members).cpu().numpy())
    result = xr.Dataset({"precipitation": (("time", "sample", "lat", "lon"), np.concatenate(outputs))},
                        coords={"time": ds.time, "sample": np.arange(members), "lat": ds.lat, "lon": ds.lon})
    result.precipitation.attrs = {"units": "mm", "interval": "(time-1h,time]"}
    result.attrs = {"model": str(path), "target_source": checkpoint["data_attrs"].get("target_source", "unknown")}
    return result
