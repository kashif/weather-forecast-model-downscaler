"""Joint temporal decoding preserves the spatial prior and stitches same-hour estimates."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import xarray as xr

from downscale.downscaler import UNet
from downscale.precipitation import RainConfig, RainNet, features, predict, save_model
from downscale.rain_sequence import TimeMix, TrajectoryUNet, predict_overlapping


def test_zero_temporal_adapters_preserve_pretrained_spatial_outputs():
    torch.manual_seed(5)
    spatial = UNet(4, 1, base=4, depth=2, upsample="bilinear")
    torch.nn.init.normal_(spatial.out.weight, std=.1)
    joint = TrajectoryUNet(4, base=4, depth=2, sequence_hours=12)
    missing, unexpected = joint.load_state_dict(spatial.state_dict(), strict=False)
    assert missing and not unexpected and all("time_" in k for k in missing)
    x = torch.randn(24, 4, 12, 16)  # twelve hours × two members
    torch.testing.assert_close(joint(x, members=2), spatial(x))


def test_time_mixer_uses_neighbouring_hours_without_mixing_members():
    mix = TimeMix(1, 12)
    with torch.no_grad():
        mix.conv.weight[0, 0, 2] = 1
    x = torch.zeros(24, 1, 2, 2)
    x[2, 0] = 7  # member zero at the next hour
    y = mix(x, members=2)
    torch.testing.assert_close(y[0], torch.ones_like(y[0]) * 7)
    assert y[1].abs().sum() == 0
    y.sum().backward()
    assert mix.conv.weight.grad is not None


def test_overlap_blends_same_hour_without_smoothing_neighbouring_hours():
    class Identity:
        cfg = SimpleNamespace(sequence_hours=12, window_hours=6)

        def __call__(self, x, baseline, members, noise):
            return baseline[:, None].repeat(1, members, 1, 1)

    times = pd.date_range("2024-01-01T01:00", periods=24, freq="h")
    baseline = np.arange(24, dtype="float32")[:, None, None] * np.ones((24, 2, 2), dtype="float32")
    result = predict_overlapping(Identity(), np.zeros((24, 1, 2, 2), dtype="float32"), baseline,
                                 torch.zeros(24, 2, 1, 1, 1), times, "cpu", 2)
    np.testing.assert_allclose(result[:, 0], baseline, rtol=1e-6, atol=1e-6)


def test_joint_rain_network_has_gradients_and_nonnegative_hourly_outputs():
    cfg = RainConfig(base=4, depth=2, architecture="joint", sequence_hours=12)
    net = RainNet(4, cfg)
    pred = net(torch.randn(12, 4, 9, 11), torch.ones(12, 9, 11), members=2)
    assert pred.shape == (12, 2, 9, 11)
    assert torch.all(pred >= 0)
    pred.mean().backward()
    assert net.unet.time_down[0].conv.weight.grad is not None


def test_warm_started_joint_training_and_inference_roundtrip(tmp_path):
    from scripts.precipitation import train
    n, h, w = 36, 5, 7
    rain = np.ones((n, h, w), dtype="float32")
    phase = np.broadcast_to(np.tile(np.arange(1, 7) / 6, 6)[:, None, None], rain.shape).astype("float32")
    ds = xr.Dataset({"predictors": (("time", "variable", "lat", "lon"), np.stack([rain, phase], 1)),
                     "target": (("time", "lat", "lon"), rain * .8)},
                    coords={"time": pd.date_range("2020-01-01T01:00", periods=n, freq="h"),
                            "variable": ["tp", "phase"], "lat": np.arange(h), "lon": np.arange(w)},
                    attrs={"forecast_interval_hours": 6, "precipitation_units": "mm", "interval": "(time-1h,time]"})
    x, mean, std = features(ds.isel(time=slice(0, 24)))
    initial = tmp_path / "initial.pt"
    save_model(initial, RainNet(x.shape[1], RainConfig(base=4, depth=2)), ds, mean, std)
    cfg = {"validation_start": "2020-01-02T01:00", "model": {"base": 4, "depth": 2, "members": 2, "architecture": "joint", "sequence_hours": 12},
           "training": {"seed": 0, "epochs": 1, "batch_windows": 1, "sequence_windows": 2,
                        "max_steps": 1, "learning_rate": .0001, "temporal_weight": .1,
                        "mean_loss_weight": .1, "init_checkpoint": str(initial), "selection": "joint"}}
    model = train(ds, cfg, tmp_path / "run", "cpu")
    output = predict(model, ds.isel(time=slice(24, None)), members=2)
    assert output.precipitation.shape == (12, 2, h, w)
    assert output.attrs["architecture"] == "joint"
