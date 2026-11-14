"""Forecast look-ahead, accumulation consistency and trajectory continuity."""
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from downscale.precipitation import RainConfig, RainNet, features, predict, rain_loss, save_model, temporal_scores
from downscale.rain_temporal import context_inputs, hourly_baseline, trajectory_noise


def pack():
    values = np.zeros((24, 3, 9, 11), dtype="float32")
    values[:, 0] = np.repeat([1., 4., 0., 2.], 6)[:, None, None]
    values[:, 1] = np.repeat([280., 286., 283., 281.], 6)[:, None, None]
    values[:, 2] = np.tile(np.arange(1, 7) / 6, 4)[:, None, None]
    return xr.Dataset({"predictors": (("time", "variable", "lat", "lon"), values),
                       "target": (("time", "lat", "lon"), values[:, 0])},
                      coords={"time": pd.date_range("2021-01-01T01:00", periods=24, freq="h"),
                              "variable": ["tp", "t2m", "phase"], "lat": np.arange(9), "lon": np.arange(11)},
                      attrs={"forecast_interval_hours": 6, "precipitation_units": "mm", "interval": "(time-1h,time]"})


def test_cumulative_interpolation_preserves_totals_and_dry_intervals():
    ds = pack()
    baseline = hourly_baseline(ds)
    assert (baseline >= 0).all()
    np.testing.assert_allclose(baseline.reshape(4, 6, 9, 11).sum(1), ds.predictors.values[:, 0].reshape(4, 6, 9, 11).sum(1), rtol=1e-6)
    np.testing.assert_array_equal(baseline[12:18], 0)
    assert not np.all(baseline[:6] == 1)
    changed = ds.copy(deep=True)
    changed.target.values[:] = 500
    np.testing.assert_array_equal(hourly_baseline(changed), baseline)


def test_next_forcing_is_visible_without_using_next_observation():
    ds = pack()
    a, names = context_inputs(ds)
    assert np.any(a[:6, names.index("next_t2m")] > 280)
    assert "phase" not in names and "phase_sin" not in names
    changed = ds.copy(deep=True)
    changed.predictors.values[6:12, 1] = 290
    b, _ = context_inputs(changed)
    assert not np.array_equal(a[:6], b[:6])
    changed = ds.copy(deep=True)
    changed.target.values[6:12] = 100
    c, _ = context_inputs(changed)
    np.testing.assert_array_equal(a, c)


def test_context_and_cumulative_rain_do_not_bridge_missing_events():
    ds = pack()
    ds = ds.assign_coords(time=np.concatenate([ds.time.values[:12], ds.time.values[12:] + np.timedelta64(30, "D")]))
    a, names = context_inputs(ds)
    assert a[11, names.index("future_context_weight"), 0, 0] < .1
    assert a[12, names.index("past_context_weight"), 0, 0] < .1
    changed = ds.copy(deep=True)
    changed.predictors.values[12:, :2] += 100
    b, _ = context_inputs(changed)
    np.testing.assert_array_equal(a[:12], b[:12])


def test_noise_does_not_reset_at_six_hour_boundaries():
    torch.manual_seed(12)
    n = trajectory_noise(pack().time.values, 2048, 1, 1, factor=1).squeeze()
    assert torch.all((n.var(1) > .85) & (n.var(1) < 1.15))
    # Member identity survives the window transition; independent redraws would
    # have correlation near zero here.
    correlation = torch.corrcoef(torch.stack([n[5], n[6]]))[0, 1]
    assert correlation > .95


def test_prediction_is_independent_of_window_batching(tmp_path):
    ds = pack()
    x, mean, std = features(ds, temporal_context=True)
    net = RainNet(x.shape[1], RainConfig(base=4, depth=2, temporal_context=True))
    torch.nn.init.normal_(net.unet.out.weight, std=.02)
    model = tmp_path / "model.pt"
    save_model(model, net, ds, mean, std)
    torch.manual_seed(123)
    a = predict(model, ds, members=3, batch=6)
    torch.manual_seed(123)
    b = predict(model, ds, members=3, batch=12)
    np.testing.assert_allclose(a.precipitation, b.precipitation, atol=1e-5, rtol=1e-5)


def test_increment_loss_penalizes_missing_real_changes():
    truth = torch.zeros(12, 3, 3)
    truth[6:] = 5
    pred = torch.ones(12, 2, 3, 3) * 2
    basic = rain_loss(pred, truth, 6)
    temporal = rain_loss(pred, truth, 6, sequence_hours=12, temporal_weight=.3)
    assert temporal > basic


def test_boundary_metric_detects_artificial_steps():
    ds = pack()
    coarse = ds.predictors.values[:, 0]
    truth = hourly_baseline(ds)
    pred = np.repeat(coarse[:, None], 2, axis=1)
    report = temporal_scores(pred, truth, coarse, truth, ds.time.values)
    assert report["model_mean"]["within_window_abs_change_mm"] == 0
    assert report["model_mean"]["boundary_abs_change_mm"] > 0
    assert report["boundary_count"] == 3
    assert report["increment_crps_mm"] > 0
