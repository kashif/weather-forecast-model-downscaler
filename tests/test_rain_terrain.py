"""Static terrain resampling, feature order, warm starts and self-contained inference."""
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from downscale.precipitation import RainConfig, RainNet, features, predict, save_model
from downscale.rain_terrain import CHANNELS, expand_terrain_weights, make_terrain, load_static


def sources(partial=False):
    axis = np.linspace(45.2, 46.8, 7) if partial else np.linspace(44., 48., 9)
    lon = axis - 44
    yy, xx = np.meshgrid(axis, lon, indexing="ij")
    fine = xr.Dataset({"elevation": (("y", "x"), np.ones_like(yy) * 500),
                       "land_fraction": (("y", "x"), np.ones_like(yy))},
                      coords={"lat": (("y", "x"), yy), "lon": (("y", "x"), xx)})
    coarse = xr.Dataset({"elevation": (("lat", "lon"), np.ones((5, 5)) * 100),
                         "land_fraction": (("lat", "lon"), np.ones((5, 5)))},
                        coords={"lat": np.linspace(44, 48, 5), "lon": np.linspace(0, 4, 5)})
    fine.elevation.attrs["units"] = coarse.elevation.attrs["units"] = "m"
    return fine, coarse


def test_constant_terrain_has_correct_units_gradients_and_land_fraction(tmp_path):
    fine, coarse = sources()
    result = make_terrain(fine, coarse, [45., 46., 47.], [1., 2., 3.])
    assert result.static_channel.values.tolist() == CHANNELS
    np.testing.assert_allclose(result.elevation, 500)
    np.testing.assert_allclose(result.static.sel(static_channel="fine_minus_coarse_km"), .4)
    np.testing.assert_allclose(result.static.sel(static_channel="slope_east_percent"), 0, atol=1e-10)
    np.testing.assert_allclose(result.static.sel(static_channel="slope_north_percent"), 0, atol=1e-10)
    np.testing.assert_allclose(result.static.sel(static_channel="fine_coverage"), 1)
    np.testing.assert_allclose(result.static.sel(static_channel="coast_proximity"), 0)
    path = tmp_path / "static.zarr"
    result.to_zarr(path, zarr_format=2)
    a, _ = load_static(path, result.lat, result.lon)
    np.testing.assert_array_equal(a, result.static)
    with pytest.raises(ValueError, match="lat"):
        load_static(path, result.lat + .1, result.lon)


def test_fine_crop_gaps_use_coarse_terrain_with_explicit_coverage():
    fine, coarse = sources(partial=True)
    result = make_terrain(fine, coarse, [45., 46., 47.], [1., 2., 3.])
    assert np.isfinite(result.static).all()
    coverage = result.static.sel(static_channel="fine_coverage").values[0, 0]
    assert coverage < .5
    assert result.elevation.values[0, 0] == pytest.approx(100 + 400 * coverage)
    assert result.elevation.values[1, 1] == pytest.approx(500)
    assert result.static.sel(static_channel="fine_coverage").values.min() < 1
    fine.elevation.attrs["units"] = "m2/s2"
    with pytest.raises(ValueError, match="metres"):
        make_terrain(fine, coarse, [45., 46., 47.], [1., 2., 3.])


def test_static_channels_preserve_warm_start_and_are_embedded_in_checkpoint(tmp_path):
    n, h, w = 12, 9, 11
    rain = np.ones((n, h, w), dtype="float32")
    phase = np.broadcast_to(np.tile(np.arange(1, 7) / 6, 2)[:, None, None], rain.shape).astype("float32")
    ds = xr.Dataset({"predictors": (("time", "variable", "lat", "lon"), np.stack([rain, phase], 1))},
                    coords={"time": pd.date_range("2021-01-01T01:00", periods=n, freq="h"),
                            "variable": ["tp", "phase"], "lat": np.linspace(45, 47, h), "lon": np.linspace(1, 3, w)},
                    attrs={"forecast_interval_hours": 6, "precipitation_units": "mm", "interval": "(time-1h,time]"})
    static = np.random.default_rng(0).normal(size=(len(CHANNELS), h, w)).astype("float32")
    x, mean, std = features(ds)
    xt, mt, st = features(ds, mean, std, static=static)
    np.testing.assert_array_equal(xt[:, -len(CHANNELS):], np.broadcast_to(static, (n, *static.shape)))
    np.testing.assert_array_equal(mean, mt)
    np.testing.assert_array_equal(std, st)
    old = RainNet(x.shape[1], RainConfig(base=4, depth=2, architecture="joint"))
    torch.nn.init.normal_(old.unet.out.weight, std=.02)
    new = RainNet(xt.shape[1], RainConfig(base=4, depth=2, architecture="joint", terrain=True))
    new.load_state_dict(expand_terrain_weights(old.state_dict(), new.state_dict(), len(CHANNELS)))
    noise = torch.randn(n, 2, 1, 1, 1)
    new(torch.tensor(xt), torch.ones(n, h, w), members=2, noise=noise).mean().backward()
    assert new.unet.inc.net[0].weight.grad[:, x.shape[1]:-1].abs().sum() > 0
    before, after = tmp_path / "old.pt", tmp_path / "terrain.pt"
    save_model(before, old, ds, mean, std)
    with pytest.raises(ValueError, match="static covariates"):
        save_model(after, new, ds, mean, std)
    save_model(after, new, ds, mean, std, static, {"source": "synthetic"})
    torch.manual_seed(12)
    p = predict(before, ds, members=2, noise_seed=5)
    torch.manual_seed(987)
    q = predict(after, ds, members=2, noise_seed=5)
    np.testing.assert_allclose(p.precipitation, q.precipitation, rtol=1e-5, atol=1e-5)
    assert q.attrs["terrain"] is True
