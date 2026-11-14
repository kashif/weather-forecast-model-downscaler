"""Animate an hourly rainfall evaluation against its coarse forcing and PARDIG."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from downscale.precipitation import check_pack


def make_movie(pairs, predictions, out, fps=2, gif=None, coastlines=True, comparison=None):
    check_pack(pairs)
    for name in ("time", "lat", "lon"):
        if not np.array_equal(pairs[name], predictions[name]):
            raise ValueError(f"prediction and target {name} differ")
    if predictions.precipitation.dims != ("time", "sample", "lat", "lon"):
        raise ValueError("expected precipitation(time, sample, lat, lon)")
    if predictions.precipitation.attrs.get("units") != "mm" or fps <= 0:
        raise ValueError("rainfall must be mm and fps must be positive")
    out = Path(out)
    if out.exists() or gif is not None and Path(gif).exists():
        raise FileExistsError("refusing to overwrite an existing movie")
    out.parent.mkdir(parents=True, exist_ok=True)
    coarse = pairs.predictors.sel(variable="tp").values
    first_title, first_label = "Coarse six-hour forcing\nERA5 interval mean", "Coarse forcing"
    if comparison is not None:
        for name in ("time", "lat", "lon"):
            if not np.array_equal(pairs[name], comparison[name]):
                raise ValueError(f"comparison {name} differs")
        coarse = comparison.precipitation.mean("sample").values
        first_title, first_label = "Previous model mean\nIndependent six-hour windows", "Previous model"
    members = predictions.precipitation.values
    mean = members.mean(1)
    truth = pairs.target.values
    times = pd.DatetimeIndex(pairs.time.values)
    if not np.all(np.diff(times.values) == np.timedelta64(1, "h")):
        raise ValueError("movie hours must be consecutive")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13, 5.8), facecolor="#f8fafc")
    grid = fig.add_gridspec(2, 4, height_ratios=[2.6, 1.2], hspace=.24, wspace=.06,
                           left=.065, right=.94, top=.82, bottom=.13)
    palette = ["#ffffff", "#c9e9f6", "#80c6e3", "#3596c5", "#2268a7", "#33428c", "#7950a0", "#c65887", "#e6874a"]
    levels = [0, .1, .5, 1, 2, 5, 10, 20, 40, 100]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(levels, cmap.N)
    meshes = []
    fields = [coarse, members[:, 0], mean, truth]
    noise_title = "Evolving noise across windows" if predictions.attrs.get("noise_mode") == "continuous_anchors" else "Shared noise within each 6h"
    if predictions.attrs.get("architecture") == "joint":
        noise_title = "Joint twelve-hour decoding"
    titles = [first_title, f"Downscaled member\n{noise_title}",
              f"Downscaled ensemble mean\nAverage of {members.shape[1]} samples", "PARDIG reference\nSatellite-derived target"]
    for j, (values, title) in enumerate(zip(fields, titles)):
        ax = fig.add_subplot(grid[0, j], projection=projection)
        ax.set_extent([float(pairs.lon.min()), float(pairs.lon.max()), float(pairs.lat.min()), float(pairs.lat.max())])
        mesh = ax.pcolormesh(pairs.lon, pairs.lat, values[0], cmap=cmap, norm=norm,
                             shading="auto", transform=projection, rasterized=True)
        if coastlines:
            ax.coastlines(resolution="50m", linewidth=.45, color="#56616e")
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=.35, edgecolor="#7a8490")
        ax.set_title(title, fontsize=10, loc="left", pad=10, color="#172b40")
        meshes.append(mesh)
    bar_ax = fig.add_axes([.955, .42, .012, .32])
    fig.colorbar(meshes[0], cax=bar_ax, ticks=[.1, 1, 5, 10, 20, 40], extend="max")
    bar_ax.set_title("mm", fontsize=9, pad=6)
    ax = fig.add_subplot(grid[1, :])
    regional = members.mean((-2, -1))
    ax.plot(times, coarse.mean((-2, -1)), color="#82909f", label=first_label, linewidth=1.5)
    ax.plot(times, regional.mean(1), color="#176dba", label="Downscaled mean", linewidth=1.8)
    ax.fill_between(times, np.quantile(regional, .05, axis=1), np.quantile(regional, .95, axis=1),
                    color="#176dba", alpha=.15, label="5–95% sample range")
    ax.plot(times, truth.mean((-2, -1)), color="#17394b", label="PARDIG", linewidth=1.7)
    cursor = ax.axvline(times[0], color="#d37a25", linewidth=1.5)
    ax.set_ylabel("Domain mean\n(mm/hour)", fontsize=9)
    ax.grid(alpha=.15)
    ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax.tick_params(labelsize=8)
    heading = fig.text(.045, .955, "Hourly rainfall over France", fontsize=19, weight="bold", color="#172b40")
    stamp = fig.text(.045, .904, "", fontsize=11, color="#485b6d")
    fig.text(.045, .025, "Held-out date • ERA5 global-state proxy, not an operational forecast • 0.1° PARDIG target • Context domain includes neighbouring land and sea",
             fontsize=8, color="#5b6d7f")

    def update(i):
        for mesh, values in zip(meshes, fields):
            mesh.set_array(values[i].ravel())
        stamp.set_text(f"Hour ending {times[i]:%d %B %Y, %H:%M} UTC  |  {i + 1:02d} / {len(times)}")
        cursor.set_xdata([times[i], times[i]])
        return [*meshes, cursor, heading, stamp]

    update(0)
    fig.savefig(out.with_suffix(".png"), dpi=130, facecolor=fig.get_facecolor())
    movie = animation.FuncAnimation(fig, update, frames=len(times), interval=1000 / fps, blit=False)
    try:
        movie.save(out, writer=animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2200,
                                                    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]), dpi=110)
        if gif is not None:
            Path(gif).parent.mkdir(parents=True, exist_ok=True)
            movie.save(gif, writer=animation.PillowWriter(fps=fps), dpi=85)
    finally:
        plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gif")
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--comparison", help="optional previous model predictions for the first panel")
    args = parser.parse_args()
    make_movie(xr.open_zarr(args.pairs).load(), xr.open_zarr(args.predictions).load(),
               args.out, args.fps, args.gif,
               comparison=xr.open_zarr(args.comparison).load() if args.comparison else None)


if __name__ == "__main__":
    main()
