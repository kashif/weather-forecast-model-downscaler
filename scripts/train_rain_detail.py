"""Paired, fixed-budget fine-tune from the annual France rain checkpoint."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import xarray as xr
import yaml

from downscale.hourly import default_device
from scripts.precipitation import train


def main():
    config = yaml.safe_load(Path("config_precipitation_year.yaml").read_text())
    config["training"].update(
        init_checkpoint="runs/precipitation_year/model.pt",
        epochs=1, learning_rate=1e-5, max_steps=144, checkpoint_every_steps=24,
    )
    pack = xr.open_zarr("data/precipitation/year_2019/train_validation.zarr")
    # The midnight report is the end of the previous day's last hour.
    dates = pd.DatetimeIndex(pack.time.values) - pd.Timedelta(hours=1)
    pack = pack.isel(time=((dates.year == 2019) & dates.day.isin([5, 10, 20, 25]))
                    | ((dates.year == 2020) & dates.month.isin([1, 4, 7, 10]))).load()
    for name, weight in [("control", 0.0), ("detail", 0.5)]:
        arm = deepcopy(config)
        arm["training"]["detail_loss_weight"] = weight
        output = Path("runs/rain_detail") / name
        if not (output / "model.pt").exists():
            train(pack, arm, output, default_device())


if __name__ == "__main__":
    main()
