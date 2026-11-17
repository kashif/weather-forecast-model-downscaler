"""Prepare fixed-date, cross-year calibration pairs and frozen ensembles."""
from __future__ import annotations
import argparse
import fcntl
import shutil
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from downscale.precip_data import open_store,crop,ERA5_NAMES,plan_pardig,PARDIG
from downscale.sources import ERA5_ARCO
from downscale.precipitation import check_pack
from downscale.rain_resume import atomic_json
from scripts.precipitation import write_pack
from scripts.precipitation_year import targets,day_pack,hour_ends,atomic_array
from scripts.precipitation_live import conditional_ensemble


def prepare(cfg,cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    lock=(cache/'.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('tail data preparation is already running') from None
    checkpoint=Path(cfg['checkpoint'])
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
    dates=[(role,pd.Timestamp(day)) for role,days in cfg['splits'].items() for day in days]
    unique=sorted({day for _,day in dates})
    if len(unique)!=len(dates):raise ValueError('fit, selection and test dates overlap')
    times=pd.DatetimeIndex(np.concatenate([hour_ends(day,day+pd.Timedelta(days=1)).values for day in unique]))
    for role,day in dates:
        stamps=hour_ends(day,day+pd.Timedelta(days=1)).values
        for split in ('training','validation'):
            a=np.datetime64(cp['data_attrs'][split+'_start']);b=np.datetime64(cp['data_attrs'][split+'_end'])
            if ((stamps>=a)&(stamps<=b)).any():raise ValueError(f'{role} overlaps downscaler {split}')
    manifest=dict(config=cfg,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  times_sha256=hashlib.sha256(times.values.tobytes()).hexdigest())
    path=cache/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('cache config/checkpoint differs')
    atomic_json(path,manifest)
    grid=xr.open_zarr(cfg['static_path']).load()
    for name in ['lat','lon']:
        if not np.array_equal(grid[name],cp[name].numpy()):raise ValueError('static/checkpoint grid differs')
    target_path=cache/'targets.npy'
    if target_path.exists():
        truth=np.load(target_path,mmap_mode='r')
        if truth.shape!=(len(times),len(grid.lat),len(grid.lon)):raise ValueError('cached target shape differs')
    else:
        plan=plan_pardig(times,tuple(cfg['region']))
        atomic_json(cache/'read_plan.json',plan)
        print(json.dumps(plan),flush=True)
        if plan['chosen']=='timeseries':
            truth=targets(times,grid,cache,workers=4)
        else:
            source_url=plan['candidates'][plan['chosen']]['url']
            result=PARDIG(source_url,tuple(cfg['region']))(times,['tp']).sel(variable='tp',drop=True)
            for name in ['lat','lon']:
                if not np.array_equal(result[name],grid[name]):raise ValueError('PARDIG/static grid differs')
            atomic_array(target_path,result.values.astype('float32')*1000)
            truth=np.load(target_path,mmap_mode='r')
    box=cfg['region']
    source=crop(open_store(ERA5_ARCO)[list(ERA5_NAMES.values())],(box[0]-1,box[1]+1,box[2]-1,box[3]+1))
    for index,(role,day) in enumerate(dates):
        directory=cache/role/f'{day:%Y%m%d}';directory.mkdir(parents=True,exist_ok=True)
        pair_path=directory/'pairs.zarr';pred_path=directory/'predictions.zarr'
        if not pair_path.exists():
            stamps=hour_ends(day,day+pd.Timedelta(days=1))
            idx=times.get_indexer(stamps)
            pack=day_pack(source,stamps,np.asarray(truth[idx]),grid,cfg)
            temporary=pair_path.with_name(pair_path.name+'.tmp')
            if temporary.exists():shutil.rmtree(temporary)
            write_pack(pack,pair_path)
        if not pred_path.exists():
            pack=xr.open_zarr(pair_path).load();check_pack(pack)
            prediction=conditional_ensemble(checkpoint,pack,cfg['members'])
            prediction.attrs['checkpoint_sha256']=manifest['checkpoint_sha256']
            temporary=pred_path.with_name(pred_path.name+'.tmp')
            if temporary.exists():shutil.rmtree(temporary)
            write_pack(prediction,pred_path)
        progress=dict(stage='frozen_predictions',completed=index+1,total=len(dates),split=role,day=str(day))
        atomic_json(cache/'progress.json',progress);print(json.dumps(progress),flush=True)
    atomic_json(cache/'progress.json',dict(stage='complete',days=len(dates),hours=len(times)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_rain_tail.yaml')
    parser.add_argument('--cache',default='data/precipitation/gpd_calibration')
    args=parser.parse_args()
    prepare(yaml.safe_load(Path(args.config).read_text()),args.cache)
