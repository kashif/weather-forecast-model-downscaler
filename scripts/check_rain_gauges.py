"""Independent hourly gauge check; frozen downscaler, calibrators and alert rules.

Run fetch_rain_gauges.py first. No satellite targets are downloaded or fitted.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from downscale.precip_data import open_store,crop,ERA5_NAMES
from downscale.sources import ERA5_ARCO
from downscale.station_interp import GridInterpolator
from downscale.rain_tail import TailCalibration,spliced_moments
from downscale.rain_alerts import paired_day_csi_interval
from downscale.rain_resume import atomic_json
from scripts.precipitation_year import day_pack,hour_ends
from scripts.precipitation_live import conditional_ensemble
from scripts.calibrate_rain_tail import forecast_features
from scripts.check_rain_alerts import probabilities,evaluate_alerts


def quality_control(rows,times,region,min_completeness):
    """Drop conflicting duplicates and moving stations; never replace missing by zero."""
    rows=rows.copy();before=len(rows)
    duplicate=rows.groupby(['station','time']).rain_mm.transform('nunique')>1
    conflicts=int(duplicate.sum());rows=rows.loc[~duplicate].drop_duplicates(['station','time'])
    south,north,west,east=region
    rows=rows.loc[rows.lat.between(south,north)&rows.lon.between(west,east)]
    spread=rows.groupby('station')[['lat','lon']].agg(lambda x:x.max()-x.min())
    moving=spread.index[(spread>1e-4).any(axis=1)]
    rows=rows.loc[~rows.station.isin(moving)]
    counts=rows.loc[rows.time.isin(times)].groupby('station').size()
    keep=counts.index[counts>=len(times)*min_completeness]
    rows=rows.loc[rows.station.isin(keep)].copy()
    stations=rows.groupby('station',sort=True).first()[['lat','lon','elevation_m','name']]
    return rows,stations,dict(input_rows=before,conflicting_rows=conflicts,moving_stations=len(moving),
                             retained_stations=len(stations),minimum_completeness=min_completeness)


def aligned_observations(rows,stations,times,offset=0):
    table=rows.pivot(index='time',columns='station',values='rain_mm')
    return table.reindex(index=times+pd.Timedelta(hours=offset),columns=stations.index).to_numpy()


def amount_scores(pred,obs,hours):
    """Nonoverlapping windows within each 24h day; require all gauge hours."""
    p=pred.reshape(-1,24//hours,hours,pred.shape[-1]).sum(2)
    o=obs.reshape(-1,24//hours,hours,obs.shape[-1]).sum(2)
    valid=np.isfinite(o)&np.isfinite(p);error=p[valid]-o[valid]
    if not len(error):return dict(windows=0,rmse_mm=None,bias_mm=None,mae_mm=None)
    return dict(windows=int(valid.sum()),rmse_mm=float(np.sqrt(np.mean(error**2))),
                bias_mm=float(error.mean()),mae_mm=float(np.abs(error).mean()))



def reliability(probability,observed):
    edges=np.array([0,.001,.003,.01,.03,.1,.3,.6,1.000001])
    groups=np.searchsorted(edges,probability,side='right')-1
    return [dict(points=int((groups==i).sum()),mean_probability=float(probability[groups==i].mean()),
                 observed_frequency=float(observed[groups==i].mean()))
            for i in range(len(edges)-1) if (groups==i).any()]


def plot_results(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    thresholds=report['configuration']['alert_rainfall_thresholds_mm']
    result=report['evaluations']['nearest_gauge_offset_+0h']['models']
    fig,axes=plt.subplots(3,len(thresholds),figsize=(13,11),constrained_layout=True)
    for j,t in enumerate(thresholds):
        for name,model in result.items():
            row=model[str(t)];curve=row['discrimination']['curve']
            ap=row['discrimination']['average_precision']
            axes[0,j].step([p['recall'] for p in curve],[p['precision'] for p in curve],where='pre',label=f'{name}: AP {ap:.3g}' if ap is not None else name)
            points=row['reliability']
            axes[1,j].plot([p['mean_probability'] for p in points],[p['observed_frequency'] for p in points],'.-',label=name)
        axes[0,j].set(title=f'> {t:g} mm/hour',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
        axes[0,j].legend(fontsize=8)
        axes[1,j].plot([0,1],[0,1],'k--',alpha=.4)
        axes[1,j].set(xlabel='Predicted probability',ylabel='Observed frequency',xlim=(0,1),ylim=(0,1))
        names=list(result)
        axes[2,j].bar(names,[result[n][str(t)]['policies']['validation_max_csi']['csi'] or 0 for n in names])
        axes[2,j].set(ylabel='CSI, frozen 2023 cutoffs')
        for i in range(3):axes[i,j].grid(alpha=.2)
    fig.suptitle('Independent Meteo-France gauges: 24 fixed dates in 2025\nNearest 0.1-degree cell; ERA5 forcing proxy; models and alert cutoffs frozen')
    fig.savefig(Path(out)/'gauge_comparison.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_rain_gauges.yaml')
    parser.add_argument('--cache',default='data/precipitation/gauge_validation')
    parser.add_argument('--out',default='runs/rain_gauges')
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    cache=Path(args.cache);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('gauge evaluation is already running') from None
    torch.set_num_threads(4)
    dates=cfg['splits']['test']
    times=pd.DatetimeIndex(np.concatenate([hour_ends(pd.Timestamp(d),pd.Timestamp(d)+pd.Timedelta(days=1)).values for d in dates]))
    if len(times.unique())!=len(times):raise ValueError('duplicate test hours')
    rows,stations,qc=quality_control(pd.read_parquet(cache/'observations.parquet'),times,cfg['region'],cfg['gauge_min_completeness'])
    if not len(stations):raise ValueError('no complete stations')
    cp=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    frozen_path=Path('runs/rain_alerts/frozen_rules.json');frozen=json.loads(frozen_path.read_text())
    if sha(cfg['checkpoint'])!=frozen['signature']['downscaler_sha256']:raise ValueError('downscaler changed')
    models={'raw':None,'gpd':TailCalibration.load('runs/rain_tail/selected.pt'),
            'exponential':TailCalibration.load('runs/rain_tail/conditional_exponential.pt')}
    files={'gpd':'runs/rain_tail/selected.pt','exponential':'runs/rain_tail/conditional_exponential.pt'}
    for name,path in files.items():
        if sha(path)!=frozen['signature']['calibrators'][name]:raise ValueError('calibrator changed')
    prior=yaml.safe_load(Path('config_rain_tail.yaml').read_text())['splits']
    if set(dates)&set(sum(prior.values(),[])+frozen['signature']['configuration']['splits']['test']):raise ValueError('previously inspected date')
    for split in ['training','validation']:
        a=np.datetime64(cp['data_attrs'][split+'_start']);b=np.datetime64(cp['data_attrs'][split+'_end'])
        if ((times.values>=a)&(times.values<=b)).any():raise ValueError('training overlap')
    signature=dict(config=cfg,downscaler=sha(cfg['checkpoint']),calibrators={n:sha(p) for n,p in files.items()},
                   rules=sha(frozen_path),observations=sha(cache/'observations.parquet'),stations=stations.index.tolist())
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=signature:raise ValueError('experiment inputs changed')
    atomic_json(path,signature);atomic_json(out/'quality_control.json',qc)
    grid=xr.Dataset(coords={k:cp[k].numpy() for k in ['lat','lon']})
    interps={m:GridInterpolator.build(grid.lat,grid.lon,stations.lat,stations.lon,m) for m in ['nearest','bilinear']}
    box=cfg['region'];source=None;thresholds=cfg['alert_rainfall_thresholds_mm']
    all_prob={m:{n:[] for n in models} for m in interps};all_mean={m:{n:[] for n in models} for m in interps}
    for index,date in enumerate(dates):
        destination=out/f'{date}.npz'
        if not destination.exists():
            if source is None:source=crop(open_store(ERA5_ARCO)[list(ERA5_NAMES.values())],(box[0]-1,box[1]+1,box[2]-1,box[3]+1))
            stamps=times[index*24:(index+1)*24]
            forcing=day_pack(source,stamps,None,grid,cfg)
            prediction=conditional_ensemble(cfg['checkpoint'],forcing,cfg['members'])
            r=prediction.precipitation.transpose('time','lat','lon','sample').values.reshape(-1,cfg['members'])
            features=forecast_features(forcing,prediction,cp['static'].numpy())
            data=SimpleNamespace(x=features,rain=r);arrays={}
            for name,model in models.items():
                prob=probabilities(data,model,thresholds)
                mean=r.mean(1).astype('float64')
                if model is not None:
                    for start in range(0,len(r),32768):
                        stop=start+32768;p,s,xi=model.parameters(features[start:stop],r[start:stop])
                        mean[start:stop]=spliced_moments(r[start:stop],model.threshold,p,s,xi)[0]
                for method,interp in interps.items():
                    field=xr.DataArray(prob.reshape(24,len(grid.lat),len(grid.lon),len(thresholds)),dims=['time','lat','lon','threshold'])
                    arrays[f'{method}_{name}_prob']=interp.apply(field).transpose('time','station','threshold').values
                    field=xr.DataArray(mean.reshape(24,len(grid.lat),len(grid.lon)),dims=['time','lat','lon'])
                    arrays[f'{method}_{name}_mean']=interp.apply(field).values
            with destination.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
            destination.with_suffix('.tmp').rename(destination)
        with np.load(destination) as stored:
            for method in interps:
                for name in models:
                    all_prob[method][name].append(stored[f'{method}_{name}_prob'])
                    all_mean[method][name].append(stored[f'{method}_{name}_mean'])
        atomic_json(out/'progress.json',dict(completed=index+1,total=len(dates),date=date))
        print(f'Gauge forecast {index+1}/{len(dates)}: {date}',flush=True)
    report=dict(configuration=cfg,quality_control=qc,reference='Meteo-France RR1, quality codes 0/1, decimal mm, metropolitan UTC',
                limitations=['ERA5 forcing proxy, not operational forecast skill','Point gauges and 0.1-degree cells have different spatial support',
                             'Primary timestamps assume preceding-hour rainfall; offsets are predeclared sensitivity checks',
                             'Calibrated marginal CDFs do not define coherent calibrated trajectories','Date bootstrap retains spatial dependence but dates need not be independent storms'],evaluations={})
    for method in interps:
        for offset in [0,-1,1]:
            obs=aligned_observations(rows,stations,times,offset);valid=np.isfinite(obs)
            day=np.broadcast_to(np.repeat(np.arange(len(dates)),24)[:,None],obs.shape)[valid]
            data=SimpleNamespace(truth=obs[valid],day=day,dates=dates)
            result={}
            for name in models:
                probability=np.concatenate(all_prob[method][name])[valid]
                result[name]=evaluate_alerts(data,probability,thresholds,frozen['rules'][name],cfg['matched_validation_false_alarm_rate'])
                for j,t in enumerate(thresholds):
                    observed=data.truth>t
                    result[name][str(t)]['reliability']=reliability(probability[:,j],observed)
                    result[name][str(t)]['mean_probability']=float(probability[:,j].mean())
                means=np.concatenate(all_mean[method][name])
                result[name]['amounts']={str(h):amount_scores(means,obs,h) for h in [1,6,24]}
            comparison={}
            for name in ['gpd','exponential']:
                comparison[name]={}
                for t in thresholds:
                    a=result['raw'][str(t)]['policies']['validation_max_csi']['per_day'];b=result[name][str(t)]['policies']['validation_max_csi']['per_day']
                    interval=paired_day_csi_interval(a,b)
                    interval['warning']='Exploratory date bootstrap; spatially dependent gauges and 24 selected dates'
                    comparison[name][str(t)]=interval
            report['evaluations'][f'{method}_gauge_offset_{offset:+d}h']=dict(station_hours=int(valid.sum()),models=result,paired_csi=comparison)
    atomic_json(out/'results.json',report)
    plot_results(report,out)
    atomic_json(out/'progress.json',dict(stage='complete',days=len(dates)))


if __name__=='__main__':main()
