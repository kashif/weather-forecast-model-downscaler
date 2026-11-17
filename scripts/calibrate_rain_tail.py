"""Fit/select/test a frozen-downscaler GPD tail calibrator on fixed dates.

Main references: OTexts That's Weird, sections 2.10 (POT) and 7.3 (distinct
surprisal application). Selection never inspects test labels. Scores use strict
Y>threshold events, hourly depth in mm, and uniform grid-cell weights.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
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
from scipy.ndimage import maximum_filter
from downscale.precipitation import check_pack
from downscale.rain_tail import TailData,TailCalibration,fit_tail,empirical_crps,spliced_crps,spliced_twcrps,spliced_survival,spliced_moments
from downscale.rain_resume import atomic_json

FEATURE_NAMES=['log_mean','log_std','log_max','log_neighbour_max_mean','log_coarse_rain',
               'dewpoint_depression_10K','wind_speed_10ms','pressure_anomaly_100hPa',
               'elevation_km','relief_500m','land_fraction','sin_year','cos_year',
               'sin_hour','cos_hour','latitude_scaled','longitude_scaled']


def forecast_features(pairs,prediction,static):
    check_pack(pairs,target=False)
    for name in ['time','lat','lon']:
        if not np.array_equal(pairs[name],prediction[name]):raise ValueError(f'{name} mismatch')
    if prediction.precipitation.dims!=('time','sample','lat','lon') or prediction.precipitation.attrs.get('units')!='mm':
        raise ValueError('expected hourly precipitation samples in mm')
    rain=prediction.precipitation.values
    if rain.shape[1]<2 or not np.isfinite(rain).all() or (rain<0).any():raise ValueError('invalid rainfall ensemble')
    mean=rain.mean(1)
    times=pd.DatetimeIndex(pairs.time.values)
    shape=mean.shape
    def broadcast(a):return np.broadcast_to(a,shape)
    raw=pairs.predictors
    arrays=[np.log1p(mean),np.log1p(rain.std(1,ddof=1)),np.log1p(rain.max(1)),
            np.log1p(maximum_filter(mean,size=(1,3,3),mode='nearest')),
            np.log1p(raw.sel(variable='tp').values),
            (raw.sel(variable='t2m').values-raw.sel(variable='d2m').values)/10,
            np.hypot(raw.sel(variable='u10m').values,raw.sel(variable='v10m').values)/10,
            (raw.sel(variable='msl').values-100000)/10000]
    arrays += [broadcast(static[i][None]) for i in [0,4,5]]
    year=(times.day_of_year.values-1+times.hour.values/24)/365.25
    hour=times.hour.values/24
    arrays += [broadcast(a[:,None,None]) for a in [np.sin(2*np.pi*year),np.cos(2*np.pi*year),np.sin(2*np.pi*hour),np.cos(2*np.pi*hour)]]
    arrays += [broadcast(pairs.lat.values[None,:,None]/90),broadcast(pairs.lon.values[None,None,:]/180)]
    x=np.column_stack([a.ravel() for a in arrays]).astype('float32')
    if not np.isfinite(x).all():raise ValueError('nonfinite forecast features')
    return x


def validate_splits(cfg,checkpoint):
    if set(cfg['splits'])!={'fit','selection','test'}:
        raise ValueError('expected fit, selection and test date lists')
    seen=set();bounds=[]
    for role in ['fit','selection','test']:
        days=pd.DatetimeIndex(cfg['splits'][role])
        if not len(days) or days.tz is not None or not days.equals(days.floor('D')):
            raise ValueError('splits need nonempty naive UTC dates at midnight')
        for day in days:
            if day in seen:raise ValueError('calibration/selection/test dates overlap')
            seen.add(day)
            start=day+pd.Timedelta(hours=1);end=day+pd.Timedelta(days=1)
            for split in ['training','validation']:
                attrs=checkpoint['data_attrs']
                if start<=pd.Timestamp(attrs[split+'_end']) and end>=pd.Timestamp(attrs[split+'_start']):
                    raise ValueError('tail dates overlap frozen-model training/validation')
        bounds.append((days.min(),days.max()+pd.Timedelta(days=1)))
    if not (bounds[0][1]<bounds[1][0] and bounds[1][1]<bounds[2][0]):
        raise ValueError('fit, selection and test periods must be chronological')


def choose_candidate(candidates,raw_crps,raw_tail,ratio):
    eligible=[i for i,c in enumerate(candidates)
              if c['selection_crps_mm']<=ratio*raw_crps and c['selection_tail_objective']<raw_tail]
    return min(eligible,key=lambda i:candidates[i]['selection_tail_objective']) if eligible else None


def load_split(cache,role,cfg,checkpoint):
    xs=[];ys=[];rs=[];ids=[]
    sha=hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest()
    for day_id,date in enumerate(cfg['splits'][role]):
        root=Path(cache)/role/pd.Timestamp(date).strftime('%Y%m%d')
        pairs=xr.open_zarr(root/'pairs.zarr').load()
        prediction=xr.open_zarr(root/'predictions.zarr').load()
        if prediction.attrs.get('checkpoint_sha256')!=sha:raise ValueError('frozen checkpoint differs')
        check_pack(pairs)
        expected=pd.date_range(pd.Timestamp(date)+pd.Timedelta(hours=1),periods=24,freq='h')
        if not np.array_equal(pairs.time.values,expected.values):raise ValueError('cached day timestamps differ')
        if prediction.sizes['sample']!=cfg['members']:raise ValueError('ensemble size differs')
        xs.append(forecast_features(pairs,prediction,checkpoint['static'].numpy()))
        ys.append(pairs.target.values.ravel())
        rs.append(prediction.precipitation.transpose('time','lat','lon','sample').values.reshape(-1,cfg['members']))
        ids.append(np.full(len(ys[-1]),day_id,dtype='int16'))
    return TailData(np.concatenate(xs),np.concatenate(rs),np.concatenate(ys),np.concatenate(ids),cfg['splits'][role],FEATURE_NAMES)


def evaluate(data,calibration=None):
    sums=defaultdict(float);days=[defaultdict(float) for _ in data.dates]
    total=0;reliability={str(t):np.zeros((10,3)) for t in [.1,1.,5.,10.,20.]}
    for start in range(0,len(data.truth),32768):
        stop=start+32768;r=data.rain[start:stop];y=data.truth[start:stop];day=data.day[start:stop]
        if calibration is None:
            mean=r.mean(1)
            values={'crps_mm':empirical_crps(r,y)}
            for cutoff in [5.,10.]:values[f'twcrps_{cutoff:g}mm']=empirical_crps(np.maximum(r,cutoff),np.maximum(y,cutoff))
        else:
            p,s,xi=calibration.parameters(data.x[start:stop],r);u=calibration.threshold
            mean,_=spliced_moments(r,u,p,s,xi)
            values={'crps_mm':spliced_crps(r,y,u,p,s,xi)}
            for cutoff in [5.,10.]:values[f'twcrps_{cutoff:g}mm']=spliced_twcrps(y,cutoff,u,p,s,xi)
        values.update(squared_error=(mean-y)**2,bias_mm=mean-y,predicted_mean_mm=mean,observed_mean_mm=y)
        for threshold in [.1,1.,5.,10.,20.]:
            obs=y>threshold
            prob=(r>threshold).mean(1) if calibration is None else spliced_survival(r,threshold,u,p,s,xi)
            if not np.isfinite(prob).all() or (prob<0).any() or (prob>1+1e-10).any():raise ValueError('invalid calibrated probabilities')
            event=prob>=.5
            values.update({f'brier_{threshold:g}mm':(prob-obs)**2,
                           f'hits_{threshold:g}mm':event&obs,f'misses_{threshold:g}mm':~event&obs,
                           f'false_alarms_{threshold:g}mm':event&~obs,f'observed_events_{threshold:g}mm':obs,
                           f'mean_probability_{threshold:g}mm':prob})
            bins=np.minimum((prob*10).astype(int),9)
            for k in range(10):
                chosen=bins==k
                reliability[str(threshold)][k]+= [chosen.sum(),prob[chosen].sum(),obs[chosen].sum()]
        for key,array in values.items():
            if not np.isfinite(array).all():raise ValueError(f'nonfinite {key}')
            sums[key]+=float(array.sum())
            for d in np.unique(day):days[d][key]+=float(array[day==d].sum())
        total+=len(y)
        for d in np.unique(day):days[d]['n']+=int((day==d).sum())
    def finish(totals,n):
        result={k:float(v/n) for k,v in totals.items() if k!='n'}
        result['rmse_mm']=float(np.sqrt(result.pop('squared_error')))
        for threshold in [.1,1.,5.,10.,20.]:
            hit=totals[f'hits_{threshold:g}mm'];miss=totals[f'misses_{threshold:g}mm'];false=totals[f'false_alarms_{threshold:g}mm']
            result[f'csi_{threshold:g}mm']=hit/(hit+miss+false) if hit+miss+false else None
            result[f'precision_{threshold:g}mm']=hit/(hit+false) if hit+false else None
            result[f'recall_{threshold:g}mm']=hit/(hit+miss) if hit+miss else None
            for name in ['hits','misses','false_alarms','observed_events']:result[f'{name}_{threshold:g}mm']=int(round(totals[f'{name}_{threshold:g}mm']))
        result['tail_objective']=result['twcrps_5mm']+result['twcrps_10mm']
        return result
    result=finish(sums,total)
    result['per_day']=[dict(date=date,**finish(item,item['n'])) for date,item in zip(data.dates,days)]
    result['points']=total
    result['reliability']={key:[dict(n=int(n),probability=float(p/n) if n else None,frequency=float(y/n) if n else None)
                                for n,p,y in bins] for key,bins in reliability.items()}
    return result


def paired_day_bootstrap(raw,calibrated,seed):
    rng=np.random.default_rng(seed);result={}
    for name in ['crps_mm','tail_objective','brier_5mm','brier_10mm']:
        delta=np.array([b[name]-a[name] for a,b in zip(raw['per_day'],calibrated['per_day'])])
        draws=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(1)
        result[name]=dict(calibrated_minus_raw=float(delta.mean()),day_bootstrap_95_interval=np.quantile(draws,[.025,.975]).tolist())
    return dict(days=len(raw['per_day']),warning='exploratory day bootstrap; four test days cannot establish general heavy-rain skill',scores=result)


def excess_pit(data,calibration):
    from downscale.rain_tail import gpd_survival
    values=[]
    for start in range(0,len(data.truth),32768):
        stop=start+32768;chosen=data.truth[start:stop]>calibration.threshold
        if not chosen.any():continue
        p,s,xi=calibration.parameters(data.x[start:stop],data.rain[start:stop])
        values.append(1-gpd_survival(data.truth[start:stop][chosen]-calibration.threshold,s[chosen],xi))
    pit=np.concatenate(values) if values else np.array([])
    counts,edges=np.histogram(pit,bins=np.linspace(0,1,11))
    return dict(n=len(pit),mean=float(pit.mean()) if len(pit) else None,counts=counts.tolist(),edges=edges.tolist(),
                note='Conditional excess CDF values on test exceedances; correlated pixels, no iid significance test')


def plot_report(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    axes=axes.ravel()
    for key,label in [('raw_test','Raw ensemble'),('calibrated_test','Selected calibration')]:
        r=report[key]
        for threshold,ax in [(5.,axes[0]),(10.,axes[1])]:
            rows=[v for v in r['reliability'][str(threshold)] if v['n']]
            ax.plot([v['probability'] for v in rows],[v['frequency'] for v in rows],'o-',label=label)
            ax.set(title=f'P(rain > {threshold:g} mm/hour)',xlabel='Forecast probability',ylabel='Observed frequency')
            ax.plot([0,1],[0,1],'k--',alpha=.3)
    for ax in axes[:2]:ax.legend(fontsize=8);ax.grid(alpha=.2)
    candidates=report['selection']['candidates']
    axes[2].plot([c['threshold_mm'] for c in candidates],[c['shape'] for c in candidates],'o-')
    axes[2].set(title='Threshold sensitivity (fit data)',xlabel='GPD threshold u (mm)',ylabel='Fitted shared shape ξ')
    axes[2].grid(alpha=.2)
    pit=report.get('excess_pit')
    if pit and pit['n']:
        axes[3].bar(np.arange(10)/10+.05,np.array(pit['counts'])/pit['n']/.1,width=.09,color='C1',alpha=.7)
        axes[3].axhline(1,color='k',linestyle='--',alpha=.4)
        axes[3].set(title='Conditional tail intensity check',xlabel='GPD CDF of observed excess',ylabel='Density (uniform reference = 1)')
    else:axes[3].text(.1,.5,'No selected tail model',transform=axes[3].transAxes)
    fig.suptitle('Exploratory GPD calibration against PARDIG; frozen downscaler and predeclared dates')
    fig.savefig(Path(out)/'tail_diagnostics.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_rain_tail.yaml')
    parser.add_argument('--cache',default='data/precipitation/gpd_calibration')
    parser.add_argument('--out',default='runs/rain_tail')
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cp=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    validate_splits(cfg,cp)
    manifest=json.loads((Path(args.cache)/'manifest.json').read_text())
    if manifest['checkpoint_sha256']!=hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest() or manifest['config']['splits']!=cfg['splits']:
        raise ValueError('prepared data/checkpoint split manifest differs')
    fit=load_split(args.cache,'fit',cfg,cp);selection=load_split(args.cache,'selection',cfg,cp)
    raw_selection=evaluate(selection)
    print(json.dumps(dict(stage='raw_selection',crps=raw_selection['crps_mm'],tail_objective=raw_selection['tail_objective'])),flush=True)
    candidates=[];models=[]
    for threshold in cfg['thresholds_mm']:
        cal=fit_tail(fit,threshold,cfg)
        cal.metadata.update(checkpoint_sha256=hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest(),feature_names=FEATURE_NAMES,
                            fit_dates=cfg['splits']['fit'],reference='PARDIG proxy; not independent radar/gauge observations')
        cal.save(out/f'gpd_u{threshold:g}.pt')
        metrics=evaluate(selection,cal)
        eligible=metrics['crps_mm']<=cfg['selection']['max_overall_crps_ratio']*raw_selection['crps_mm']
        record=dict(**cal.metadata,selection_crps_mm=metrics['crps_mm'],selection_tail_objective=metrics['tail_objective'],eligible=bool(eligible))
        candidates.append(record);models.append(cal)
        atomic_json(out/'selection_progress.json',dict(candidates=candidates));print(json.dumps(record),flush=True)
    chosen=choose_candidate(candidates,raw_selection['crps_mm'],raw_selection['tail_objective'],cfg['selection']['max_overall_crps_ratio'])
    decision=dict(config=cfg,raw_crps_mm=raw_selection['crps_mm'],raw_tail_objective=raw_selection['tail_objective'],
                  candidates=candidates,selected_threshold_mm=None if chosen is None else cfg['thresholds_mm'][chosen],
                  selected='raw' if chosen is None else 'gpd',decision_made_before_loading_test=True)
    atomic_json(out/'selection.json',decision)
    selected=None if chosen is None else models[chosen]
    if selected:selected.save(out/'selected.pt')
    elif (out/'selected.pt').exists():(out/'selected.pt').unlink()
    ablations={}
    if selected:
        # Explanatory comparisons at the already selected u. They do not alter
        # the selection rule or replace the chosen model after inspecting test.
        u=selected.threshold
        exponential=fit_tail(fit,u,{**cfg,'fixed_shape':0.})
        pooled=TailData(np.broadcast_to(np.float32(0),fit.x.shape),np.broadcast_to(np.float32(0),fit.rain.shape),fit.truth,fit.day,fit.dates,fit.names)
        climatology=fit_tail(pooled,u,cfg)
        for name,model in [('conditional_exponential',exponential),('pooled_gpd',climatology)]:
            model.metadata.update(checkpoint_sha256=hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest(),fit_dates=cfg['splits']['fit'])
            model.save(out/(name+'.pt'))
            ablations[name]=model
            print(json.dumps(dict(stage='ablation_fit',name=name,shape=model.metadata['shape'])),flush=True)
    del fit,selection
    test=load_split(args.cache,'test',cfg,cp)
    raw_test=evaluate(test);calibrated=evaluate(test,selected) if selected else raw_test
    report=dict(reference='PARDIG proxy; not independent radar/gauge validation',event_definition='Y > threshold (strict), hourly depth in mm',
                book='https://otexts.com/weird/univariate.html (section 2.10); section 7.3 is a separate surprisal use',
                selection=decision,raw_test=raw_test,calibrated_test=calibrated,
                uncertainty=paired_day_bootstrap(raw_test,calibrated,cfg['seed']),
                excess_pit=excess_pit(test,selected) if selected else None,
                ablation_note='Explanatory fits at the selected threshold; not used to select the deployed model',
                ablations={name:evaluate(test,model) for name,model in ablations.items()})
    atomic_json(out/'evaluation.json',report);plot_report(report,out)
    print(json.dumps(dict(selected=decision['selected'],threshold=decision['selected_threshold_mm'],raw_test_crps=raw_test['crps_mm'],
                          calibrated_test_crps=calibrated['crps_mm'],raw_test_tail=raw_test['tail_objective'],calibrated_test_tail=calibrated['tail_objective'])),flush=True)


if __name__=='__main__':main()
