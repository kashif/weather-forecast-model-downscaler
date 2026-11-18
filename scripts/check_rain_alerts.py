"""Freeze 2023 alert rules, then assess discrimination and alerts on fresh 2025 dates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from downscale.rain_tail import TailCalibration,spliced_survival
from downscale.rain_alerts import contingency,select_rules,precision_recall,paired_day_csi_interval
from downscale.rain_resume import atomic_json
from scripts.calibrate_rain_tail import load_split


def probabilities(data,model,thresholds):
    output=np.empty((len(data.rain),len(thresholds)),dtype='float64')
    for start in range(0,len(data.rain),32768):
        stop=start+32768;r=data.rain[start:stop]
        if model is not None and hasattr(model,'probabilities'):
            output[start:stop]=model.probabilities(data.x[start:stop],r,thresholds)
            continue
        if model is not None:p,s,xi=model.parameters(data.x[start:stop],r)
        for j,t in enumerate(thresholds):
            output[start:stop,j]=(r>t).mean(1) if model is None else spliced_survival(r,t,model.threshold,p,s,xi)
    return output


def evaluate_alerts(data,probability,thresholds,rules,budget):
    result={}
    for j,t in enumerate(thresholds):
        obs=data.truth>t;prob=probability[:,j]
        item=dict(discrimination=precision_recall(prob,obs,budget),brier=float(np.mean((prob-obs)**2)),policies={})
        for policy,cutoff in [('fixed_50_percent',.5),('validation_max_csi',rules[str(t)]['max_csi']),('validation_fpr_budget',rules[str(t)]['fpr_budget'])]:
            row=dict(cutoff=cutoff,**contingency(prob,obs,cutoff))
            row['per_day']=[dict(date=date,**contingency(prob[data.day==k],obs[data.day==k],cutoff)) for k,date in enumerate(data.dates)]
            item['policies'][policy]=row
        result[str(t)]=item
    return result


def plot_comparison(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    thresholds=report['configuration']['alert_rainfall_thresholds_mm']
    names=list(report['test']);fig,axes=plt.subplots(2,len(thresholds),figsize=(14,8),constrained_layout=True)
    for j,t in enumerate(thresholds):
        for name in names:
            row=report['test'][name][str(t)];curve=row['discrimination']['curve'];ap=row['discrimination']['average_precision']
            axes[0,j].step([v['recall'] for v in curve],[v['precision'] for v in curve],where='pre',label=f'{name} (AP {ap:.3g})' if ap is not None else name)
            point=row['policies']['validation_max_csi']
            if point['recall'] is not None and point['precision'] is not None:axes[0,j].scatter(point['recall'],point['precision'],s=35)
        peak=max(v['precision'] for n in names for v in report['test'][n][str(t)]['discrimination']['curve'])
        axes[0,j].set_ylim(0,min(1,max(.1,1.15*peak)))
        prevalence=report['test']['raw'][str(t)]['discrimination']['prevalence']
        axes[0,j].axhline(prevalence,color='k',ls='--',alpha=.3,label='Prevalence')
        axes[0,j].set(title=f'Rain > {t:g} mm/hour',xlabel='Recall',ylabel='Precision',xlim=(0,1))
        axes[0,j].legend(fontsize=7);axes[0,j].grid(alpha=.2)
        pos=np.arange(len(names))
        for offset,policy,label in [(-.18,'fixed_50_percent','50% cutoff'),(.18,'validation_max_csi','2023-selected cutoff')]:
            axes[1,j].bar(pos+offset,[report['test'][n][str(t)]['policies'][policy]['csi'] or 0 for n in names],width=.35,label=label)
        axes[1,j].set(xticks=pos,xticklabels=names,ylabel='CSI on fresh 2025 dates')
        axes[1,j].legend(fontsize=8);axes[1,j].grid(axis='y',alpha=.2)
    fig.suptitle('Fresh 2025 assessment against PARDIG: alert cutoffs frozen on 2023 dates\nPR dots show those frozen cutoffs; four test days, not independent radar verification',fontsize=12)
    fig.savefig(Path(out)/'alert_comparison.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_rain_alerts.yaml')
    parser.add_argument('--selection-cache',default='data/precipitation/gpd_calibration')
    parser.add_argument('--test-cache',default='data/precipitation/gpd_alerts_2025')
    parser.add_argument('--tail-run',default='runs/rain_tail')
    parser.add_argument('--out',default='runs/rain_alerts')
    parser.add_argument('--selection-only',action='store_true')
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());tail=Path(args.tail_run)
    torch.set_num_threads(4)
    previous=json.loads((tail/'selection.json').read_text())['config']
    if previous['checkpoint']!=cfg['checkpoint'] or previous['members']!=cfg['members']:raise ValueError('different frozen model/ensemble')
    if set(previous['splits']['selection']) & set(cfg['splits']['test']):raise ValueError('selection/test dates overlap')
    if max(previous['splits']['test'])>=min(cfg['splits']['test']):raise ValueError('fresh test must follow the inspected 2024 test')
    checkpoint=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    files={'gpd':tail/'selected.pt','exponential':tail/'conditional_exponential.pt'}
    models={'raw':None,**{name:TailCalibration.load(path) for name,path in files.items()}}
    sha=hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest()
    for model in models.values():
        if model is not None and model.metadata.get('checkpoint_sha256')!=sha:raise ValueError('calibration checkpoint mismatch')
    signature=dict(configuration=cfg,selection_dates=previous['splits']['selection'],downscaler_sha256=sha,
                   calibrators={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rules_path=out/'frozen_rules.json'
    if rules_path.exists():
        frozen=json.loads(rules_path.read_text())
        if frozen['signature']!=signature:raise ValueError('frozen rules differ; choose a new output directory')
    else:
        selection=load_split(args.selection_cache,'selection',previous,checkpoint)
        rules={};validation={}
        for name,model in models.items():
            prob=probabilities(selection,model,cfg['alert_rainfall_thresholds_mm'])
            rules[name]={str(t):select_rules(prob[:,j],selection.truth>t,cfg['probability_cutoffs'],cfg['matched_validation_false_alarm_rate'])
                         for j,t in enumerate(cfg['alert_rainfall_thresholds_mm'])}
            validation[name]=evaluate_alerts(selection,prob,cfg['alert_rainfall_thresholds_mm'],rules[name],cfg['matched_validation_false_alarm_rate'])
            print(json.dumps(dict(stage='validation_rules',model=name,cutoffs={t:v['max_csi'] for t,v in rules[name].items()})),flush=True)
        frozen=dict(signature=signature,rules=rules,validation=validation,written_before_loading_fresh_test=True)
        atomic_json(rules_path,frozen)
        del selection,prob
    if args.selection_only:return
    # This is intentionally after the persisted, fingerprinted rule decision.
    test=load_split(args.test_cache,'test',cfg,checkpoint)
    results={}
    for name,model in models.items():
        prob=probabilities(test,model,cfg['alert_rainfall_thresholds_mm'])
        results[name]=evaluate_alerts(test,prob,cfg['alert_rainfall_thresholds_mm'],frozen['rules'][name],cfg['matched_validation_false_alarm_rate'])
        print(json.dumps(dict(stage='fresh_test',model=name,metrics={t:dict(ap=v['discrimination']['average_precision'],csi=v['policies']['validation_max_csi']['csi']) for t,v in results[name].items()})),flush=True)
    bootstrap={name:{str(t):paired_day_csi_interval(results['raw'][str(t)]['policies']['validation_max_csi']['per_day'],results[name][str(t)]['policies']['validation_max_csi']['per_day'])
                     for t in cfg['alert_rainfall_thresholds_mm']} for name in ['gpd','exponential']}
    report=dict(configuration=cfg,reference='PARDIG proxy with ERA5 forcing; not independent radar/gauge or operational forecast verification',
                event_definition='Y > threshold; alerts when probability >= selected cutoff',
                fpr_definition='false positives / all observed non-events; distinct from false alarm ratio',
                rules_file_sha256=hashlib.sha256(rules_path.read_bytes()).hexdigest(),test=results,day_bootstrap=bootstrap,
                caveat='Seasonal fixed-date pilot: four fresh days; no tuning after inspecting 2025 labels')
    atomic_json(out/'evaluation.json',report);plot_comparison(report,out)


if __name__=='__main__':main()
