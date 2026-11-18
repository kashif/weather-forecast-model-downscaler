"""Fit a cheap categorical rainfall head on cached 2021 forecasts; select on 2023."""
from __future__ import annotations
import argparse
import fcntl
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
import yaml
from downscale.rain_categorical import CategoricalRain,BOUNDS,labels,design
from downscale.rain_resume import atomic_json
from downscale.rain_alerts import select_rules
from downscale.rain_tail import TailCalibration
from scripts.calibrate_rain_tail import load_split,validate_splits
from scripts.check_rain_alerts import probabilities,evaluate_alerts


def sampling_indices(y,limit=200000,seed=17):
    """Retain every >5 mm event; sample other rows with population weights."""
    rng=np.random.default_rng(seed);positive=np.flatnonzero(y>5);negative=np.flatnonzero(y<=5)
    chosen=rng.choice(negative,min(limit,len(negative)),replace=False)
    index=np.r_[positive,chosen]
    weight=np.r_[np.ones(len(positive)),np.full(len(chosen),len(negative)/max(len(chosen),1))].astype('float32')
    return index,weight/weight.mean()


def predict_probabilities(data,model,thresholds):
    result=np.empty((len(data.rain),len(thresholds)),dtype='float64')
    for start in range(0,len(result),32768):
        stop=start+32768
        result[start:stop]=model.probabilities(data.x[start:stop],data.rain[start:stop],thresholds)
    return result


def selection_score(prob,truth,thresholds):
    scores=[np.mean((prob[:,j]-(truth>t))**2)/max(np.mean(truth>t),1e-8) for j,t in enumerate(thresholds)]
    return float(np.mean(scores[:2]))  # Equal prevalence-normalized Brier at >5 and >10.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='runs/rain_categorical')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('categorical fitting is already running') from None
    if (out/'selected.pt').exists():raise ValueError('selected head exists; use a new run directory')
    torch.set_num_threads(4)
    cfg=yaml.safe_load(Path('config_rain_tail.yaml').read_text())
    cp=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    validate_splits(cfg,cp)
    sha=hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest()
    settings=dict(widths=[0,32],epochs=12,assess_epochs=[4,8,12],learning_rate=.001,batch=8192,
                  seed=17,max_nonheavy=200000,weight_decay=.001,selection='mean prevalence-normalized Brier at 5/10mm',
                  fit_dates=cfg['splits']['fit'],selection_dates=cfg['splits']['selection'],downscaler_sha256=sha)
    atomic_json(out/'protocol.json',settings)
    fit=load_split('data/precipitation/gpd_calibration','fit',cfg,cp)
    selection=load_split('data/precipitation/gpd_calibration','selection',cfg,cp)
    z=design(fit.x,fit.rain);mean=z.mean(0);std=z.std(0).clip(.01)
    classes=labels(fit.truth);counts=np.bincount(classes,minlength=len(BOUNDS)+1)
    defaults=np.r_[BOUNDS,60.].copy()
    rates=np.array([fit.truth[classes==i].mean() if counts[i] else defaults[i] for i in range(len(counts))],dtype='float32')
    indexes,weight=sampling_indices(fit.truth,settings['max_nonheavy'],settings['seed'])
    x=torch.from_numpy(((z[indexes]-mean)/std).astype('float32'));y=torch.from_numpy(classes[indexes]);w=torch.from_numpy(weight)
    thresholds=[5.,10.,20.];prior=torch.from_numpy(np.log((counts+.5)/(counts.sum()+.5*len(counts))).astype('float32'))
    baseline={n:selection_score(probabilities(selection,m,thresholds),selection.truth,thresholds) for n,m in
              [('raw',None),('exponential',TailCalibration.load('runs/rain_tail/conditional_exponential.pt'))]}
    records=[];best=float('inf');chosen=None
    for width in settings['widths']:
        torch.manual_seed(settings['seed']);rng=np.random.default_rng(settings['seed'])
        model=CategoricalRain(mean,std,rates,width)
        with torch.no_grad():model.net[-1].bias.copy_(prior)
        opt=torch.optim.AdamW(model.net.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        for epoch in range(1,settings['epochs']+1):
            model.net.train();order=rng.permutation(len(x));losses=[]
            for start in range(0,len(order),settings['batch']):
                index=order[start:start+settings['batch']];opt.zero_grad()
                loss=(F.cross_entropy(model.net(x[index]),y[index],reduction='none')*w[index]).mean()
                loss.backward();opt.step();losses.append(float(loss.detach()))
            if epoch in settings['assess_epochs']:
                model.net.eval();prob=predict_probabilities(selection,model,thresholds)
                score=selection_score(prob,selection.truth,thresholds)
                row=dict(width=width,epoch=epoch,training_loss=float(np.mean(losses)),selection_score=score)
                records.append(row);print(json.dumps(row),flush=True)
                if score<best:best=score;chosen=copy.deepcopy(model);chosen.metadata=dict(settings,selected_epoch=epoch,selection_score=score,feature_names=fit.names)
            atomic_json(out/'progress.json',dict(width=width,epoch=epoch,records=records,baselines=baseline))
    chosen.save(out/'selected.pt');prob=predict_probabilities(selection,chosen,thresholds)
    alert_cfg=yaml.safe_load(Path('config_rain_alerts.yaml').read_text())
    rules={str(t):select_rules(prob[:,j],selection.truth>t,alert_cfg['probability_cutoffs'],alert_cfg['matched_validation_false_alarm_rate']) for j,t in enumerate(thresholds)}
    atomic_json(out/'frozen_rules.json',dict(model_sha256=hashlib.sha256((out/'selected.pt').read_bytes()).hexdigest(),
                downscaler_sha256=sha,selection_dates=cfg['splits']['selection'],rules=rules))
    validation=evaluate_alerts(selection,prob,thresholds,rules,alert_cfg['matched_validation_false_alarm_rate'])
    atomic_json(out/'selection.json',dict(protocol=settings,records=records,baselines=baseline,selected=chosen.metadata,validation=validation,
                fit_rows=len(fit.truth),sampled_rows=len(indexes),class_counts=counts.tolist(),class_means_mm=rates.tolist()))
    print(json.dumps(dict(stage='complete',best=best,baselines=baseline)),flush=True)


if __name__=='__main__':main()
