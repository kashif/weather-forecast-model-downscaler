"""Conditional peaks-over-threshold calibration with a generalized Pareto tail.

Book reference: https://otexts.com/weird/univariate.html, section 2.10.
The GPD models reference rainfall excesses, not the surprisal scores of section
7.3. Dry/light rainfall retains the raw empirical body conditional on Y <= u.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TailData:
    x: np.ndarray
    rain: np.ndarray
    truth: np.ndarray
    day: np.ndarray
    dates: list[str]
    names: list[str]


class TailHead(nn.Module):
    """Separate exceedance odds and severity scale, with one shared tail shape.

    This pilot restricts xi to [0, .45]: exponential/heavy tails with finite
    variance. Negative-shape, bounded tails are not fitted in this experiment.
    """
    def __init__(self, features, shape_range=(0., .45), fixed_shape=None):
        super().__init__()
        if not 0 <= shape_range[0] < shape_range[1] < .5:
            raise ValueError('pilot shape bounds must satisfy 0 <= lower < upper < .5')
        self.odds=nn.Linear(features,1)
        self.severity=nn.Linear(features,1)
        self.shape_raw=nn.Parameter(torch.tensor(-1.25,dtype=torch.float64))
        if fixed_shape is not None and not 0 <= fixed_shape < .5:
            raise ValueError('fixed shape must have finite variance')
        self.fixed_shape=fixed_shape
        if fixed_shape is not None:self.shape_raw.requires_grad_(False)
        self.shape_range=tuple(shape_range)
        self.double()
        nn.init.zeros_(self.odds.weight);nn.init.zeros_(self.severity.weight)

    def forward(self,x):
        lo,hi=self.shape_range
        xi=lo+(hi-lo)*self.shape_raw.sigmoid() if self.fixed_shape is None else x.new_tensor(self.fixed_shape)
        return self.odds(x).squeeze(-1),F.softplus(self.severity(x).squeeze(-1))+1e-4,xi


def design(data, threshold):
    fraction=(data.rain>threshold).mean(1).astype('float32')
    return np.column_stack([data.x,fraction])


def fit_tail(data,threshold,settings):
    """Fit weighted occurrence likelihood and conditional GPD excess likelihood.

    All exceedances are retained; uniformly sampled non-exceedances are inverse-
    probability weighted. No labels from selection/test enter fitting or scaling.
    """
    x=design(data,threshold);y=np.asarray(data.truth,dtype='float64')
    positive=np.flatnonzero(y>threshold);negative=np.flatnonzero(y<=threshold)
    if len(positive)<100 or not len(negative):
        raise ValueError(f'insufficient fit support: {len(positive)} exceedances, {len(negative)} non-exceedances')
    rng=np.random.default_rng(settings.get('seed',17))
    keep=rng.choice(negative,min(len(negative),settings.get('max_negative_fit_points',200000)),replace=False)
    index=np.r_[positive,keep]
    weights=np.r_[np.ones(len(positive)),np.full(len(keep),len(negative)/len(keep))]
    weights/=weights.sum()
    mean=np.average(x[index],axis=0,weights=weights)
    std=np.maximum(np.sqrt(np.average((x[index]-mean)**2,axis=0,weights=weights)),1e-3)
    inputs=torch.from_numpy((x[index]-mean)/std)
    event=torch.from_numpy((y[index]>threshold).astype('float64'))
    excess=torch.from_numpy(y[positive]-threshold)
    weight=torch.from_numpy(weights)
    head=TailHead(x.shape[1],settings.get('shape_range',(0.,.45)),settings.get('fixed_shape'))
    p=len(positive)/len(y)
    with torch.no_grad():
        head.odds.bias.fill_(np.log(p/(1-p)))
        scale0=max(float(excess.mean())*.9,1e-3)
        head.severity.bias.fill_(np.log(np.expm1(scale0)))
    opt=torch.optim.LBFGS(head.parameters(),max_iter=settings.get('max_iterations',120),line_search_fn='strong_wolfe',tolerance_grad=1e-6)
    trace=[]
    def closure():
        opt.zero_grad()
        logits,scale,xi=head(inputs)
        occurrence=(F.binary_cross_entropy_with_logits(logits,event,reduction='none')*weight).sum()
        # torch 2.15 GPD.log_prob constructs a float32 zero internally;
        # use float32 for this likelihood, preserving gradients into the head.
        if head.fixed_shape==0:
            severity=-torch.distributions.Exponential(1/scale[:len(positive)]).log_prob(excess).mean()
        else:
            distribution=torch.distributions.GeneralizedPareto(torch.zeros_like(excess,dtype=torch.float32),scale[:len(positive)].float(),xi.float())
            severity=-distribution.log_prob(excess.float()).mean()
        penalty=settings.get('regularization',.001)*(head.odds.weight.square().sum()+head.severity.weight.square().sum())
        loss=occurrence+severity+penalty
        if not torch.isfinite(loss):raise ValueError('nonfinite GPD fitting loss')
        loss.backward();trace.append(float(loss.detach()))
        return loss
    opt.step(closure)
    head.eval()
    metadata=dict(threshold_mm=float(threshold),fit_points=len(y),exceedances=len(positive),
                  fit_days=len(data.dates),exceedance_days=int(np.unique(data.day[positive]).size),
                  likelihood_evaluations=len(trace),initial_loss=trace[0],final_loss=trace[-1],
                  shape=float(head(torch.zeros(1,x.shape[1],dtype=torch.float64))[2].detach()))
    return TailCalibration(head,float(threshold),mean,std,data.names+['raw_exceedance_fraction'],metadata)


class TailCalibration:
    def __init__(self,head,threshold,mean,std,names,metadata=None):
        self.head=head
        self.threshold=float(threshold)
        self.mean=np.asarray(mean)
        self.std=np.asarray(std)
        self.names=list(names)
        self.metadata=metadata or {}

    def parameters(self,x,rain):
        values=np.column_stack([x,(rain>self.threshold).mean(1)])
        if values.shape[1]!=len(self.mean):raise ValueError('calibration feature layout differs')
        with torch.no_grad():
            logits,scale,xi=self.head(torch.from_numpy(((values-self.mean)/self.std).astype('float64')))
        return logits.sigmoid().numpy(),scale.numpy(),float(xi)

    def save(self,path):
        torch.save(dict(format='rain-gpd-v1',threshold=self.threshold,mean=torch.from_numpy(self.mean),
                        std=torch.from_numpy(self.std),names=self.names,metadata=self.metadata,
                        shape_range=list(self.head.shape_range),fixed_shape=self.head.fixed_shape,state_dict=self.head.state_dict()),path)

    @classmethod
    def load(cls,path):
        cp=torch.load(path,map_location='cpu',weights_only=True)
        if cp['format']!='rain-gpd-v1':raise ValueError('unknown tail checkpoint format')
        head=TailHead(len(cp['names']),cp['shape_range'],cp.get('fixed_shape'));head.load_state_dict(cp['state_dict']);head.eval()
        return cls(head,cp['threshold'],cp['mean'].numpy(),cp['std'].numpy(),cp['names'],cp['metadata'])


def gpd_survival(excess,scale,xi):
    excess=np.maximum(excess,0)
    if abs(xi)<1e-8:return np.exp(-excess/scale)
    support=1+xi*excess/scale
    return np.where(support>0,np.exp(-np.log(np.maximum(support,1e-300))/xi),0)


def empirical_crps(rain,truth):
    sorted_rain=np.sort(rain,axis=1)
    m=rain.shape[1];coefficient=2*np.arange(m)-m+1
    return np.abs(rain-truth[:,None]).mean(1)-(sorted_rain*coefficient).sum(1)/(m*m)


def body_statistics(rain,truth,threshold):
    """Empirical body conditioned on <=u; fallback is a point mass at u."""
    ordered=np.sort(rain,axis=1).astype('float64')
    valid=ordered<=threshold
    count=valid.sum(1)
    empty=count==0
    ordered[empty,0]=threshold;valid[empty,0]=True;count=np.maximum(count,1)
    body=np.where(valid,ordered,0)
    mean=body.sum(1)/count
    absolute=(np.abs(ordered-truth[:,None])*valid).sum(1)/count
    coefficient=2*np.arange(rain.shape[1])[None,:]-count[:,None]+1
    pair=2*(body*coefficient).sum(1)/(count*count)
    second=(body*body).sum(1)/count
    return ordered,valid,count,mean,absolute,pair,second


def spliced_crps(rain,truth,threshold,p,scale,xi):
    """Exact CRPS of an empirical body plus a shifted GPD tail, xi<1."""
    _,_,_,body_mean,body_abs,body_pair,_=body_statistics(rain,truth,threshold)
    tail_mean=scale/(1-xi)
    z=np.maximum(truth-threshold,0)
    stop_loss=(scale+xi*z)/(1-xi)*gpd_survival(z,scale,xi)
    tail_abs=np.where(truth<=threshold,threshold+tail_mean-truth,z-tail_mean+2*stop_loss)
    tail_pair=2*scale/((1-xi)*(2-xi))
    pair=(1-p)**2*body_pair+2*p*(1-p)*(threshold+tail_mean-body_mean)+p*p*tail_pair
    return (1-p)*body_abs+p*tail_abs-.5*pair


def spliced_twcrps(truth,cutoff,threshold,p,scale,xi):
    """Threshold-weighted CRPS: score max(Y,cutoff), for cutoff >= u."""
    if cutoff<threshold:raise ValueError('tail score cutoff must be at least the GPD threshold')
    delta=cutoff-threshold
    pc=p*gpd_survival(delta,scale,xi)
    sc=scale+xi*delta
    body=np.full((len(truth),1),cutoff)
    return spliced_crps(body,np.maximum(truth,cutoff),cutoff,pc,sc,xi)


def spliced_survival(rain,amount,threshold,p,scale,xi):
    """P(Y>amount), with zero/light rainfall retained in the empirical body."""
    if amount>=threshold:return p*gpd_survival(amount-threshold,scale,xi)
    count=np.maximum((rain<=threshold).sum(1),1)
    body=((rain>amount)&(rain<=threshold)).sum(1)/count
    # All-raw-tail fallback puts the body mass exactly at u.
    body[(rain<=threshold).sum(1)==0]=1.
    return p+(1-p)*body


def spliced_moments(rain,threshold,p,scale,xi):
    _,_,_,bm,_,_,bsecond=body_statistics(rain,np.zeros(len(rain)),threshold)
    tm=threshold+scale/(1-xi)
    tv=scale*scale/((1-xi)**2*(1-2*xi))
    mean=(1-p)*bm+p*tm
    variance=(1-p)*bsecond+p*(tv+tm*tm)-mean*mean
    return mean,np.maximum(variance,0)
