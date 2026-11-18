"""Small categorical rainfall head for frozen forecast and terrain features.

This supplies marginal intensity probabilities, not joint rainfall trajectories.
Bins are right-closed so summing upper classes gives exact strict exceedances.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn

BOUNDS=np.array([0.,.1,.5,1.,2.,5.,10.,20.,40.],dtype='float32')


def labels(rain):
    rain=np.asarray(rain)
    if not np.isfinite(rain).all() or (rain<0).any():raise ValueError('invalid rainfall labels')
    return np.searchsorted(BOUNDS,rain,side='left')


def design(x,rain):
    return np.column_stack([x,*[(rain>t).mean(1) for t in [5,10,20]]]).astype('float32')


class CategoricalRain:
    def __init__(self,mean,std,rates,width=32):
        self.mean=np.asarray(mean,dtype='float32');self.std=np.asarray(std,dtype='float32')
        self.rates=np.asarray(rates,dtype='float32');self.width=width
        n=len(self.mean);k=len(BOUNDS)+1
        self.net=nn.Sequential(nn.Linear(n,width),nn.SiLU(),nn.Linear(width,k)) if width else nn.Sequential(nn.Linear(n,k))
        self.metadata={}

    def distribution(self,x,rain):
        z=(design(x,rain)-self.mean)/self.std
        if not np.isfinite(z).all():raise ValueError('nonfinite categorical features')
        with torch.no_grad():return self.net(torch.from_numpy(z)).softmax(-1).numpy()

    def probabilities(self,x,rain,thresholds):
        indexes=[]
        for t in thresholds:
            match=np.flatnonzero(BOUNDS==t)
            if not len(match):raise ValueError('threshold must be a trained bin boundary')
            indexes.append(int(match[0])+1)
        p=self.distribution(x,rain)
        return np.column_stack([p[:,i:].sum(1) for i in indexes])

    def rainfall_mean(self,x,rain):
        return self.distribution(x,rain)@self.rates

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(format='categorical-rain-v1',bounds=torch.from_numpy(BOUNDS),mean=torch.from_numpy(self.mean),
                        std=torch.from_numpy(self.std),rates=torch.from_numpy(self.rates),width=self.width,
                        state_dict=self.net.state_dict(),metadata=self.metadata),path)

    @classmethod
    def load(cls,path):
        cp=torch.load(path,map_location='cpu',weights_only=True)
        if cp['format']!='categorical-rain-v1' or not np.array_equal(cp['bounds'].numpy(),BOUNDS):raise ValueError('categorical format mismatch')
        model=cls(cp['mean'].numpy(),cp['std'].numpy(),cp['rates'].numpy(),cp['width'])
        model.net.load_state_dict(cp['state_dict']);model.net.eval();model.metadata=cp['metadata']
        return model
