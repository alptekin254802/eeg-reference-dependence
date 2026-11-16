"""Diagnosis-stratified conditional participant bootstrap."""
import numpy as np
from sklearn.metrics import roc_auc_score
class PersonBootstrap:
    def __init__(self,y,seed=20261110,reps=2000):
        self.y=np.asarray(y);self.neg=np.flatnonzero(self.y==0);self.pos=np.flatnonzero(self.y==1)
        assert len(self.neg)>1 and len(self.pos)>1
        rng=np.random.default_rng(seed)
        # Multinomial counts are equivalent to resampling participants with replacement.
        self.nc=rng.multinomial(len(self.neg),np.ones(len(self.neg))/len(self.neg),size=reps)
        self.pc=rng.multinomial(len(self.pos),np.ones(len(self.pos))/len(self.pos),size=reps)
        self.reps=reps

    def auc_samples(self,p):
        p=np.asarray(p);a=p[self.pos,None];b=p[self.neg][None]
        wins=(a>b).astype(float)+.5*(a==b)
        point=wins.mean();assert abs(point-roc_auc_score(self.y,p))<1e-12
        return ((self.pc@wins)*self.nc).sum(1)/(len(self.pos)*len(self.neg))

    def evaluate(self,p):
        p=np.asarray(p);boot=self.auc_samples(p);lo,hi=np.percentile(boot,[2.5,97.5])
        tp=(p[self.pos]>=.5).astype(float);tn=(p[self.neg]<.5).astype(float)
        ba_boot=.5*((self.pc@tp)/len(tp)+(self.nc@tn)/len(tn));blo,bhi=np.percentile(ba_boot,[2.5,97.5])
        q=np.clip(p,1e-7,1-1e-7)
        return {'auc':float(roc_auc_score(self.y,p)),'auc_ci95':[float(lo),float(hi)],
            'balanced_accuracy':float(.5*(tp.mean()+tn.mean())),'ba_ci95':[float(blo),float(bhi)],
            'log_loss':float(-(self.y*np.log(q)+(1-self.y)*np.log1p(-q)).mean()),
            'brier':float(np.mean((p-self.y)**2))}

    def mean_ci(self,x):
        x=np.asarray(x);samples=((self.pc@x[self.pos])+(self.nc@x[self.neg]))/len(x)
        lo,hi=np.percentile(samples,[2.5,97.5]);return {'mean':float(x.mean()),'ci95':[float(lo),float(hi)]}

    def auc_difference(self,a,b):
        difference=self.auc_samples(a)-self.auc_samples(b);lo,hi=np.percentile(difference,[2.5,97.5])
        return {'difference':float(roc_auc_score(self.y,a)-roc_auc_score(self.y,b)),'ci95':[float(lo),float(hi)]}
