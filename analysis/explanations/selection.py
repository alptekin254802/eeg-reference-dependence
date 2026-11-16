"""Deterministic panel/background selection and attribution summaries."""
import numpy as np
from scipy.stats import spearmanr

def panel_and_background(labels,train,test,fold,large=False):
 y=np.asarray(labels);tr=np.asarray(train);te=np.asarray(test)
 rng=np.random.default_rng(3371+fold)
 panel=np.r_[rng.choice(te[y[te]==0],4,replace=False),rng.choice(te[y[te]==1],4,replace=False)]
 rng=np.random.default_rng((5441 if large else 4431)+fold);n=16 if large else 4
 # The larger background's group draw order follows the published implementation.
 groups=[1,0] if large else [0,1]
 people=np.concatenate([rng.choice(tr[y[tr]==k],n,replace=False) for k in groups]);windows=rng.integers(0,15,len(people))
 return panel,people,windows

def completeness_check(records):
 r=np.array([v['relative_residual'] for v in records]);return {'median':float(np.median(r)),'p95':float(np.percentile(r,95)),'accepted':bool(np.median(r)<=.05 and np.percentile(r,95)<=.20)}

def regional_rank_correlation(first,second,regions):
 a=[first[:,ix,:].sum() for ix in regions.values()];b=[second[:,ix,:].sum() for ix in regions.values()]
 return float(spearmanr(a,b).statistic)

def absolute_cosine(first,second):
 a=np.abs(first).ravel();b=np.abs(second).ravel();return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
