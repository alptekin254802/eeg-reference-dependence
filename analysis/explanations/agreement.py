"""Mean pairwise kappa, sharing diagnosis-stratified draws across every pair."""
import argparse,json,itertools
from pathlib import Path
import numpy as np
def metrics(a, b, draws=None, categories=5):
    aa, bb = (a[None, :], b[None, :]) if draws is None else (a[draws], b[draws])
    po = (aa == bb).mean(axis=1)
    ma = np.stack([(aa == k).mean(axis=1) for k in range(categories)], axis=1)
    mb = np.stack([(bb == k).mean(axis=1) for k in range(categories)], axis=1)
    pe = (ma * mb).sum(axis=1)
    kappa = np.divide(po-pe, 1-pe, out=np.full_like(po, np.nan), where=pe != 1)
    return po, pe, kappa

def summary(point, values):
    valid = np.isfinite(values)
    return {'estimate': float(point) if np.isfinite(point) else None,
            'ci95_percentile': np.quantile(values[valid], [.025, .975]).tolist() if valid.any() else None,
            'bootstrap_valid_n': int(valid.sum()),
            'bootstrap_undefined_n': int((~valid).sum())}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input');p.add_argument('--manifest');p.add_argument('--method',default='IG');p.add_argument('--target',choices=['predicted','adhd'],default='predicted');p.add_argument('--output',required=True);a=p.parse_args()
 if a.manifest:
  from summarize import read_records
  spec=json.loads(Path(a.manifest).read_text());arrays=[];ids=None;y=None
  for seed in ['1109','2209','3309']:
   refs=[]
   for ref in ['linked','average']:
    rows=read_records(spec[seed][ref],a.method);order=sorted(rows)
    if ids is None:ids=order;y=np.array([rows[i]['label'] for i in ids])
    assert ids==order
    refs.append([int(np.argmax(list(rows[i]['values']['roi_scores' if a.target=='predicted' else 'signed_roi_scores'].values()))) for i in ids])
   arrays.append(refs)
  x=np.array(arrays)
 else:
  z=np.load(a.input,allow_pickle=False);x=z['top_regions'];y=z['labels']
 assert x.shape==(3,2,len(y)) and set(y.tolist())=={0,1} and x.min()>=0 and x.max()<5
 rng=np.random.default_rng(20261116);draws=np.column_stack([rng.choice(np.flatnonzero(y==k),size=(20000,int((y==k).sum())),replace=True) for k in [1,0]])
 pairs=[(x[i,r],x[j,r]) for r in range(2) for i,j in itertools.combinations(range(3),2)]+[(x[s,0],x[s,1]) for s in range(3)]
 point=np.array([np.array(metrics(u,v))[:,0] for u,v in pairs]);boot=np.stack([np.stack(metrics(u,v,draws),axis=1) for u,v in pairs]);groups={'within_linked':list(range(3)),'within_average':list(range(3,6)),'within_pooled':list(range(6)),'cross_same_seed':list(range(6,9))}
 result={name:{m:summary(point[ix,j].mean(),boot[ix,:,j].mean(axis=0)) for j,m in enumerate(['raw_agreement','chance_agreement','kappa'])} for name,ix in groups.items()}
 result['pairs']={str(j):{m:summary(point[j,k],boot[j,:,k]) for k,m in enumerate(['raw_agreement','chance_agreement','kappa'])} for j in range(9)}
 result['within_pooled_minus_cross']={m:summary(point[:6,j].mean()-point[6:,j].mean(),boot[:6,:,j].mean(axis=0)-boot[6:,:,j].mean(axis=0)) for j,m in [(0,'raw_agreement'),(2,'kappa')]}
 Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
