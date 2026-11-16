"""Three-group archive-annotation power contrasts."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def clean(v):
 if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
 if isinstance(v,np.generic): return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v): return None
 return v
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+"\n",encoding="utf-8")
def estimate(x,g,B=20000):
 rng=np.random.default_rng(20261116);groups=[x[g==k] for k in range(3)]
 assert all(len(a)>1 for a in groups)
 counts=[rng.multinomial(len(a),np.full(len(a),1/len(a)),size=B) for a in groups]
 means=[c@a/len(a) for c,a in zip(counts,groups)]
 out=[]
 for a,b,name in [(1,0,'annotated/control'),(2,0,'unannotated/control'),(1,2,'annotated/unannotated')]:
  d=groups[a].mean(0)-groups[b].mean(0); bd=means[a]-means[b]
  for k,component in enumerate(['common','CAR']):
   ci=np.quantile(bd[:,k],[.025,.975]);out.append(dict(contrast=name,component=component,mean_log10_difference=float(d[k]),mean_log10_difference_ci95=ci.tolist(),geometric_power_ratio=float(10**d[k]),geometric_power_ratio_ci95=(10**ci).tolist()))
 for i in range(min(25,B)):
  explicit=[a[np.repeat(np.arange(len(a)),c[i])].mean(0) for a,c in zip(groups,counts)]
  assert np.allclose(explicit,np.array(means)[:,i],rtol=1e-12,atol=1e-12)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--groups',type=Path,required=True);p.add_argument('--indices',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 z=np.load(a.input);names=z['names'].tolist();x=z['features'][:,[names.index('common_total_1_40_log10_power'),names.index('car_total_1_40_log10_power')]];g=np.load(a.groups);g=np.where(z['labels']==0,0,np.where(g=='annotated',1,2)) if g.dtype.kind in 'US' else g;ids=np.asarray(read(a.indices),int)
 assert x.shape==(189,2) and g.shape==(189,) and set(g)=={0,1,2};assert len(ids)==len(set(ids))==186
 assert np.bincount(g[ids],minlength=3).tolist()==[131,39,16];assert np.bincount(g,minlength=3).tolist()==[133,40,16]
 dump(a.output/'results.json',{'primary_186':estimate(x[ids],g[ids]),'overlapping_189':estimate(x,g)})
if __name__=='__main__':main()
