"""Erbs arithmetic-mean event-template amplitude comparison."""
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
from scipy.special import gammaln
SEED=20261116;N_BOOT=20000
def hedges_correction(n0,n1):
    df=n0+n1-2
    return float(np.exp(gammaln(df/2)-.5*np.log(df/2)-gammaln((df-1)/2)))

def effects(a0,a1):
    n0,n1=len(a0),len(a1)
    pooled=np.sqrt(((n0-1)*np.var(a0,ddof=1)+(n1-1)*np.var(a1,ddof=1))/(n0+n1-2))
    difference=float(np.mean(a1)-np.mean(a0))
    log_difference=float(np.mean(np.log10(a1))-np.mean(np.log10(a0)))
    return np.array([10**log_difference,difference,hedges_correction(n0,n1)*difference/pooled,log_difference])

def explicit_effects(a0,a1):
    # Separate scalar moment route; independent of np.mean / np.var vector route.
    m0=sum(float(v) for v in a0)/len(a0)
    m1=sum(float(v) for v in a1)/len(a1)
    var0=sum((float(v)-m0)**2 for v in a0)/(len(a0)-1)
    var1=sum((float(v)-m1)**2 for v in a1)/(len(a1)-1)
    pooled=(( (len(a0)-1)*var0+(len(a1)-1)*var1)/(len(a0)+len(a1)-2))**.5
    ld=sum(float(np.log10(v)) for v in a1)/len(a1)-sum(float(np.log10(v)) for v in a0)/len(a0)
    return np.array([10**ld,m1-m0,hedges_correction(len(a0),len(a1))*(m1-m0)/pooled,ld])

def summary(values):
    return {'n':len(values), 'mean_uv':float(values.mean()), 'sd_uv':float(values.std(ddof=1)),
            'median_uv':float(np.median(values)), 'iqr_uv':np.quantile(values,[.25,.75]).tolist(),
            'geometric_mean_uv':float(10**np.log10(values).mean()), 'range_uv':[float(values.min()),float(values.max())]}

def population(amps,labels,indices,rng):
    values=amps[indices]; y=labels[indices]
    assert np.isfinite(values).all() and (values>0).all(), 'No silent deletion of invalid amplitude'
    a0,a1=values[y==0],values[y==1]
    assert len(a0)>1 and len(a1)>1
    point=effects(a0,a1)
    assert np.allclose(point,explicit_effects(a0,a1),rtol=1e-12,atol=1e-12)
    draws=np.empty((N_BOOT,4))
    max_error=0.
    for b in range(N_BOOT):
        r0=rng.integers(0,len(a0),size=len(a0));r1=rng.integers(0,len(a1),size=len(a1))
        b0,b1=a0[r0],a1[r1]
        draws[b]=effects(b0,b1)
        if b in (0,1,2,N_BOOT-3,N_BOOT-2,N_BOOT-1):
            explicit=explicit_effects(b0,b1)
            assert np.allclose(draws[b],explicit,rtol=1e-12,atol=1e-12)
            max_error=max(max_error,float(np.max(np.abs(draws[b]-explicit))))
    assert np.isfinite(draws).all()
    names=['geometric_mean_ratio_ADHD_over_control','raw_mean_difference_ADHD_minus_control_uv','raw_hedges_g_ADHD_minus_control','log10_mean_difference_ADHD_minus_control']
    return {'n':len(indices), 'groups':{'control':summary(a0),'ADHD':summary(a1)},
            'effects':{name:{'estimate':float(point[k]), 'percentile_95_ci':np.quantile(draws[:,k],[.025,.975]).tolist()} for k,name in enumerate(names)},
            'verification':{'point_scalar_moments':'PASS','bootstrap_explicit_replicates':[0,1,2,N_BOOT-3,N_BOOT-2,N_BOOT-1], 'bootstrap_max_abs_error':max_error, 'bootstrap_replicates':N_BOOT, 'all_draws_finite':True}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--indices',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 records=read(a.input)['records'];assert len(records)==189 and [r['index'] for r in records]==list(range(189))
 amps=np.array([r['morphology_descriptive']['template_peak_to_peak_uv'][0] if r['morphology_descriptive']['template_peak_to_peak_uv'][0] is not None else np.nan for r in records]);labels=np.load(a.labels);ids=np.asarray(read(a.indices),int)
 assert labels.shape==(189,) and set(labels)=={0,1};assert len(ids)==len(set(ids))==186;assert set(ids)=={r['index'] for r in records if r['eligible']}
 rng=np.random.default_rng(SEED);result={'primary_186':population(amps,labels,ids,rng)}
 if np.isfinite(amps).all() and (amps>0).all():result['overlapping_189']=population(amps,labels,np.arange(189),rng)
 else:result['overlapping_189']={'status':'not evaluated: nonfinite or nonpositive amplitude; no additional exclusions'}
 dump(a.output/'results.json',result)
if __name__=='__main__':main()
