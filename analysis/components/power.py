"""Common/CAR power and diagnosis contrasts from prepared microvolt tensors."""
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
from scipy.signal import welch
from scipy.special import gammaln
BANDS = [('delta',1,4),('theta',4,8),('alpha',8,13),('beta',13,30),('gamma',30,40.01),('total_1_40',1,40.01)]
def power(x):
    f,p=welch(x,fs=200,nperseg=400,axis=-1)
    return np.stack([p[...,(f>=lo)&(f<hi)].sum(-1)*.5 for _,lo,hi in BANDS],axis=-1)
def extract():
    linked=np.load(SOURCE/'linked_basic.npy',mmap_mode='r')
    car=np.load(SOURCE/'average_basic.npy',mmap_mode='r')
    assert linked.shape==car.shape==(189,15,19,800)
    assert linked.dtype==car.dtype==np.float32
    common=np.asarray(linked).mean(2)
    pc=power(common)  # participant, window, band
    cr=np.empty((189,15,6),float); pl=np.empty_like(cr)
    var_ratio=np.empty(189); fraction=np.empty(189)
    max_wave_normalized=0.; max_power_relative=0.; max_var_relative=0.
    rms_reconstructed=[]
    for i in range(189):
        x=np.asarray(linked[i]); r=np.asarray(car[i]); m=common[i]
        reconstruction=x-m[:,None,:]
        scale=max(float(np.abs(x).max()),1.)
        max_wave_normalized=max(max_wave_normalized,float(np.abs(r-reconstruction).max())/scale)
        cr[i]=power(r).mean(1)
        pl[i]=power(x).mean(1)
        max_power_relative=max(max_power_relative,float(np.max(np.abs(pl[i]-pc[i]-cr[i])/pl[i])))
        xv=x.astype(float).transpose(1,0,2).reshape(19,-1)
        rv=r.astype(float).transpose(1,0,2).reshape(19,-1)
        cv=m.astype(float).reshape(-1).var()
        vr=rv.var(1).mean(); vl=xv.var(1).mean()
        var_ratio[i]=cv/vr
        max_var_relative=max(max_var_relative,abs(vl-cv-vr)/vl)
        rms_reconstructed.append(np.median(xv.std(1)))
        fraction[i]=pc[i,:,-1].mean()/pl[i,:,-1].mean()
    eps=np.finfo(np.float32).eps
    assert max_wave_normalized<64*eps
    assert max_power_relative<128*eps
    assert max_var_relative<128*eps
    assert np.all((fraction>=0)&(fraction<=1+128*eps))
    lc=np.log10(np.maximum(pc,1e-12)).mean(1)
    lr=np.log10(np.maximum(cr,1e-12)).mean(1)
    q=np.load(SOURCE/'quality_features.npy'); manifest=read(SOURCE/'manifest.json')
    rms=q[:,manifest['quality_features'].index('median_channel_rms_uv')]
    assert np.allclose(rms,rms_reconstructed,rtol=2e-6,atol=1e-6)
    # Independent explicit Welch: three overlapping, individually detrended Hann segments.
    z=linked[7,3].astype(float); hann=.5-.5*np.cos(2*np.pi*np.arange(400)/400)
    seg=np.stack([z[:,s:s+400] for s in (0,200,400)],axis=1)
    seg-=seg.mean(-1,keepdims=True)
    spec=np.abs(np.fft.rfft(seg*hann,axis=-1))**2/(200*(hann@hann))
    spec[...,1:-1]*=2
    manual=spec.mean(1); f=np.arange(201)*.5
    independent=np.stack([manual[:,(f>=lo)&(f<hi)].sum(-1)*.5 for _,lo,hi in BANDS],axis=-1)
    expected=power(z); assert np.allclose(independent,expected,rtol=1e-12,atol=1e-12)
    names=['common_'+b[0]+'_log10_power' for b in BANDS]+['car_'+b[0]+'_log10_power' for b in BANDS]+['median_channel_rms_uv','median_channel_rms_log10_uv']
    features=np.column_stack([lc,lr,rms,np.log10(rms)])
    assert np.isfinite(features).all()
    return features,names,{'common_to_car_broadband_variance_ratio':var_ratio,'common_fraction_mean_linked_1_40_power':fraction}, {
        'max_waveform_identity_abs_error_div_max_linked':max_wave_normalized,
        'max_spectral_mean_power_identity_relative_error':max_power_relative,
        'max_variance_identity_relative_error':max_var_relative,
        'rms_max_relative_error':np.max(np.abs(np.array(rms_reconstructed)-rms)/rms),
        'explicit_hann_fft_welch_max_abs_error':np.max(np.abs(independent-expected)),
        'cached_car':'Computed in original float64 preparation before separate float32 storage; linked-minus-float32-spatial-mean differs only by rounding',
        'units':'Cached signals microvolt; Welch density microvolt^2/Hz; integrated power microvolt^2; RMS microvolt'}
def effect(x,y):
    a=x[y==1]; b=x[y==0]; df=len(a)+len(b)-2
    j=np.exp(gammaln(df/2)-.5*np.log(df/2)-gammaln((df-1)/2))
    delta=a.mean(0)-b.mean(0)
    pooled=np.sqrt(((len(a)-1)*a.var(0,ddof=1)+(len(b)-1)*b.var(0,ddof=1))/df)
    return delta,j*delta/pooled,j

def cohort(ids,features,names,descriptive,y):
    x=features[ids]; labels=y[ids]; groups=[x[labels==k] for k in (0,1)]
    rng=np.random.default_rng(20261116)
    counts=[rng.multinomial(len(a),np.full(len(a),1/len(a)),size=20000) for a in groups]
    means=[c@a/len(a) for c,a in zip(counts,groups)]
    variances=[(c@(a*a)-len(a)*m*m)/(len(a)-1) for c,a,m in zip(counts,groups,means)]
    delta,g,j=effect(x,labels); df=len(x)-2
    bd=means[1]-means[0]
    bg=j*bd/np.sqrt(((len(groups[0])-1)*variances[0]+(len(groups[1])-1)*variances[1])/df)
    # Explicit resampling uses the same multinomial participant multiplicities.
    for k in range(25):
        sample=np.concatenate([a[np.repeat(np.arange(len(a)),c[k])] for a,c in zip(groups,counts)])
        sy=np.r_[np.zeros(len(groups[0]),int),np.ones(len(groups[1]),int)]
        dd,gg,_=effect(sample,sy)
        assert np.allclose(dd,bd[k],rtol=1e-11,atol=1e-11)
        assert np.allclose(gg,bg[k],rtol=1e-10,atol=1e-10)
    a,b=groups[1],groups[0]
    s=np.sqrt((np.sum((a-a.mean(0))**2,axis=0)+np.sum((b-b.mean(0))**2,axis=0))/df)
    assert np.allclose(g,j*(a.mean(0)-b.mean(0))/s,rtol=1e-12,atol=1e-12)
    output={}
    for k,name in enumerate(names):
        ci=np.quantile(bd[:,k],[.025,.975]); gc=np.quantile(bg[:,k],[.025,.975])
        row={'control':{'mean':b[:,k].mean(),'sd':b[:,k].std(ddof=1)},'adhd':{'mean':a[:,k].mean(),'sd':a[:,k].std(ddof=1)},
             'mean_difference':delta[k],'mean_difference_ci95':ci,'hedges_g':g[k],'hedges_g_ci95':gc}
        if 'log10' in name:
            row.update({'geometric_ratio':10**delta[k],'geometric_ratio_ci95':10**ci,
                        'ratio_kind':'amplitude' if 'rms' in name else 'power'})
        output[name]=row
    desc={}
    for name,values in descriptive.items():
        v=values[ids]
        desc[name]={group:{'n':int(mask.sum()),'mean':v[mask].mean(),'median':np.median(v[mask]),'q25':np.quantile(v[mask],.25),'q75':np.quantile(v[mask],.75),'min':v[mask].min(),'max':v[mask].max()}
          for group,mask in [('all',np.ones(len(ids),bool)),('control',labels==0),('adhd',labels==1)]}
    return {'n':len(ids),'control_n':len(b),'adhd_n':len(a),'hedges_J':j,'features':output,'descriptive':desc,
            'checks':'25 bootstrap replicates independently expanded to repeated participant rows and recomputed; Hedges pooled SD independently via centered sums of squares'}
def main():
 global SOURCE
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--indices',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 # The input directory contains prepared arrays in their original row order.
 SOURCE=a.input; features,names,desc,checks=extract();y=np.load(SOURCE/'labels.npy');ids=np.asarray(read(a.indices),int)
 assert y.shape==(189,) and set(y)=={0,1};assert len(ids)==len(set(ids))==186
 a.output.mkdir(parents=True,exist_ok=True)
 np.savez(a.output/'features.npz',features=features,names=np.array(names),labels=y)
 dump(a.output/'results.json',{'numerical_checks':checks,'cohorts':{'primary_186':cohort(ids,features,names,desc,y),'overlapping_189':cohort(np.arange(189),features,names,desc,y)}})
if __name__=='__main__':main()
