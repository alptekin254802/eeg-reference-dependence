"""Train-only spectral estimation and classical controls."""
import argparse,json,warnings
from pathlib import Path
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.exceptions import ConvergenceWarning
def logistic(x,y,z,C=1):
    scaler=StandardScaler().fit(x)
    for iterations in [3000,10000]:
        model=LogisticRegression(C=C,class_weight='balanced',max_iter=iterations,random_state=17)
        with warnings.catch_warnings(record=True) as ww:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(scaler.transform(x),y)
        if not any(issubclass(w.category,ConvergenceWarning) for w in ww):break
    else:raise RuntimeError('Logistic regression did not converge at 10000 iterations.')
    return model.predict_proba(scaler.transform(z))[:,1]

def oof(x,y,folds,residualize=None,nested=False):
    p=np.full(len(y),np.nan);audit=[]
    for f in folds:
        tr=np.array(f['train']);te=np.array(f['test']);assert not set(tr)&set(te)
        xx=x[tr];zz=x[te];C=1
        if residualize is not None:
            nx=StandardScaler().fit(residualize[tr]);sx=StandardScaler().fit(xx)
            ridge=Ridge(alpha=1,fit_intercept=True).fit(nx.transform(residualize[tr]),sx.transform(xx))
            xx=sx.transform(xx)-ridge.predict(nx.transform(residualize[tr]))
            zz=sx.transform(zz)-ridge.predict(nx.transform(residualize[te]))
        if nested:
            scores=[]
            for c in [.01,.1,1,10,100]:
                losses=[]
                for it,iv in StratifiedKFold(3,shuffle=True,random_state=841+f['fold']).split(xx,y[tr]):
                    q=np.clip(logistic(xx[it],y[tr][it],xx[iv],c),1e-7,1-1e-7)
                    target=y[tr][iv]
                    losses.append(float(.5*(-np.log(q[target==1]).mean()-np.log1p(-q[target==0]).mean())))
                scores.append(float(np.mean(losses)))
            C=[.01,.1,1,10,100][int(np.argmin(scores))]
        p[te]=logistic(xx,y[tr],zz,C)
        audit.append({'fold':f['fold'],'C':C,'train_n':len(tr),'test_n':len(te)})
    return p,audit

def spectral(x):
    f,psd=welch(x,fs=200,nperseg=400,axis=-1)
    return np.concatenate([np.log10(np.maximum(psd[...,(f>=lo)&(f<hi)].sum(-1)*.5,1e-12)).mean(1)
        for lo,hi in [(1,4),(4,8),(8,13),(13,30),(30,40.01)]],axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path);p.add_argument('--nested',action='store_true');p.add_argument('--permutations',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    y=np.load(a.input/'labels.npy');folds=json.loads((a.input/'folds.json').read_text())
    rows=json.loads((a.input/'cohort.json').read_text());arms=json.loads((a.input/'manifest.json').read_text())['arms']
    dem=np.array([[float(r['age_years']),float(r['sex']=='Male')] for r in rows]);quality=np.load(a.input/'quality_features.npy');nuisance=np.c_[dem,quality]
    predictions={}
    def save(name,x,ff=folds,**kw):
        predictions[name]=oof(x,y,ff,**kw)[0]
    save('demographics',dem);save('quality_eog',quality);save('demographics_quality',nuisance)
    archive=np.array([[r['dataset_membership'] or 'not_reported'] for r in rows]);pred=np.full(len(y),np.nan)
    for f in folds:
        tr,te=np.array(f['train']),np.array(f['test']);encoder=OneHotEncoder(handle_unknown='ignore').fit(archive[tr])
        model=LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=17).fit(encoder.transform(archive[tr]),y[tr]);pred[te]=model.predict_proba(encoder.transform(archive[te]))[:,1]
    predictions['archive_membership']=pred
    features={}
    for arm in arms:
        x=np.load(a.input/(arm+'_spectral.npy'));features[arm]=x
        save('spectral_'+arm,x)
        if a.nested:save('nested_spectral_'+arm,x,nested=True)
        save('demographics_spectral_'+arm,np.c_[dem,x]);save('demographics_quality_spectral_'+arm,np.c_[nuisance,x])
        save('residual_spectral_'+arm,x,residualize=nuisance)
        for split in [20261110,20261111]:
            ff=[{'fold':k,'train':tr.tolist(),'test':te.tolist()} for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=split).split(y,y))]
            save('split'+str(split)+'_spectral_'+arm,x,ff)
    for suffix in ['basic','eog']:
        common=spectral(np.load(a.input/('linked_'+suffix+'.npy'),mmap_mode='r').mean(2,keepdims=True))
        save('common_'+suffix,common);save('average_plus_common_'+suffix,np.c_[features['average_'+suffix],common])
    if a.plan:
        plan=json.loads(a.plan.read_text());groups={}
        for job in plan['training_jobs']:
            if job['mode'] not in ['age_sex_matched','strict_quality','csd']:continue
            key=(job['mode'],job['arm']);groups.setdefault(key,{})[job['fold']]={k:job[k] for k in ['fold','train','test']}
        for (mode,arm),ff in groups.items():
            x=spectral(np.load(a.input/(arm+'.npy')));save(mode+'_spectral_'+arm,x,list(ff.values()))
    # Common-waveform frequency vectors reported in the exploratory sensitivity.
    common=np.load(a.input/'linked_basic.npy',mmap_mode='r').mean(2)
    freq,psd=welch(common,fs=200,nperseg=400,axis=-1)
    masks=[(freq>=1)&(freq<4),(freq>=4)&(freq<8),(freq>=8)&(freq<13),(freq>=13)&(freq<=20),(freq>20)&(freq<30),(freq>=30)&(freq<=40)]
    band=np.stack([np.log10(np.maximum(psd[...,mask].sum(-1)*.5,1e-12)).mean(1) for mask in masks],axis=1)
    for name,cols in [('low_four_bands',[0,1,2,3]),('high_two_bands',[4,5]),('all_six_bands',[0,1,2,3,4,5])]:
        values=band[:,cols];x=np.zeros((len(y),2*len(cols)));x[:,::2]=values
        pred=np.full(len(y),np.nan)
        for f in folds:
            tr,te=np.array(f['train']),np.array(f['test']);sc=StandardScaler().fit(x[tr]);model=LogisticRegression(C=1,class_weight='balanced',max_iter=10000,random_state=17).fit(sc.transform(x[tr]),y[tr]);assert model.n_iter_.max()<10000;pred[te]=model.predict_proba(sc.transform(x[te]))[:,1]
        predictions[name]=pred
    np.savez(a.output/'oof_predictions.npz',**predictions)
    if a.permutations:
        from sklearn.metrics import roc_auc_score
        rng=np.random.default_rng(20261111);result=[]
        for b in range(a.permutations):
            yp=rng.permutation(y);ff=[{'fold':k,'train':tr.tolist(),'test':te.tolist()} for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261109).split(yp,yp))]
            auc={arm:float(roc_auc_score(yp,oof(x,yp,ff)[0])) for arm,x in features.items()}
            result.append({'permutation':b,'auc':auc,'reference_difference_basic':auc['linked_basic']-auc['average_basic'],'reference_difference_eog':auc['linked_eog']-auc['average_eog']})
        (a.output/'spectral_permutations.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
