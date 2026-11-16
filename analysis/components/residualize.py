"""Training-only covariate residualization and common-bandpower classification."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
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
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
QUALITY=['max_window_ptp_uv','fraction_channel_windows_ptp_gt_200','median_channel_rms_uv','veog_rms_uv','heog_rms_uv','max_eeg_eog_abs_corr']
BAND_COLUMNS=[b+'_log10_power' for b in ['delta','theta','alpha','beta','gamma']]
HR='heart_rate_bpm'; MQ=['mass_log10_power_60_90','beat_quality_f1']
def array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

class Design:
    """Fit medians and numeric missing indicators using training rows only."""
    def fit(self, frame):
        self.columns=list(frame.columns)
        if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in self.columns):
            raise ValueError('All fixed covariate covariates must be numeric')
        self.medians=[]
        for col in self.columns:
            v=frame[col].to_numpy(float)
            if np.isinf(v).any(): raise ValueError('Infinite input')
            self.medians.append(float(np.nanmedian(v)) if np.isfinite(v).any() else 0.)
        return self
    def transform(self,frame):
        if list(frame.columns)!=self.columns: raise ValueError('Column order mismatch')
        parts=[]
        for col,med in zip(self.columns,self.medians):
            v=frame[col].to_numpy(float)
            if np.isinf(v).any(): raise ValueError('Infinite input')
            parts.extend([np.nan_to_num(v,nan=med)[:,None],np.isnan(v)[:,None]])
        return np.concatenate(parts,axis=1).astype(float)
    def state(self):
        return {'columns':self.columns,'medians':self.medians,
                'expanded_columns':[n for c in self.columns for n in (c,c+'_missing')]}

def scaler_state(scaler):
    return {'mean':scaler.mean_,'scale':scaler.scale_,'var':scaler.var_,
            'n_samples_seen':scaler.n_samples_seen_}

def fit_fold(features, y, train, test, nuisance=None):
    design=Design().fit(features.iloc[train])
    x=design.transform(features.iloc[train]); z=design.transform(features.iloc[test])
    scaler=StandardScaler().fit(x)
    x=scaler.transform(x); z=scaler.transform(z)
    state={'feature_design':design.state(),'feature_scaler':scaler_state(scaler)}
    counts={'design':1,'scaler':1,'residualizer':0,'classifier':1}
    if nuisance is not None:
        nd=Design().fit(nuisance.iloc[train])
        nx=nd.transform(nuisance.iloc[train]); nz=nd.transform(nuisance.iloc[test])
        ns=StandardScaler().fit(nx); nx=ns.transform(nx); nz=ns.transform(nz)
        ridge=Ridge(alpha=1).fit(nx,x)
        x=x-ridge.predict(nx); z=z-ridge.predict(nz)
        state.update({'nuisance_design':nd.state(),'nuisance_scaler':scaler_state(ns),
            'ridge':{'alpha':1,'coef':ridge.coef_,'intercept':ridge.intercept_},
            'training_residual_sha256':array_sha(x)})
        rs=StandardScaler().fit(x); x=rs.transform(x); z=rs.transform(z)
        state['residual_scaler']=scaler_state(rs)
        counts.update({'design':2,'scaler':3,'residualizer':1})
    model=LogisticRegression(C=1,class_weight='balanced',max_iter=10000,random_state=17)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,y[train])
    if model.n_iter_.max()>=10000: raise RuntimeError('Logistic model did not converge')
    state['classifier']={'C':1,'class_weight':'balanced','random_state':17,'max_iter':10000,
        'coef':model.coef_,'intercept':model.intercept_,'classes':model.classes_,
        'n_iter':model.n_iter_,'training_design_sha256':array_sha(x)}
    return model.predict_proba(z)[:,1],state,counts

def intersect_folds(folds, ids, y):
    if len(folds)!=5: raise ValueError('Exactly five source folds required')
    keep=set(map(int,ids)); output=[]; seen=[]
    for f in folds:
        tr=np.array([i for i in f['train'] if i in keep],int)
        te=np.array([i for i in f['test'] if i in keep],int)
        if set(tr)&set(te): raise ValueError('Train/test overlap')
        if set(tr)|set(te)!=keep: raise ValueError('Source fold incomplete')
        a=np.bincount(y[tr],minlength=2); b=np.bincount(y[te],minlength=2)
        if min(a)<10 or min(b)<2: raise ValueError('STOP: fixed fold class minimum failed; no resplit')
        output.append({'fold':f['fold'],'train':tr,'test':te,
                       'train_class_counts':a,'test_class_counts':b})
        seen.extend(te.tolist())
    if sorted(seen)!=sorted(keep): raise ValueError('OOF must predict every eligible participant exactly once')
    return output

def oof(features,y,folds,ids,nuisance=None):
    p=np.full(len(y),np.nan); audit=[]; counts={k:0 for k in ('design','scaler','residualizer','classifier')}
    for f in intersect_folds(folds,ids,y):
        tr=f['train'];te=f['test']
        pred,state,n=fit_fold(features,y,tr,te,nuisance)
        if np.isfinite(p[te]).any(): raise ValueError('Duplicate OOF prediction')
        p[te]=pred
        audit.append({**f,'fit_parameters':state})
        for k,v in n.items():counts[k]+=v
    if np.isfinite(p).sum()!=len(ids):raise ValueError('OOF coverage mismatch')
    return p,audit,counts

def metrics(y,p):
    p=np.clip(np.asarray(p,float),np.finfo(float).eps,1-np.finfo(float).eps)
    losses=-(y*np.log(p)+(1-y)*np.log1p(-p))
    return {'auc':float(roc_auc_score(y,p)),'log_loss':float(losses.mean()),
            'balanced_log_loss':float(.5*(losses[y==0].mean()+losses[y==1].mean()))}

def bootstrap_metrics(y,probabilities,n=20000,seed=20261116):
    """Conditional stratified participant bootstrap; identical draws, no refits."""
    rng=np.random.default_rng(seed); classes=[np.flatnonzero(y==k) for k in (0,1)]
    counts=[rng.multinomial(len(ix),np.ones(len(ix))/len(ix),size=n) for ix in classes]
    output={}
    for name,p in probabilities.items():
        p=np.clip(np.asarray(p,float),np.finfo(float).eps,1-np.finfo(float).eps)
        neg=p[classes[0]];pos=p[classes[1]]
        pair=(pos[:,None]>neg[None,:]).astype(float)+.5*(pos[:,None]==neg[None,:])
        auc=np.einsum('bi,ij,bj->b',counts[1],pair,counts[0],optimize=True)/(len(pos)*len(neg))
        l0=-np.log1p(-neg); l1=-np.log(pos)
        ll0=counts[0]@l0;ll1=counts[1]@l1
        output[name]={'auc':auc,'log_loss':(ll0+ll1)/len(y),
                      'balanced_log_loss':.5*(ll0/len(neg)+ll1/len(pos))}
    return output,counts

def summarize(y,probabilities):
    boots,_=bootstrap_metrics(y,probabilities)
    points={name:metrics(y,p) for name,p in probabilities.items()}
    arms={name:{metric:{'estimate':v,'ci95':np.quantile(boots[name][metric],[.025,.975])}
                for metric,v in row.items()} for name,row in points.items()}
    contrasts={}
    for name in ('BH','BM'):
        contrasts[name+'-BD']={metric:{'estimate':points[name][metric]-points['BD'][metric],
            'ci95':np.quantile(boots[name][metric]-boots['BD'][metric],[.025,.975])}
            for metric in points[name]}
    return {'arms':arms,'contrasts':contrasts,'bootstrap':{'n':20000,'seed':20261116,
            'unit':'participant','stratified_by':'diagnosis','refits':0,
            'interval_type':'conditional percentile','primary':'BH-BD','secondary':'BM-BD',
            'loss_contrast_direction':'negative means lower loss in adjusted arm'}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--folds',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 frame=pd.read_csv(a.input);assert len(frame)==189
 y=frame['label'].to_numpy(int);ids=np.flatnonzero(frame['eligible'].to_numpy(int));assert len(ids)==186 and set(y)=={0,1}
 x=frame[BAND_COLUMNS];dem=frame[['age_years','sex_male']];bd=frame[['age_years','sex_male']+QUALITY];folds=read(a.folds)
 assert np.isfinite(frame.loc[ids,[HR]+MQ].to_numpy(float)).all()
 specs={'B0':(x,None),'BD':(x,bd),'BH':(x,frame[list(bd.columns)+[HR]]),'BM':(x,frame[list(bd.columns)+MQ]),'HRonly':(frame[[HR]],None),'MQonly':(frame[MQ],None),'BAge':(x,dem)}
 probs={};audits={}
 for name,(features,nuisance) in specs.items():
  pred,audit,_=oof(features,y,folds,ids,nuisance);probs[name]=pred[ids];audits[name]=audit
 result=summarize(y[ids],probs);boots,_=bootstrap_metrics(y[ids],probs);points={k:metrics(y[ids],v) for k,v in probs.items()}
 for first,second in [('BAge','B0'),('BD','BAge')]:
  result['contrasts'][first+'-'+second]={m:{'estimate':points[first][m]-points[second][m],'ci95':np.quantile(boots[first][m]-boots[second][m],[.025,.975])} for m in points[first]}
 coefficients={}
 for arm,folds in audits.items():
  standardized=[];original=[]
  for f in folds:
   s=f['fit_parameters'];coef=np.asarray(s['classifier']['coef'])[0];rs=np.asarray(s['residual_scaler']['scale']) if 'residual_scaler' in s else np.ones(len(coef))
   standardized.append(coef[::2]);original.append((coef/(np.asarray(s['feature_scaler']['scale'])*rs))[::2])
  coefficients[arm]={'columns':s['feature_design']['columns'],'standardized_fold_coefficients':standardized,'original_unit_fold_coefficients':original,'mean_standardized':np.mean(standardized,axis=0),'mean_original_unit':np.mean(original,axis=0)}
 dump(a.output/'coefficients.json',coefficients)
 dump(a.output/'results.json',result);dump(a.output/'fit_parameters.json',audits);dump(a.output/'oof.json',{'indices':ids,'labels':y[ids],'probabilities':probs})
if __name__=='__main__':main()
