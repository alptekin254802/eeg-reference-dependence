"""Participant-wise RMS normalization with matched 30-epoch EEGNet training."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
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
import sys,time
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from model import EEGNet
ARMS=('linked_basic','average_basic')
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
def require_training_runtime():
 assert torch.__version__=='2.0.1+cu118' and np.__version__=='1.26.4', 'PyTorch 2.0.1+cu118 and NumPy 1.26.4 are required'

def normalize(x):
 assert x.ndim==4 and np.isfinite(x).all()
 rms=np.sqrt(np.mean(np.square(x,dtype=np.float64),axis=(1,2,3)))
 assert np.isfinite(rms).all() and (rms>0).all()
 return (x/rms[:,None,None,None]).astype(np.float32),rms

def predict(m,x):
 m.eval()
 with torch.no_grad():return np.concatenate([m(b).cpu().numpy() for b in x.split(4)])

def fit(x,tr,y,seed):
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
 tx=x[torch.tensor(tr,device='cuda',dtype=torch.long)];scale=float(tx.double().square().mean().sqrt().cpu());tx=tx/scale
 target=torch.tensor(y,dtype=torch.float32,device='cuda');weight=torch.tensor(float((1-y).sum()/y.sum()),device='cuda');m=EEGNet().cuda();opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=1e-4);curve=[]
 for ep in range(1,31):
  m.train();losses=[]
  for batch in np.array_split(rng.permutation(len(tr)),int(np.ceil(len(tr)/4))):
   b=torch.tensor(batch,device='cuda',dtype=torch.long);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(m(tx[b]),target[b],pos_weight=weight)
   assert torch.isfinite(loss);loss.backward();opt.step();m.constrain();losses.append(float(loss.detach().cpu()))
  curve.append({'epoch':ep,'weighted_train_loss':float(np.mean(losses))})
 return m,scale,curve
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','report']);p.add_argument('--input',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 y=np.load(a.input/'labels.npy');folds=read(a.input/'folds.json');assert y.shape==(189,) and y.sum()==56 and len(folds)==5
 hits=np.zeros(189,int)
 for f in folds:
  tr=f['train'];te=f['test'];assert not set(tr)&set(te) and set(tr)|set(te)==set(range(189));hits[te]+=1
 assert (hits==1).all()
 a.output.mkdir(parents=True,exist_ok=True)
 if a.mode=='train':
  require_training_runtime();assert torch.cuda.is_available()
  torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
  for arm in ARMS:
   raw=np.load(a.input/f'{arm}.npy');assert raw.shape==(189,15,19,800) and raw.dtype==np.float32
   z,r=normalize(raw);dump(a.output/f'{arm}_rms.json',{'rms':r});x=torch.from_numpy(z).cuda()
   for f in folds:
    k=f['fold'];out=a.output/f'{arm}_fold{k}.json'
    if out.exists():raise FileExistsError(out)
    t=time.time();tr=np.array(f['train']);te=np.array(f['test']);seed=1109+100*k+50000
    m,scale,curve=fit(x,tr,y[tr],seed);pred=predict(m,x[torch.tensor(te,device='cuda')]/scale)
    torch.save({'state':m.cpu().state_dict(),'scale':scale,'train_ids':tr.tolist(),'test_ids':te.tolist(),'seed':seed,'epoch':30},a.output/f'{arm}_fold{k}.pt')
    dump(out,{'arm':arm,'fold':k,'logits':pred,'test_ids':te,'scale':scale,'curve':curve,'seconds':time.time()-t});del m;torch.cuda.empty_cache()
   del x;torch.cuda.empty_cache()
 else:
  if a.baseline is None:p.error('--baseline is required for report')
  prob={};b=PersonBootstrap(y,20261116,20000)
  for arm in ARMS:
   z=np.empty(189);old=np.empty(189)
   for f in folds:
    k=f['fold'];j=read(a.output/f'{arm}_fold{k}.json');assert j['test_ids']==f['test'];z[f['test']]=j['logits']
    j=read(a.baseline/f'fixed30_eegnet_{arm}_seed1109_split20261109_fold{k}.json');assert j['test_ids']==f['test'];old[f['test']]=j['logits_by_epoch']['30']
   prob['rms_'+arm]=expit(z);prob['original_'+arm]=expit(old)
  out={'uncertainty':'Conditional on fixed OOF predictions, cohort and training; no training/split uncertainty. Shared diagnosis-stratified participant bootstrap draws.','bootstrap_reps':20000,'seed':20261116,'metrics':{k:b.evaluate(v) for k,v in prob.items()},'contrasts':{}}
  for arm in ARMS:out['contrasts']['rms_minus_original_'+arm]=b.auc_difference(prob['rms_'+arm],prob['original_'+arm])
  for typ in ('rms','original'):out['contrasts'][typ+'_linked_minus_average']=b.auc_difference(prob[typ+'_linked_basic'],prob[typ+'_average_basic'])
  d=(b.auc_samples(prob['rms_linked_basic'])-b.auc_samples(prob['rms_average_basic']))-(b.auc_samples(prob['original_linked_basic'])-b.auc_samples(prob['original_average_basic']))
  out['reference_contrast_change']={'difference':out['contrasts']['rms_linked_minus_average']['difference']-out['contrasts']['original_linked_minus_average']['difference'],'ci95':np.percentile(d,[2.5,97.5])}
  dump(a.output/'results.json',out);dump(a.output/'oof.json',prob)
if __name__=='__main__':main()
