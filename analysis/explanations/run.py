"""Explain one held-out participant using a source training-only background."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from models import make_model
from attribution import WindowTarget,expected_ig,deep_shap,measures,perturbations,shap
REGIONS={'frontal':['Fp1','Fp2','F7','F3','Fz','F4','F8'],'central':['C3','Cz','C4'],'temporal':['T7','T8'],'parietal':['P7','P3','Pz','P4','P8'],'occipital':['O1','O2']}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['input','background','checkpoint','channels','output']:p.add_argument('--'+key,required=True)
 p.add_argument('--architecture',default='eegnet',choices=['eegnet','temporal_cnn']);p.add_argument('--method',default='IG',choices=['IG','DeepSHAP']);p.add_argument('--average',action='store_true');p.add_argument('--index',type=int,required=True);p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=32);p.add_argument('--reset',choices=['classifier','separable','all']);p.add_argument('--random-seed',type=int,default=7109);a=p.parse_args()
 torch.manual_seed(a.random_seed);torch.use_deterministic_algorithms(True)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=make_model(a.architecture).to(a.device);model.load_state_dict(ck['state']);torch.manual_seed(a.random_seed)
 if a.reset=='all':model=make_model(a.architecture).to(a.device);model.constrain()
 elif a.reset:
  for name in (['head'] if a.reset=='classifier' else ['head','depth','point','bn3']):getattr(model,name).reset_parameters()
 model.eval()
 for v in model.parameters():v.requires_grad_(False)
 channels=json.loads(Path(a.channels).read_text());rois={k:[channels.index(c) for c in cs] for k,cs in REGIONS.items()}
 xx=np.load(a.input,allow_pickle=False);bb=np.load(a.background,allow_pickle=False)
 assert xx.ndim==3 and xx.shape[1:]==(19,800) and bb.ndim==3 and bb.shape[1:]==(19,800)
 x=torch.tensor(xx[:,None]/ck['scale'],device=a.device);bg=torch.tensor(bb[:,None]/ck['scale'],device=a.device);wrapper=WindowTarget(model,a.average).eval()
 records=[]
 for steps in [a.steps]:
  attr=expected_ig(wrapper,x,bg,steps) if a.method=='IG' else deep_shap(wrapper,x,shap.DeepExplainer(wrapper,bg))
  assert np.isfinite(attr).all()
  rec=measures(model,wrapper,x,bg,attr,'average' if a.average else 'linked',rois,a.index,steps);records.append(rec)
  if rec['relative_residual']<=.05 or a.method=='DeepSHAP':break
 rec['perturbations']=perturbations(model,x,bg,attr,rec,'average' if a.average else 'linked',a.index,rois)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),attribution=attr);out.write_text(json.dumps({'method':a.method,'values':rec,'integration_attempts':records},indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
