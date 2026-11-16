"""Unadapted external predictions from source checkpoints."""
import argparse,json,sys,os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from models import make_model

def evaluate(model,x):
 with torch.no_grad():return np.concatenate([model(b).cpu().numpy() for b in x.split(4)])
def main():
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--checkpoints',nargs='+');p.add_argument('--training-plan');p.add_argument('--training');p.add_argument('--arm');p.add_argument('--seeds',nargs='+',type=int,default=[1109]);p.add_argument('--architecture',default='eegnet');p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);a=p.parse_args()
 if a.training_plan:
  jobs=json.loads(Path(a.training_plan).read_text())['training_jobs'];selected=sorted([j for j in jobs if j['mode']=='fixed30' and j['architecture']==a.architecture and j['arm']==a.arm and j['seed'] in a.seeds and j['split_seed']==20261109],key=lambda j:(j['seed'],j['fold']));assert len(selected)==5*len(a.seeds);a.checkpoints=[str(Path(a.training)/(j['id']+'_epoch30.pt')) for j in selected]
 assert a.checkpoints,'Supply checkpoints or a training plan'
 x=np.load(a.input,allow_pickle=False);predictions={'none':[]}
 for checkpoint in a.checkpoints:
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model(a.architecture).to(a.device);model.load_state_dict(ck['state']);model.eval();scaled=torch.tensor(x/ck['scale'],device=a.device)
  predictions['none'].append(expit(evaluate(model,scaled)).tolist())
 out={name:np.mean(values,axis=0).tolist() for name,values in predictions.items()};Path(a.output).write_text(json.dumps({'probabilities':out,'individual':predictions},indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
