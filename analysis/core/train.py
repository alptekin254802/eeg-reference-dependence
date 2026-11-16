"""Participant-level weighted-loss training on prespecified folds."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,hashlib,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import make_model
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def predict(model,x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(b).cpu().numpy() for b in x.split(4)])

def fit(x, ids, labels, seed, architecture, epochs, validation=None, capture=()):
    # No outer-test labels or arrays enter fitting or model selection.
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    train_x=x[torch.tensor(ids,device=x.device,dtype=torch.long)]
    scale=float(train_x.double().square().mean().sqrt().cpu())
    assert np.isfinite(scale) and scale>0
    train_x=train_x/scale
    target=torch.tensor(labels,dtype=torch.float32,device=x.device)
    weight=torch.tensor(float((1-labels).sum()/labels.sum()),device=x.device)
    model=make_model(architecture).to(x.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    curve=[]; states={}; best=float('inf'); chosen=epochs; stale=0
    if validation is not None:
        vi,vy=validation
        assert not set(ids)&set(vi)
        vx=x[torch.tensor(vi,device=x.device,dtype=torch.long)]/scale
        vt=torch.tensor(vy,dtype=torch.float32,device=x.device)
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for batch in np.array_split(rng.permutation(len(ids)),int(np.ceil(len(ids)/4))):
            b=torch.tensor(batch,device=x.device,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(model(train_x[b]),target[b],pos_weight=weight)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward(); optimizer.step(); model.constrain()
            losses.append(float(loss.detach().cpu()))
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise FloatingPointError('Nonfinite model state')
        point={'epoch':epoch,'weighted_train_loss':float(np.mean(losses))}
        if validation is not None:
            logits=torch.tensor(predict(model,vx),device=x.device)
            per=F.binary_cross_entropy_with_logits(logits,vt,reduction='none')
            value=float((.5*(per[vt==0].mean()+per[vt==1].mean())).cpu())
            point['balanced_validation_bce']=value
            if value<best:best=value;chosen=epoch;stale=0
            else:stale+=1
        curve.append(point)
        if epoch in capture:
            states[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if validation is not None and epoch>=8 and stale>=8:break
    states[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.eval()
    return model,scale,curve,chosen,states


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--max-jobs',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());y=np.load(a.input/'labels.npy');done=0
    hashes={'plan':sha(a.plan),'labels':sha(a.input/'labels.npy'),'code':{n:sha(Path(__file__).with_name(n)) for n in ['train.py','model.py','models.py']}}
    (a.output/'environment.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':a.device,'hashes':hashes},indent=2))
    for job in plan['training_jobs']:
        output=a.output/(job['id']+'.json')
        if output.exists():
            old=json.loads(output.read_text());assert old['job']==job and old['hashes']==hashes
            assert old['input_sha256']==sha(a.input/(job['arm']+'.npy'))
            assert all(sha(a.output/(job['id']+'_epoch'+ep+'.pt'))==h for ep,h in old['checkpoint_sha256'].items())
            continue
        start=time.time();source=a.input/(job['arm']+'.npy');data=np.load(source)
        x=torch.from_numpy(data).to(a.device);tr=np.array(job['train']);te=np.array(job['test']);k=job['fold'];seed=job['seed'];assert not set(tr)&set(te)
        assert np.isfinite(data[np.r_[tr,te]]).all()
        chosen=job['epochs'];inner_curve=None
        labels=y[tr]
        if job['mode']=='label_null':labels=np.random.default_rng(77119+k).permutation(labels)
        if job['mode']=='balanced_stop':
            ti=np.array(job['inner_train']);vi=np.array(job['inner_val']);assert set(ti)|set(vi)==set(tr) and not set(ti)&set(vi)
            inner,_,inner_curve,chosen,_=fit(x,ti,y[ti],seed+100*k,job['architecture'],30,(vi,y[vi]));del inner
        captures=[10,20,30,60,100] if job['mode']=='long100' else [chosen]
        model,scale,curve,_,states=fit(x,tr,labels,seed+100*k+50000,job['architecture'],chosen,capture=captures)
        predictions={};checkpoints={}
        for ep,state in sorted(states.items()):
            model.load_state_dict(state);model.eval();predictions[str(ep)]=predict(model,x[torch.tensor(te,device=x.device)]/scale).tolist()
            ck=a.output/(job['id']+'_epoch'+str(ep)+'.pt')
            torch.save({'state':state,'scale':scale,'train_ids':tr.tolist(),'test_ids':te.tolist(),'job':job,'epoch':ep,'hashes':hashes},ck);checkpoints[str(ep)]=sha(ck)
        result={'job':job,'test_ids':te.tolist(),'logits_by_epoch':predictions,'epochs':chosen,'scale':scale,'inner_curve':inner_curve,'refit_curve':curve,'seconds':time.time()-start,'checkpoint_sha256':checkpoints,'hashes':hashes,'input_sha256':sha(source)}
        output.write_text(json.dumps(result,indent=2));print(job['id'],flush=True);done+=1
        del x,model,states
        if a.max_jobs and done>=a.max_jobs:break
if __name__=='__main__':main()
