"""Evaluate signal transformations with each participant's held-out source model."""
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from models import make_model
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external'))
from metrics import PersonBootstrap

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_heldout(path,fold,device):
    # Check the full source partition even when evaluating only a temporal subset.
    ck=torch.load(path,map_location='cpu')
    assert ck['train_ids']==fold['train'] and ck['test_ids']==fold['test'],'Checkpoint must match this outer fold'
    assert not set(ck['train_ids'])&set(ck['test_ids']),'Training/test overlap'
    assert ck['epoch']==30,'Use the 30-epoch source model'
    if 'job' in ck:
        assert ck['job']['architecture']=='eegnet' and ck['job']['seed']==1109
    assert np.isfinite(ck['scale']) and ck['scale']>0
    model=make_model('eegnet').to(device);model.load_state_dict(ck['state']);model.eval()
    return model,ck['scale']

@torch.no_grad()
def evaluate(input_dir,folds,checkpoint_dir,template,device,mode,temporal=None):
    y=np.load(input_dir/'labels.npy');arms=read(input_dir/'manifest.json')['arms'];output={};checkpoint_hashes={}
    ids=sorted(read(temporal/'COMPLETE.json')['eligible_ids']) if temporal is not None else list(range(len(y)))
    assert len(ids)==len(set(ids)) and all(0<=i<len(y) for i in ids)
    assert sorted(i for f in folds for i in f['test'])==list(range(len(y)))
    if mode=='common_removal':arms=['linked_basic','linked_eog']
    for arm in arms:
        if mode=='common_removal':arrays={'original':np.load(input_dir/(arm+'.npy'),mmap_mode='r')}
        else:arrays={t:np.load(temporal/(t+'_'+arm+'.npy'),mmap_mode='r') for t in ['early','late']}
        logits={name:np.full(len(y),np.nan) for name in (['original','common_removed'] if mode=='common_removal' else ['early','late'])}
        for fold in folds:
            path=checkpoint_dir/template.format(arm=arm,fold=fold['fold']);model,scale=load_heldout(path,fold,device);checkpoint_hashes[path.name]=sha(path)
            for i in sorted(set(fold['test'])&set(ids)):
                if mode=='common_removal':
                    x=torch.tensor(np.asarray(arrays['original'][i]).copy()/scale,device=device)
                    assert torch.isfinite(x).all();z=x-x.mean(-2,keepdim=True)
                    q=model(torch.stack([x,z])).cpu().numpy();logits['original'][i]=q[0];logits['common_removed'][i]=q[1]
                else:
                    for name,array in arrays.items():
                        x=torch.tensor(np.asarray(array[i]).copy()/scale,device=device)
                        assert x.shape==(13,19,800) and torch.isfinite(x).all()
                        logits[name][i]=float(model(x[None]).cpu())
            del model
        values={name:v[ids] for name,v in logits.items()};assert all(np.isfinite(v).all() for v in values.values());boot=PersonBootstrap(y[ids]);prob={name:expit(v) for name,v in values.items()}
        if mode=='common_removal':
            output[arm]={'original':boot.evaluate(prob['original']),'common_removed':boot.evaluate(prob['common_removed']),'auc_change':boot.auc_difference(prob['common_removed'],prob['original']),'logit_change':boot.mean_ci(values['common_removed']-values['original'])}
        else:
            output[arm]={'n':len(ids),'early':boot.evaluate(prob['early']),'late':boot.evaluate(prob['late']),'late_minus_early_auc':boot.auc_difference(prob['late'],prob['early']),'probability_change':boot.mean_ci(prob['late']-prob['early'])}
    return {'metrics':output,'n':len(ids),'mode':mode,'bootstrap':{'replicates':2000,'seed':20261110,'unit':'participant stratified by diagnosis','conditioning':'fixed source models'},'checkpoint_sha256':checkpoint_hashes}

def self_test():
    torch.set_num_threads(2);torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);cache=root/'cache';ckdir=root/'checkpoints';temporal=root/'temporal'
        for d in [cache,ckdir,temporal]:d.mkdir()
        y=np.array([0,1,0,1]);np.save(cache/'labels.npy',y);(cache/'manifest.json').write_text(json.dumps({'arms':['linked_basic','average_basic','linked_eog','average_eog']}));(temporal/'COMPLETE.json').write_text(json.dumps({'eligible_ids':[0,1,2,3]}))
        folds=[{'fold':0,'train':[2,3],'test':[0,1]},{'fold':1,'train':[0,1],'test':[2,3]}];rng=np.random.default_rng(5)
        for arm in ['linked_basic','average_basic','linked_eog','average_eog']:
            np.save(cache/(arm+'.npy'),rng.normal(size=(4,15,19,800)).astype(np.float32))
            for interval in ['early','late']:np.save(temporal/(interval+'_'+arm+'.npy'),rng.normal(size=(4,13,19,800)).astype(np.float32))
            for fold in folds:
                torch.save({'state':make_model('eegnet').state_dict(),'scale':1.2,'train_ids':fold['train'],'test_ids':fold['test'],'epoch':30},ckdir/(arm+str(fold['fold'])+'.pt'))
        for mode in ['common_removal','temporal']:
            result=evaluate(cache,folds,ckdir,'{arm}{fold}.pt','cpu',mode,temporal if mode=='temporal' else None);assert result['n']==4
            assert len(result['metrics'])==(2 if mode=='common_removal' else 4)
        wrong={'fold':0,'train':[0,1],'test':[2,3]}
        try:load_heldout(ckdir/'linked_basic0.pt',wrong,'cpu');raise RuntimeError('Mismatched held-out partition accepted')
        except AssertionError:pass
    print('Synthetic CPU checks passed: paired transforms, temporal shapes, held-out checkpoint enforcement')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');p.add_argument('--input',type=Path);p.add_argument('--checkpoints',type=Path);p.add_argument('--checkpoint-template',default='fixed30_eegnet_{arm}_seed1109_split20261109_fold{fold}_epoch30.pt');p.add_argument('--mode',choices=['common_removal','temporal']);p.add_argument('--temporal',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.self_test:self_test();return
    if any(v is None for v in [a.input,a.checkpoints,a.mode,a.output]):p.error('--input, --checkpoints, --mode and --output are required')
    if a.mode=='temporal' and a.temporal is None:p.error('--temporal is required for temporal mode')
    result=evaluate(a.input,read(a.input/'folds.json'),a.checkpoints,a.checkpoint_template,a.device,a.mode,a.temporal if a.mode=='temporal' else None)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print('Wrote aggregate',a.mode,'metrics')
if __name__=='__main__':main()
