"""Run complete explanation configurations with shared numerical acceptance gates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from models import make_model
from run import REGIONS
from attribution import WindowTarget,expected_ig,deep_shap,measures,perturbations,shap
from selection import panel_and_background,completeness_check

def run(config,output):
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 xarray=np.load(config['input'],mmap_mode='r');barray=np.load(config.get('background_input',config['input']),mmap_mode='r');labels=np.load(config['labels']);channels=config['channels'];rois={k:[channels.index(c) for c in cs] for k,cs in REGIONS.items()};device=config.get('device','cpu');methods=config.get('methods',['IG','DeepSHAP']);folds=config['folds'];summary={};rows_by_method={};indices=config.get('input_indices',list(range(len(xarray))));lookup={i:j for j,i in enumerate(indices)}
 for method in methods:
  start=config.get('steps',64 if config.get('background')=='zero' else 32)
  for steps in ([start] if method=='DeepSHAP' else sorted(set([start,64,128]))):
   records=[]
   for foldno,fold in enumerate(folds):
    ck=torch.load(fold['checkpoint'],map_location='cpu',weights_only=False);assert ck['train_ids']==fold['train'] and ck['test_ids']==fold['test'];torch.manual_seed(config.get('random_seed',7109))
    model=make_model(config.get('architecture','eegnet')).to(device);model.load_state_dict(ck['state']);reset=config.get('reset');torch.manual_seed(config.get('random_seed_by_fold',{}).get(str(foldno),config.get('random_seed',7109)))
    if reset=='all':model=make_model(config.get('architecture','eegnet')).to(device);model.constrain()
    elif reset:
     for name in (['head'] if reset=='classifier' else ['head','depth','point','bn3']):getattr(model,name).reset_parameters()
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    panel,bgids,bgwindows=panel_and_background(labels,fold['train'],fold['test'],foldno,config.get('background')=='large')
    people=fold['test'] if config.get('selection')=='all' else panel.tolist();people=[i for i in people if i in config.get('eligible_indices',indices)];assert all(i in lookup for i in people),'Missing selected input row'
    assert not set(bgids)&set(people) and set(bgids)<=set(ck['train_ids'])
    bg=np.stack([barray[i,w] for i,w in zip(bgids,bgwindows)])[:,None]/ck['scale']
    if config.get('background')=='zero':bg=np.zeros((1,1,19,800),dtype='float32')
    background=torch.tensor(bg,device=device);wrapper=WindowTarget(model,config.get('average',False)).eval();explainer=shap.DeepExplainer(wrapper,background) if method=='DeepSHAP' else None
    for i in people:
     assert i in ck['test_ids'] and i not in ck['train_ids'];x=torch.tensor(np.asarray(xarray[lookup[i],:,None]).copy()/ck['scale'],device=device);attr=expected_ig(wrapper,x,background,steps) if method=='IG' else deep_shap(wrapper,x,explainer);assert np.isfinite(attr).all();arm='average' if config.get('average') else 'linked';values=measures(model,wrapper,x,background,attr,arm,rois,i,steps)
     if config.get('perturbations',False):values['perturbations']=perturbations(model,x,background,attr,values,arm,i,rois)
     record={'index':i,'label':int(labels[i]),'fold':foldno,'values':values};records.append(record);base=output/f'{method}_steps{steps}_{i}';np.savez_compressed(base.with_suffix('.npz'),attribution=attr);base.with_suffix('.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
   assert len({r['index'] for r in records})==len(records),'Repeated held-out participant';gate=completeness_check([r['values'] for r in records]);gate.update(steps=steps,n=len(records));summary.setdefault(method,[]).append(gate)
   if gate['accepted'] or method=='DeepSHAP':rows_by_method[method]=records;break
 (output/'complete.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 return summary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();config=json.loads(Path(a.manifest).read_text(encoding='utf-8-sig'));out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.use_deterministic_algorithms(True);run(config,out)
if __name__=='__main__':main()
