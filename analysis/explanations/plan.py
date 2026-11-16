"""Generate all reported explanation configurations from public preprocessing/training plans."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--training',type=Path,nargs='+',required=True);p.add_argument('--training-plan',type=Path,nargs='+',required=True);p.add_argument('--temporal',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--randomization-controls',action='store_true');a=p.parse_args();meta=json.loads((a.input/'manifest.json').read_text());jobs=[j for path in a.training_plan for j in json.loads(path.read_text())['training_jobs']];arms=meta['arms'];a.output.mkdir(parents=True,exist_ok=True);configs={};(a.output/'channels.json').write_text(json.dumps(meta['channels'])+'\n')
 def checkpoint(job):
  candidates=[d/(job['id']+'_epoch30.pt') for d in a.training if (d/(job['id']+'_epoch30.pt')).is_file()]
  if len(candidates)!=1:raise FileNotFoundError('Expected one checkpoint for '+job['id'])
  return str(candidates[0].resolve())
 def add(name,arm,architecture='eegnet',seed=1109,mode='fixed30',**extra):
  jj=sorted([j for j in jobs if j['mode']==mode and j['architecture']==architecture and j['seed']==seed and j['arm']==arm and j['split_seed']==20261109],key=lambda j:j['fold']);assert len(jj)==5,f'Missing training-plan jobs: {mode}/{architecture}/{arm}/{seed}'
  folds=[{'train':j['train'],'test':j['test'],'checkpoint':checkpoint(j)} for j in jj];c={'input':str((a.input/(arm+'.npy')).resolve()),'labels':str((a.input/'labels.npy').resolve()),'channels':meta['channels'],'architecture':architecture,'average':arm.startswith('average'),'folds':folds,'device':a.device,**extra};key=name+'_'+arm;path=a.output/(key+'.json');path.write_text(json.dumps(c,indent=2)+'\n');configs[key]=str(path.resolve())
 for seed in [1109,2209,3309]:
  for arm in arms:add(f'initialization{seed}',arm,seed=seed,perturbations=seed==1109)
 for arm in arms:
  add('temporal_cnn',arm,architecture='temporal_cnn');add('zero_background',arm,background='zero',steps=64,methods=['IG']);add('large_background',arm,background='large',methods=['IG']);add('full_cohort',arm,selection='all',methods=['IG'])
  for stage in ['classifier','separable','all']:
   for seed in [7109,7209,7309]:add(f'randomized_{stage}_{seed}',arm,reset=stage,random_seed=seed)
  if a.randomization_controls:
   add('parameter_control',arm,reset='all',random_seed_by_fold={str(k):1109+100*k+50000 for k in range(5)})
   add('permuted_label_control',arm,mode='label_null')
 if a.temporal:
  eligibility=json.loads((a.temporal/'COMPLETE.json').read_text())
  for interval in ['early','late']:
   for arm in arms:add('temporal_'+interval,arm,input=str((a.temporal/(interval+'_'+arm+'.npy')).resolve()),background_input=str((a.input/(arm+'.npy')).resolve()),eligible_indices=eligibility['eligible_ids'],selection='all',methods=['IG'])
 for arm in ['csd_basic','csd_eog']:
  if any(j['mode']=='csd' and j['arm']==arm for j in jobs):add('surface_laplacian',arm,mode='csd',methods=['IG'])
 (a.output/'index.json').write_text(json.dumps(configs,indent=2)+'\n')
 # This explicit index also provides primary paired contrast and agreement input maps.
 contrast={'initializations':{str(s):{arm:str((a.output/f'initialization{s}_{arm}').resolve()) for arm in arms} for s in [1109,2209,3309]}}
 if a.randomization_controls:contrast['controls']={name:{arm:str((a.output/(name+'_'+arm)).resolve()) for arm in arms} for name in ['parameter_control','permuted_label_control']}
 (a.output/'contrasts.json').write_text(json.dumps(contrast,indent=2)+'\n')
 for suffix in ['basic','eog']:
  agreement={str(s):{ref:str((a.output/f'initialization{s}_{ref}_{suffix}').resolve()) for ref in ['linked','average']} for s in [1109,2209,3309]};(a.output/('agreement_'+suffix+'.json')).write_text(json.dumps(agreement,indent=2)+'\n')
 print('Generated',len(configs),'configurations; no model inference or fitting')
if __name__=='__main__':main()
