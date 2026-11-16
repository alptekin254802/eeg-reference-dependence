"""Aggregate accepted explanation configurations and matched sensitivity contrasts."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external'))
from metrics import PersonBootstrap

def read_records(directory,method):
 d=Path(directory);g=json.loads((d/'complete.json').read_text())[method][-1];assert g['accepted'],'Configuration failed completeness limits';steps=g['steps'];rows={}
 for p in d.glob(f'{method}_steps{steps}_*.json'):
  r=json.loads(p.read_text());r['map']=p.with_suffix('.npz');rows[r['index']]=r
 assert len(rows)==g['n'];return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--comparison');p.add_argument('--method',choices=['IG','DeepSHAP'],default='IG');p.add_argument('--output',required=True);a=p.parse_args();rows=read_records(a.input,a.method);ids=sorted(rows);y=np.array([rows[i]['label'] for i in ids]);v=[rows[i]['values'] for i in ids];boot=PersonBootstrap(y);keys=['faithfulness','normalized_faithfulness','probability_faithfulness','energy_matched_faithfulness','energy_matched_probability_faithfulness'];result={'n':len(ids),'effects':{k:boot.mean_ci(np.array([r[k] for r in v])) for k in keys}}
 for k in ['deletion_auc_difference','donor_probability_faithfulness']:
  if all('perturbations' in r for r in v):result['effects'][k]=boot.mean_ci(np.array([r['perturbations'][k] for r in v]))
 bins={'correct':[r for r,label in zip(v,y) if (r['logit']>=0)==bool(label)],'incorrect':[r for r,label in zip(v,y) if (r['logit']>=0)!=bool(label)]}
 bins.update({f'abs_logit_{lo}_{hi}':[r for r in v if lo<=abs(r['logit'])<hi] for lo,hi in [(0,1),(1,2),(2,float('inf'))]})
 result['confidence_descriptive']={name:{'n':len(rr),'mean_faithfulness':float(np.mean([r['faithfulness'] for r in rr])) if rr else None} for name,rr in bins.items()}
 if a.comparison:
  other=read_records(a.comparison,a.method);assert set(other)==set(rows);corr=[];cos=[];abscos=[]
  for i in ids:
   u=rows[i]['values']['signed_roi_scores'];w=other[i]['values']['signed_roi_scores'];corr.append(float(spearmanr(list(u.values()),[w[k] for k in u]).statistic))
   with np.load(rows[i]['map']) as z:x=z['attribution'].astype('float64').ravel()
   with np.load(other[i]['map']) as z:b=z['attribution'].astype('float64').ravel()
   if x.shape==b.shape:
    den=np.linalg.norm(x)*np.linalg.norm(b);cos.append(float(x@b/den) if den>0 else None);abscos.append(float(np.abs(x)@np.abs(b)/den) if den>0 else None)
  valid=np.isfinite(corr);result['comparison']={'signed_rank_median':float(np.median(np.asarray(corr)[valid])) if valid.any() else None,'signed_rank_mean':float(np.mean(np.asarray(corr)[valid])) if valid.any() else None,'signed_rank_undefined_n':int((~valid).sum()),'signed_cosine_mean':float(np.mean([x for x in cos if x is not None])) if any(x is not None for x in cos) else None,'absolute_cosine_mean':float(np.mean([x for x in abscos if x is not None])) if any(x is not None for x in abscos) else None,'faithfulness_difference':boot.mean_ci(np.array([rows[i]['values']['faithfulness']-other[i]['values']['faithfulness'] for i in ids])),'top_region_agreement':float(np.mean([rows[i]['values']['top_region']==other[i]['values']['top_region'] for i in ids]))}
  # Undefined rank values have no numerical replacement in summaries.
 Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
