"""Paired ocular-regression and randomization summaries across initializations/references."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external'))
from metrics import PersonBootstrap
from summarize import read_records

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--method',choices=['IG','DeepSHAP'],default='IG');p.add_argument('--output',required=True);a=p.parse_args();spec=json.loads(Path(a.manifest).read_text());sets=spec['initializations'];records={s:{arm:read_records(path,a.method) for arm,path in arms.items()} for s,arms in sets.items()};first=next(iter(next(iter(records.values())).values()));ids=sorted(first);y=np.array([first[i]['label'] for i in ids]);boot=PersonBootstrap(y)
 assert all(set(rows)==set(ids) for arms in records.values() for rows in arms.values());keys=['faithfulness','normalized_faithfulness','probability_faithfulness','energy_matched_faithfulness','energy_matched_probability_faithfulness'];result={}
 for title,seeds in [('primary_new_initializations',['2209','3309']),('initialization1109',['1109']),('three_initializations',['1109','2209','3309'])]:
  if not all(s in records for s in seeds):continue
  per={}
  for key in keys:
   values=np.array([np.mean([.5*(records[s]['linked_eog'][i]['values'][key]-records[s]['linked_basic'][i]['values'][key]+records[s]['average_eog'][i]['values'][key]-records[s]['average_basic'][i]['values'][key]) for s in seeds]) for i in ids]);per[key]=boot.mean_ci(values)
  result[title]=per
 if 'controls' in spec:
  control_results={};baseline=records['1109']
  for name,arms in spec['controls'].items():
   controls={arm:read_records(path,a.method) for arm,path in arms.items()};ranks=[];cosines=[]
   for i in ids:
    rr=[];cc=[]
    for arm,rows in controls.items():
     u=baseline[arm][i];v=rows[i];aa=u['values']['signed_roi_scores'];bb=v['values']['signed_roi_scores'];rr.append(float(spearmanr(list(aa.values()),[bb[k] for k in aa]).statistic))
     with np.load(u['map']) as z:x=z['attribution'].astype('float64').ravel()
     with np.load(v['map']) as z:b=z['attribution'].astype('float64').ravel()
     cc.append(float(np.abs(x)@np.abs(b)/(np.linalg.norm(x)*np.linalg.norm(b))))
    ranks.append(np.mean(rr));cosines.append(np.mean(cc))
   control_results[name]={'signed_regional_rank':boot.mean_ci(np.array(ranks)),'absolute_map_cosine':boot.mean_ci(np.array(cosines))}
  result['randomization']=control_results
 Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
