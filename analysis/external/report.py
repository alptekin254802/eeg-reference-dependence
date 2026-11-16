"""Conditional external performance of unadapted source checkpoints."""
import argparse,json
from pathlib import Path
import numpy as np
from metrics import PersonBootstrap

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',required=True);p.add_argument('--labels',required=True);p.add_argument('--output',required=True);p.add_argument('--comparison',help='CAR prediction JSON for linked-minus-CAR AUC');a=p.parse_args();z=json.loads(Path(a.predictions).read_text())['probabilities'];y=np.load(a.labels,allow_pickle=False);b=PersonBootstrap(y);result={}
 for name,values in z.items():
  v=np.asarray(values);row=b.evaluate(v);row['sensitivity']=float((v[y==1]>=.5).mean());row['specificity']=float((v[y==0]<.5).mean());result[name]=row
 if a.comparison:
  comparison=json.loads(Path(a.comparison).read_text())['probabilities']
  result['reference_contrasts']={name:b.auc_difference(np.asarray(values),np.asarray(comparison[name])) for name,values in z.items()}
 Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
