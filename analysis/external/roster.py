"""Build external input manifests from released metadata."""
import argparse,csv,json,zipfile,hashlib
from pathlib import Path
from scipy.io import loadmat
from prepare import sha

def tsv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def save(path,rows):Path(path).write_text(json.dumps(rows,indent=2)+'\n')

def wroclaw(archive,out):
 out.mkdir(parents=True,exist_ok=True);conversion=[];cohort=[]
 with zipfile.ZipFile(archive) as z:
  names=sorted(n for n in z.namelist() if n.endswith('/Session 1/2_ClosedEyes_Raw.mat'));assert len(names)==16
  for i,name in enumerate(names):
   blob=z.read(name);native=out/f'recording{i:03d}.mat';native.write_bytes(blob);converted=out/f'converted{i:03d}.mat';parts=name.split('/');assert parts[1] in ['ADHD','Control','Controls','CONTROL','control','Healthy']
   conversion.append({'input_path':str(native.resolve()),'output_path':str(converted.resolve())});cohort.append({'path':str(converted.resolve()),'input_path':str(native.resolve()),'input_sha256':hashlib.sha256(blob).hexdigest(),'label':int(parts[1]=='ADHD')})
 save(out/'conversion.json',conversion);save(out/'cohort.json',cohort)

def nasrabadi(adhd,control,out):
 rows=[]
 for label,folder in [(1,adhd),(0,control)]:
  for path in sorted(folder.rglob('*.mat')):
   data=loadmat(path);v=[v for k,v in data.items() if not k.startswith('__')];assert len(v)==1 and v[0].ndim==2 and v[0].shape[1]==19
   rows.append({'path':str(path.resolve()),'sha256':sha(path),'label':label,'format':'mat','shape':list(v[0].shape),'fs':128,'start':4,'stop':60})
 assert len(rows)==121 and sum(r['label'] for r in rows)==61,'Expected released 61 ADHD / 60 control roster';save(out,rows)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',required=True,choices=['wroclaw','wroclaw-converted','nasrabadi']);p.add_argument('--input',type=Path);p.add_argument('--adhd',type=Path);p.add_argument('--control',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.kind=='wroclaw':wroclaw(a.input,a.output)
 elif a.kind=='wroclaw-converted':
  rows=json.loads((a.input/'cohort.json').read_text())
  for row in rows:assert sha(row['input_path'])==row['input_sha256'];row['sha256']=sha(row['path'])
  save(a.output,rows)
 elif a.kind=='nasrabadi':nasrabadi(a.adhd,a.control,a.output)
if __name__=='__main__':main()
