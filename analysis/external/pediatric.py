"""Native-rate pediatric common-component power from a verified recording manifest."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter,sosfiltfilt,welch
from power import features,valid_windows,summarize,serial,require
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(1048576),b''):h.update(b)
 return h.hexdigest()
def fft_check(w,fs):
 n=int(fs*2);hop=int(fs);h=0.5-0.5*np.cos(2*np.pi*np.arange(n)/n)
 segments=np.stack([w[...,s:s+n] for s in (0,hop,2*hop)],axis=-2)
 segments-=segments.mean(axis=-1,keepdims=True)
 p=np.abs(np.fft.rfft(segments*h,axis=-1))**2/(fs*np.sum(h*h));p[...,1:-1]*=2;p=p.mean(axis=-2)
 f,q=welch(w,fs=fs,window='hann',nperseg=n,noverlap=hop,detrend='constant',scaling='density',axis=-1)
 err=float(np.max(np.abs(p-q)/np.maximum(q,np.finfo(float).tiny)))
 mask=(f>=1)&(f<=40)
 require(np.allclose(p[...,mask],q[...,mask],rtol=1e-9,atol=0),'In-band FFT failed')
 require(np.max(np.abs(p-q)) <= 1e-12*np.max(q),'Global FFT absolute error failed');return err

def load_record(row):
 p=Path(row['path']);require(sha(p)==row['sha256'],'Input hash mismatch')
 require(row['format']=='mat','Expected a released Nasrabadi MAT record')
 vals=[v for k,v in loadmat(p).items() if not k.startswith('__')]
 require(len(vals)==1 and list(vals[0].shape)==row['shape'],'MAT shape changed')
 x=np.asarray(vals[0].T,dtype=float);fs=row['fs']
 require(x.shape[0]==19 and x.shape[1]>=int(row['stop']*fs),'Insufficient samples')
 require(np.isfinite(x).all(),'Nonfinite raw signal')
 return x,fs
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();rows=json.loads(Path(a.manifest).read_text(encoding='utf-8-sig'));records=[];excluded=[];checks=[]
 for i,row in enumerate(rows):
  x,fs=load_record(row);intervals,rejected,nzero=valid_windows(x,fs,row['start'],row['stop'])
  if len(intervals)<10:excluded.append(i);continue
  z=sosfiltfilt(butter(4,[1,40],btype='bandpass',fs=fs,output='sos'),x,axis=-1);w=np.stack([z[:,u:v] for u,v in intervals]);w-=w.mean(axis=-1,keepdims=True);feat,pc,pr,verify=features(w,fs)
  if not records:
   verify['explicit_fft_max_relative_error']=fft_check(w[0],fs)
   for gain in (1e-3,1e3):
    changed,*_=features(w*gain,fs);require(np.allclose(changed-feat,2*np.log10(gain),atol=1e-10,rtol=0),'Gain scaling failed')
  records.append({'label':row['label'],'features':feat,'window_common':pc,'window_car':pr,'n_windows':len(intervals)});checks.append(verify)
 require(len(records)>0,'No eligible participants');require(np.median([c['common_fraction'] for c in checks])>1e-10,'Common numerically absent: inference stopped');y=np.array([r['label'] for r in records]);require(set(y.tolist())=={0,1} and min(np.bincount(y))>=2,'Group too small');f=np.array([r['features'] for r in records]);result,_=summarize(f,y)
 z=np.array([np.r_[np.log(r['window_common']).mean(axis=0),np.log(r['window_car']).mean(axis=0)] for r in records]);delta=z[y==1].mean(axis=0)-z[y==0].mean(axis=0);direct=np.exp(np.r_[delta,delta[5]-delta[11]]);expected=np.array([v['ratio'] for v in result['estimands'].values()]);require(np.allclose(direct,expected,rtol=1e-12,atol=0),'Independent log reconstruction failed');result['leave_one_out'].pop('all_omissions')
 out={'n':len(y),'group_sizes':[int((y==0).sum()),int((y==1).sum())],'excluded_n':len(excluded),'retained_windows_range':[min(r['n_windows'] for r in records),max(r['n_windows'] for r in records)],'analysis':result,'verification':{'max_psd_decomposition_relative_error':max(c['psd_decomposition_relative_error'] for c in checks),'first_record':checks[0],'natural_log_ratio_max_relative_error':float(np.max(np.abs(direct-expected)/expected))}}
 Path(a.output).write_text(json.dumps(serial(out),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
