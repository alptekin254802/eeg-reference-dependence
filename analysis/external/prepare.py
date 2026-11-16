"""Prepare external and paired temporal waveforms from locally acquired records."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter,sosfiltfilt,resample_poly
CHANNELS='Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2'.split()
EOG=['VPVA','VNVB','HPHL','HNHR']

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def quality(w):
 return bool(not np.isfinite(w).all() or (w.std(-1)<.1).any() or (np.ptp(w,axis=-1)>1000).any())

def representations(segment,average_first=False):
 n=segment.shape[-1]//800;eeg=segment[:19].copy()
 if average_first:eeg-=eeg.mean(0,keepdims=True)
 eeg-=eeg.mean(-1,keepdims=True);out={'linked_basic':eeg,'average_basic':eeg if average_first else eeg-eeg.mean(0,keepdims=True)}
 if len(segment)>19:
  eog=np.stack([segment[19]-segment[20],segment[21]-segment[22]]);eog-=eog.mean(-1,keepdims=True)
  corrected=eeg-np.linalg.lstsq(eog.T,eeg.T,rcond=1e-8)[0].T@eog
  out.update(linked_eog=corrected,average_eog=corrected if average_first else corrected-corrected.mean(0,keepdims=True))
 w={k:v.reshape(19,n,800).transpose(1,0,2).astype('float32') for k,v in out.items()}
 # QC is computed before float32 conversion and before ocular regression.
 excluded=quality(eeg.reshape(19,n,800).transpose(1,0,2))
 return w,excluded

def wroclaw(row):
 d=loadmat(row['path'],simplify_cells=True);x=np.asarray(d['eeg'],dtype='float64');t=np.asarray(d['time_seconds']).ravel()
 assert x.shape==(len(t),19) and np.isfinite(x).all() and np.allclose(np.diff(t),.002,atol=1e-5,rtol=0) and abs(t[0])<1e-6 and t[-1]>=64
 expected=['EEG'+c+'_AA' for c in 'Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 O2'.split()]
 assert list(d['channel_names'])==expected
 units=d['metadata']['variable_units']
 def nonempty(v):
  if isinstance(v,str):return bool(v)
  return any(nonempty(u) for u in np.asarray(v).ravel()) if np.asarray(v).size else False
 assert not nonempty(units), 'Explicit units require documentation-led handling'
 z=resample_poly(sosfiltfilt(butter(4,[1,40],fs=500,btype='bandpass',output='sos'),x.T,axis=-1),2,5,axis=-1)[:,800:12800]
 assert z.shape==(19,12000)
 w,bad=representations(z);return {'main':w},bad,hashlib.sha256(x.tobytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['wroclaw'],required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();rows=json.loads(Path(a.manifest).read_text(encoding='utf-8-sig'));out=Path(a.output);out.mkdir(parents=True,exist_ok=True);arrays={};labels=[];kept=[];quality_rows=[];signals=[]
 for i,row in enumerate(rows):
  assert sha(row['path'])==row['sha256'],'Input hash mismatch'
  w,bad,signal_hash=globals()[a.kind](row);quality_rows.append({'index':i,'excluded':bad})
  if signal_hash:signals.append(signal_hash)
  if bad:continue
  labels.append(row['label']);kept.append(i)
  for interval,reps in w.items():
   for arm,x in reps.items():arrays.setdefault(interval+'_'+arm,[]).append(x)
 assert len(set(signals))==len(signals),'Duplicate waveforms require identity review'
 np.save(out/'labels.npy',np.asarray(labels,dtype='int64'))
 if all('age' in rows[i] for i in kept):np.save(out/'ages.npy',np.array([rows[i]['age'] for i in kept]))
 for name,values in arrays.items():np.save(out/(name+'.npy'),np.stack(values))
 (out/'manifest.json').write_text(json.dumps({'channels':CHANNELS,'kept_indices':kept,'qc':quality_rows,'n':len(labels),'kind':a.kind},indent=2)+'\n')
if __name__=='__main__':main()
