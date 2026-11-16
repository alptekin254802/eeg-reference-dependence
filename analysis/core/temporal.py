"""Prepare the reported nonoverlapping time intervals for held-out inference."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import argparse,json,hashlib
from pathlib import Path
CACHE=None
OUT=None
DATA=None
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2))
def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()
os.environ['MNE_LOGGING_LEVEL']='ERROR'
import csv
import numpy as np
import mne
from scipy.signal import butter,sosfiltfilt,resample_poly

def main():
    target=OUT;target.mkdir(parents=True,exist_ok=True)
    complete=target/'COMPLETE.json';codehash=sha(Path(__file__))
    if complete.exists():
        assert read(complete)['code_sha256']==codehash;return
    data=DATA
    rows=read(CACHE/'cohort.json');manifest=read(CACHE/'manifest.json')
    inventory=list(csv.DictReader(INVENTORY.open(encoding='utf-8-sig')))
    lookup={(r['participant_id'],r['session'],r['task']):r for r in inventory}
    source={r['participant_id']:r for r in read(CACHE/'source_hashes.json')}
    arrays={}
    for interval in ['early','late']:
        for arm in manifest['arms']:
            path=target/f'{interval}_{arm}.npy'
            arrays[interval,arm]=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(rows),13,19,800))
            arrays[interval,arm][:]=np.nan
    sos=butter(4,[1,40],fs=500,btype='bandpass',output='sos');quality=[];eligible=[]
    for i,row in enumerate(rows):
        path=data/lookup[row['participant_id'],row['selected_session'],'restEC']['filename']
        assert sha(path)==source[row['participant_id']]['bdf_sha256']
        raw=mne.io.read_raw_bdf(path,preload=False,verbose='ERROR')
        entry={'person_index':i,'duration_seconds':raw.n_times/raw.info['sfreq']}
        if raw.info['sfreq']!=500 or raw.n_times<58000:
            entry['eligible']=False;entry['reason']='Insufficient duration or sampling mismatch';quality.append(entry);continue
        signal=raw.get_data(picks=manifest['channels']+['VPVA','VNVB','HPHL','HNHR'])*1e6
        signal=resample_poly(sosfiltfilt(sos,signal,axis=-1),2,5,axis=-1)
        values={};good=True
        for interval,begin,end in [('early',4,56),('late',64,116)]:
            segment=signal[:,begin*200:end*200]
            assert segment.shape==(23,10400)
            eeg=segment[:19].copy();eog=np.stack([segment[19]-segment[20],segment[21]-segment[22]])
            eeg-=eeg.mean(-1,keepdims=True);eog-=eog.mean(-1,keepdims=True)
            windows=eeg.reshape(19,13,800).transpose(1,0,2);ptp=np.ptp(windows,axis=-1)
            bad=not np.isfinite(windows).all() or bool((windows.std(-1)<.1).any()) or bool((ptp>1000).any())
            good=good and not bad
            entry[interval]={'gross_failure':bad,'max_window_ptp_uv':float(ptp.max())}
            corrected=eeg-np.linalg.lstsq(eog.T,eeg.T,rcond=1e-8)[0].T@eog
            values[interval]={'linked_basic':eeg,'average_basic':eeg-eeg.mean(0,keepdims=True),
                'linked_eog':corrected,'average_eog':corrected-corrected.mean(0,keepdims=True)}
        entry['eligible']=good;quality.append(entry)
        if good:
            eligible.append(i)
            for interval in values:
                for arm,v in values[interval].items():arrays[interval,arm][i]=v.reshape(19,13,800).transpose(1,0,2)
        if (i+1)%30==0:print(f'Temporal eligibility {i+1}/{len(rows)}',flush=True)
    for arr in arrays.values():arr.flush()
    y=np.load(CACHE/'labels.npy');counts=np.bincount(y[eligible],minlength=2)
    dump(complete,{'at':now(),'code_sha256':codehash,'eligible_ids':eligible,'n':len(eligible),
        'n_adhd':int(counts[1]),'n_control':int(counts[0]),'group_summaries_eligible':bool(min(counts)>=10),
        'quality':quality,'intervals_seconds':{'early':[4,56],'late':[64,116]},
        'hashes':{f'{a}_{b}':sha(target/f'{a}_{b}.npy') for a,b in arrays}})
    print(f'Temporal intervals prepared for {len(eligible)} paired participants.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();CACHE,DATA,INVENTORY,OUT=a.input,a.data,a.inventory,a.output
    main()
