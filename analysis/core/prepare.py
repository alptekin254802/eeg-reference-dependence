"""Label-independent signal preparation; all four arms share exact input times."""
import os
from pathlib import Path
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'
import argparse, csv, json, hashlib, time
from collections import Counter
import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt, resample_poly, welch
import mne
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

DATA = None
ROSTER = None
INVENTORY = None
OUTPUT = None
CHANNELS = 'Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2'.split()
ARMS = ['linked_basic', 'average_basic', 'linked_eog', 'average_eog']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def main():
    t0 = time.time()
    out = OUTPUT; out.mkdir(parents=True,exist_ok=True)
    rows = list(csv.DictReader(ROSTER.open(encoding='utf-8-sig', newline='')))
    inv = list(csv.DictReader(INVENTORY.open(encoding='utf-8-sig', newline='')))
    idx = {(r['participant_id'], r['session'], r['task']): r for r in inv}
    all_x = {a: [] for a in ARMS}; kept = []; qc = []; excluded = []
    references = Counter(); hashes = []
    band = butter(4, [1., 40.], fs=500., btype='bandpass', output='sos')
    for j, row in enumerate(rows):
        key = (row['participant_id'], row['selected_session'], 'restEC')
        rec = idx[key]; path = DATA / rec['filename']
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))
        reference = meta['EEGReference']; references[reference] += 1
        if reference != 'linked mastoids (A1A2)':
            raise ValueError('Unexpected reference; protocol review needed')
        raw = mne.io.read_raw_bdf(path, preload=False, verbose='ERROR')
        assert raw.info['sfreq'] == 500
        wanted = CHANNELS + ['VPVA','VNVB','HPHL','HNHR']
        signal = raw.get_data(picks=wanted) * 1e6
        # Filtering continuous recordings before taking 4-64 s avoids epoch-edge filters.
        signal = resample_poly(sosfiltfilt(band, signal, axis=-1), 2, 5, axis=-1)
        segment = signal[:, 800:12800]
        assert segment.shape == (23,12000)
        eeg = segment[:19].copy()
        eog = np.stack([segment[19]-segment[20], segment[21]-segment[22]])
        eeg -= eeg.mean(-1,keepdims=True); eog -= eog.mean(-1,keepdims=True)
        win = eeg.reshape(19,15,800).transpose(1,0,2)
        peak = np.ptp(win,axis=-1)
        flat = win.std(axis=-1) < .1
        # Predefined gross failure exclusion only; no class-specific or arm-specific rejection.
        gross = (~np.isfinite(eeg).all()) or flat.any() or (peak > 1000).any()
        metrics = {'participant_id': row['participant_id'], 'max_window_ptp_uv': float(peak.max()),
                   'fraction_channel_windows_ptp_gt_200': float((peak>200).mean()),
                   'median_channel_rms_uv': float(np.median(eeg.std(-1))),
                   'veog_rms_uv': float(eog[0].std()), 'heog_rms_uv': float(eog[1].std()),
                   'max_eeg_eog_abs_corr': float(np.abs(np.corrcoef(eeg,eog)[:19,19:]).max()),
                   'gross_failure': bool(gross)}
        qc.append(metrics)
        hashes.append({'participant_id':row['participant_id'], 'bdf_sha256':sha(path),
                       'json_sha256':sha(path.with_suffix('.json'))})
        if gross:
            excluded.append(row['participant_id']); continue
        # Per-recording, label-free least squares.
        beta = np.linalg.lstsq(eog.T, eeg.T, rcond=1e-8)[0]
        corrected = eeg - beta.T @ eog
        # Re-reference after correction; linear operations commute up to floating point.
        arms = {'linked_basic':eeg, 'average_basic':eeg-eeg.mean(0,keepdims=True),
                'linked_eog':corrected, 'average_eog':corrected-corrected.mean(0,keepdims=True)}
        for arm, values in arms.items():
            windows = values.reshape(19,15,800).transpose(1,0,2).astype('float32')
            assert np.isfinite(windows).all()
            all_x[arm].append(windows)
        kept.append(row)
        if (j+1)%20==0: print(f'Prepared {j+1}/{len(rows)}; kept {len(kept)}',flush=True)
    y = np.array([int(r['formal_status']=='ADHD') for r in kept],dtype='int64')
    assert len(set(r['participant_id'] for r in kept))==len(kept)
    np.save(out/'labels.npy',y)
    for arm, values in all_x.items():
        arr = np.stack(values); np.save(out/f'{arm}.npy',arr)
        freq, psd = welch(arr,fs=200,nperseg=400,axis=-1)
        features = []
        for low, high in [(1,4),(4,8),(8,13),(13,30),(30,40.01)]:
            p = psd[..., (freq>=low)&(freq<high)].sum(-1)*.5
            features.append(np.log10(np.maximum(p,1e-12)).mean(1))
        np.save(out/f'{arm}_spectral.npy',np.concatenate(features,axis=1))
    folds = []
    for k,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261109).split(y,y)):
        inner_train, inner_val = next(StratifiedShuffleSplit(1,test_size=.2,random_state=741+k).split(train,y[train]))
        folds.append({'fold':k,'train':train.tolist(),'test':test.tolist(),
                      'inner_train':train[inner_train].tolist(),'inner_val':train[inner_val].tolist()})
        assert not set(train)&set(test)
        assert set(train[inner_train])|set(train[inner_val])==set(train)
    assert sorted(i for f in folds for i in f['test'])==list(range(len(y)))
    dump(out/'folds.json',folds); dump(out/'cohort.json',kept)
    dump(out/'quality.json',qc); dump(out/'source_hashes.json',hashes)
    # Quality-only baseline uses only acquired-reference, uncorrected signals.
    byid={q['participant_id']:q for q in qc}
    names=['max_window_ptp_uv','fraction_channel_windows_ptp_gt_200','median_channel_rms_uv',
           'veog_rms_uv','heog_rms_uv','max_eeg_eog_abs_corr']
    np.save(out/'quality_features.npy',np.array([[byid[r['participant_id']][n] for n in names] for r in kept]))
    dump(out/'manifest.json',{'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'n_candidate':len(rows),'n_kept':len(y),'n_adhd':int(y.sum()),'n_control':int((1-y).sum()),
        'excluded_count':len(excluded),'reference_counts':dict(references),'channels':CHANNELS,'arms':ARMS,
        'shape':[len(y),15,19,800],'sample_rate':200,'quality_features':names,
        'duration_sec':time.time()-t0,'mne':mne.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
        'roster_sha256':sha(ROSTER),
        'folds_sha256':sha(out/'folds.json'),'code_sha256':sha(Path(__file__))})
    paths = [str((DATA / idx[(r['participant_id'], r['selected_session'], 'restEC')]['filename']).resolve()) for r in kept]
    dump(out/'ordered_bdf_paths.json',paths)
    np.save(out/'archive_groups.npy', np.array(['annotated' if r['dataset_membership'] else 'unannotated' for r in kept]))
    print(json.dumps(json.loads((out/'manifest.json').read_text()),indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--roster',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    DATA,ROSTER,INVENTORY,OUTPUT=args.data,args.roster,args.inventory,args.output
    main()
