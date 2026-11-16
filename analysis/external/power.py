"""Common-component and CAR power ratios with participant bootstrap intervals."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import welch
BANDS = [("delta",1,4),("theta",4,8),("alpha",8,13),("beta",13,30),("gamma",30,40),("total_1_40",1,40)]
NAMES = [c+"_"+b[0] for c in ("common","car") for b in BANDS]
TOL=2e-5
def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def psd(x):
    return welch(np.asarray(x, dtype=np.float64), fs=200, window='hann',
                 nperseg=400, noverlap=200, detrend='constant',
                 return_onesided=True, scaling='density', axis=-1, average='mean')

def bandpower(x):
    frequencies, density = psd(x)
    powers = []
    for name, lo, hi in BANDS:
        mask = (frequencies >= lo) & ((frequencies <= hi)
                if name in ('gamma', 'total_1_40') else (frequencies < hi))
        powers.append(density[..., mask].sum(axis=-1) * 0.5)
    result = np.stack(powers, axis=-1)
    require(np.isfinite(result).all() and (result > 0).all(),
            'NOT ESTIMABLE: nonpositive or nonfinite native-unit power; no floor permitted')
    return result

def extract(linked, car):
    """Float64 arithmetic throughout; per-window channel power averaged before log."""
    x = np.asarray(linked, dtype=np.float64)
    r = np.asarray(car, dtype=np.float64)
    require(x.shape == r.shape and x.ndim == 4 and x.shape[2:] == (19, 800),
            'Unexpected extraction shape')
    require(np.isfinite(x).all() and np.isfinite(r).all(), 'Nonfinite waveform')
    common = x.mean(axis=2, dtype=np.float64)
    pc = bandpower(common)
    pr = bandpower(r).mean(axis=2, dtype=np.float64)
    return np.concatenate([np.log10(pc).mean(axis=1),
                           np.log10(pr).mean(axis=1)], axis=1), pc, pr

def contrast(features, labels):
    return features[labels == 1].mean(axis=0) - features[labels == 0].mean(axis=0)

def augmented(features):
    # Final column is the within-person common/CAR total log contrast.
    return np.column_stack([features, features[:, 5] - features[:, 11]])

def fft_check(z):
    z = np.asarray(z, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
    segments = np.stack([z[..., start:start + 400] for start in (0, 200, 400)], axis=-2)
    centered = segments - segments.mean(axis=-1, keepdims=True)
    raw = np.abs(np.fft.rfft(centered * hann, axis=-1)) ** 2
    raw /= 200 * np.sum(hann * hann)
    raw[..., 1:-1] *= 2
    manual = raw.mean(axis=-2)
    freq, expected = psd(z)
    require(np.allclose(manual, expected, rtol=1e-10, atol=0),
            'Independent explicit FFT Welch check failed')
    return {'participant_index': 0, 'window_index': 0,
            'max_relative_psd_error': float(np.max(np.abs(manual - expected) / expected)),
            'rtol': 1e-10, 'atol': 0, 'frequency_step_hz': float(freq[1] - freq[0])}

def numerical_checks(linked, car, labels, features, pc, pr):
    x = np.asarray(linked, dtype=np.float64)
    r = np.asarray(car, dtype=np.float64)
    common = x.mean(axis=2)
    scale = float(np.max(np.abs(x)))
    require(scale > 0, 'Zero linked waveform')
    wave_error = float(np.max(np.abs(r - (x - common[:, :, None, :]))) / scale)
    pl = bandpower(x).mean(axis=2)
    spectral_error = float(np.max(np.abs(pl - pc - pr) / pl))
    require(wave_error <= TOL and spectral_error <= TOL,
            'Cached CAR or PSD decomposition outside locked relative tolerance')
    fft = fft_check(x[0, 0])
    gains = {}
    for gain in (1e-3, 1e3):
        f, _, _ = extract(x * gain, r * gain)
        shift_error = float(np.max(np.abs(f - features - 2 * np.log10(gain))))
        delta_error = float(np.max(np.abs(contrast(f, labels) - contrast(features, labels))))
        require(shift_error < 1e-10 and delta_error < 1e-10,
                'Global waveform gain equivariance/invariance failed')
        # Checks all twelve ratios, rather than treating absolute log power as invariant.
        require(np.allclose(10 ** contrast(f, labels), 10 ** contrast(features, labels),
                            rtol=1e-10, atol=0), 'Global gain group ratio changed')
        gains[str(gain)] = {'max_log_feature_shift_error': shift_error,
                            'max_group_logratio_error_all12': delta_error}
    person_gains = np.linspace(0.5, 2.0, len(x))
    scaled, _, _ = extract(x * person_gains[:, None, None, None],
                           r * person_gains[:, None, None, None])
    shift = 2 * np.log10(person_gains)[:, None]
    person_error = float(np.max(np.abs((scaled - features) - shift)))
    ror_error = float(np.max(np.abs(augmented(scaled)[:, -1] - augmented(features)[:, -1])))
    require(person_error < 1e-10 and ror_error < 1e-10,
            'Per-recording scalar gain cancellation failed')
    # The TDBRAIN calculation took the channel mean in float32 before Welch.
    float32_common = np.asarray(linked, dtype=np.float32).mean(axis=2, dtype=np.float32)
    float32_common_features = np.log10(bandpower(float32_common)).mean(axis=1)
    comparison = float32_common_features[:, 5] - features[:, 5]
    return {'waveform_max_absolute_error_div_global_max_linked': wave_error,
            'psd_decomposition_max_relative_error': spectral_error,
            'cache_tolerance': TOL, 'independent_fft_welch': fft,
            'global_gain_all12': gains,
            'per_person_gain': {'max_expected_log_shift_error': person_error,
                'max_within_person_common_car_logcontrast_error': ror_error,
                'absolute_group_logratio_change': contrast(scaled, labels) - contrast(features, labels)},
            'float32_common_comparison': {'max_absolute_primary_feature_log10_difference':
                float(np.max(np.abs(comparison))), 'primary_ratio_float32_common':
                float(10 ** contrast(float32_common_features, labels)[5]),
                'primary_ratio_float64_common': float(10 ** contrast(features, labels)[5]),
                'note': 'Only spatial-mean arithmetic uses float32 in compatibility comparator; Welch remains explicit float64.'}}

def summarize(features, labels):
    x = augmented(features)
    names = NAMES + ['common_over_car_total_ratio_of_ratios']
    controls, adhd = x[labels == 0], x[labels == 1]
    rng = np.random.default_rng(20261116)
    # These explicit participant index draws are shared across every estimand.
    draws0 = rng.integers(0, len(controls), size=(20000, len(controls)))
    draws1 = rng.integers(0, len(adhd), size=(20000, len(adhd)))
    boot = adhd[draws1].mean(axis=1) - controls[draws0].mean(axis=1)
    first_checks = []
    for k in range(25):
        # Independently accumulate explicit participant rows in Python, avoiding
        # reuse of the vectorized bootstrap reduction implementation.
        direct = np.array([sum(float(adhd[j, col]) for j in draws1[k]) / len(adhd)
                           - sum(float(controls[j, col]) for j in draws0[k]) / len(controls)
                           for col in range(x.shape[1])])
        error = float(np.max(np.abs(direct - boot[k])))
        require(error < 1e-12, 'Explicit-row bootstrap verification failed')
        first_checks.append({'replicate_zero_based': k, 'control_indices_within_group': draws0[k],
            'adhd_indices_within_group': draws1[k], 'direct_logratio': direct,
            'vectorized_logratio': boot[k], 'max_abs_error': error})
    delta = contrast(x, labels)
    ci = np.quantile(boot, [0.025, 0.975], axis=0, method='linear')
    rows = {}
    for j, name in enumerate(names):
        rows[name] = {'control_mean_log10': controls[:, j].mean(),
                      'adhd_mean_log10': adhd[:, j].mean(),
                      'adhd_minus_control_log10': delta[j],
                      'ratio': 10 ** delta[j], 'ratio_ci95_pointwise': 10 ** ci[:, j]}
    omissions = []
    for omitted in range(len(x)):
        keep = np.arange(len(x)) != omitted
        omissions.append({'omitted_participant_index': omitted,
                          'omitted_diagnosis': int(labels[omitted]),
                          'primary_ratio': float(10 ** contrast(x[keep], labels[keep])[5])})
    values = [v['primary_ratio'] for v in omissions]
    return {'estimands': rows,
            'leave_one_out': {'range': [min(values), max(values)], 'all_omissions': omissions},
            'bootstrap': {'replicates': 20000, 'seed': 20261116,
                'unit': 'participant', 'group_sizes': [len(controls), len(adhd)],
                'method': 'diagnosis-stratified explicit index draws; same draws all13 estimands; percentile linear quantile',
                'first25_explicit_row_max_error': max(v['max_abs_error'] for v in first_checks)}}, first_checks
def power(x,fs):
 f,p=welch(x,fs=fs,window='hann',nperseg=int(2*fs),noverlap=int(fs),detrend='constant',scaling='density',average='mean',axis=-1)
 out=[]
 for name,lo,hi in BANDS:
  mask=(f>=lo)&((f<=hi) if name in ['gamma','total_1_40'] else (f<hi))
  out.append(p[...,mask].sum(axis=-1)*0.5)
 return np.stack(out,axis=-1)

def features(w,fs):
 c=w.mean(axis=1);r=w-c[:,None,:]
 pc=power(c,fs);pr=power(r,fs).mean(axis=1);px=power(w,fs).mean(axis=1)
 require(np.isfinite(pc).all() and np.isfinite(pr).all() and (pc>0).all() and (pr>0).all(),'Nonpositive/nonfinite power: no floor')
 error=float(np.max(np.abs(px-pc-pr)/px));require(error<1e-10,'PSD decomposition failed')
 feat=np.concatenate([np.log10(pc).mean(axis=0),np.log10(pr).mean(axis=0)])
 return feat,pc,pr,{'psd_decomposition_relative_error':error,'common_fraction':float(np.mean(pc[:,-1]/px[:,-1]))}

def valid_windows(x,fs,start,stop):
 zero=np.all(x==0,axis=0)
 edges=np.diff(np.r_[False,zero,False].astype(int));runs=[(a,b) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=2]
 keep=[];rejected=[]
 for i,t in enumerate(range(int(start*fs),int(stop*fs),int(4*fs))):
  end=t+int(4*fs);reason=[]
  if any(t<b+fs and end>a-fs for a,b in runs):reason.append('all-channel zero run with 1s guard')
  if np.any(np.ptp(x[:,t:end],axis=1)==0):reason.append('constant raw channel')
  if reason:rejected.append({'window':i,'reason':reason})
  else:keep.append((t,end))
 return keep,rejected,len(runs)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--linked',required=True);p.add_argument('--labels',required=True);p.add_argument('--car');p.add_argument('--fs',type=float,default=200);p.add_argument('--output',required=True);a=p.parse_args()
 x=np.load(a.linked,allow_pickle=False);y=np.load(a.labels,allow_pickle=False)
 require(x.ndim==4 and x.shape[0]==len(y) and x.shape[2]==19,'Expected N x W x 19 x T')
 require(set(y.tolist())=={0,1} and min(np.bincount(y))>=2,'At least two people per group required')
 if a.car:
  require(a.fs==200,'Cached CAR extraction requires 200 Hz');r=np.load(a.car,allow_pickle=False);f,pc,pr=extract(x,r);checks=numerical_checks(x,r,y,f,pc,pr)
 else:
  rows=[features(np.asarray(w,dtype=float),a.fs) for w in x];f=np.array([r[0] for r in rows]);checks={'max_psd_decomposition_relative_error':max(r[3]['psd_decomposition_relative_error'] for r in rows)}
 result,_=summarize(f,y)
 # Omission identities are not needed in the aggregate report.
 result['leave_one_out'].pop('all_omissions')
 Path(a.output).write_text(json.dumps(serial({'n':len(y),'analysis':result,'checks':checks}),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
