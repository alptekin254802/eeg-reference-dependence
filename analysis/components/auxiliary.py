"""Fixed auxiliary Erbs/MASS measurements from ordered 500-Hz BDF recordings."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def clean(v):
 if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
 if isinstance(v,np.generic): return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v): return None
 return v
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+"\n",encoding="utf-8")
import os
os.environ['MNE_DONTWRITE_HOME']='true'
import mne
from mne.preprocessing.ecg import qrs_detector
from scipy.signal import welch
mne.set_log_level('ERROR')
def match_peaks(a, b, tolerance):
    """Maximum-cardinality chronological one-to-one match in a fixed tolerance."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return matched

def in_window(events, fs, lo, hi):
    return events[(events >= round(lo * fs)) & (events < round(hi * fs))]

def interval_metrics(events, sensitivity, fs, lo, hi):
    p, s = in_window(events, fs, lo, hi), in_window(sensitivity, fs, lo, hi)
    rr = np.diff(p) / fs
    hr = 60 / np.median(rr) if len(rr) else np.nan
    rate = len(p) * 60 / (hi - lo)
    matched = match_peaks(p, s, round(.05 * fs))
    return dict(event_count=len(p), sensitivity_event_count=len(s), matched_event_count=matched,
                heart_rate_bpm=float(hr), count_rate_bpm=rate,
                count_rr_relative_disagreement=float(abs(rate - hr) / hr) if np.isfinite(hr) and hr > 0 else np.nan,
                rr_in_range_fraction=float(np.mean((rr >= .375) & (rr <= 1.5))) if len(rr) else np.nan,
                rr_median_s=float(np.median(rr)) if len(rr) else np.nan,
                rr_iqr_s=float(np.ptp(np.quantile(rr, [.25, .75]))) if len(rr) else np.nan,
                beat_quality_f1=2 * matched / (len(p) + len(s)) if len(p) + len(s) else 0.)

def mass_feature(x_uv, fs):
    assert fs == 500
    assert len(x_uv) == 30000
    windows = x_uv.reshape(15, 2000)
    f, p = welch(windows, fs=fs, window='hann', nperseg=1000, noverlap=500,
                 detrend='constant', return_onesided=True, scaling='density', axis=-1)
    powers = np.sum(p[:, (f >= 60) & (f <= 90)], axis=-1) * (f[1] - f[0])
    logs = np.log10(powers) if np.all(powers > 0) else np.full(len(powers), np.nan)
    return float(np.mean(logs)), powers, f, np.mean(p, axis=0)

def filtered_ecg(x, fs):
    return mne.filter.filter_data(x, fs, 5, 35, filter_length='10s', l_trans_bandwidth=.5,
                                h_trans_bandwidth=.5, phase='zero-double', fir_window='hann',
                                fir_design='firwin2', verbose='ERROR')

def templates(x, events, fs, lo, hi):
    before, after = round(.2 * fs), round(.4 * fs)
    p = in_window(events, fs, lo, hi)
    p = p[(p - before >= round(lo * fs)) & (p + after < round(hi * fs))]
    if not len(p):
        return None, 0
    epochs = np.stack([x[k-before:k+after+1] for k in p])
    epochs -= epochs[:, :round(.1 * fs)].mean(axis=1, keepdims=True)
    return epochs.mean(axis=0), len(p)

def correlation(a, b):
    if a is None or b is None or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def read_record(path):
    raw = mne.io.read_raw_bdf(path, preload=False, verbose='ERROR')
    names = {s.casefold(): s for s in raw.ch_names}
    assert 'erbs' in names and 'mass' in names
    fs = float(raw.info['sfreq'])
    assert fs == 500
    channels = [names['erbs'], names['mass']]
    values = raw.get_data(picks=channels) * 1e6
    units = raw._orig_units
    assert all(units[c].replace('µ','u').replace('μ','u').lower() == 'uv' for c in channels), 'unexpected source units'
    return fs, values, channels, [units[c] for c in channels]

def analyze_record(index, path):
    fs, (ecg, mass), names, units = read_record(path)
    reasons = []
    primary_mass = mass[2000:32000]
    finite_ecg = bool(np.all(np.isfinite(ecg)))
    finite_mass = bool(len(primary_mass) == 30000 and np.all(np.isfinite(primary_mass)))
    flat_ecg = bool(np.ptp(ecg) == 0) if finite_ecg else False
    flat_mass = bool(np.ptp(primary_mass) == 0) if finite_mass else False
    if not finite_ecg: reasons.append('Erbs_nonfinite')
    if not finite_mass: reasons.append('MASS_nonfinite_or_incomplete')
    if flat_ecg: reasons.append('Erbs_flat')
    if flat_mass: reasons.append('MASS_flat')
    events, sensitivity = np.array([],dtype=int), np.array([],dtype=int)
    if finite_ecg and not flat_ecg:
        events = qrs_detector(fs, ecg, thresh_value=.6, l_freq=5,h_freq=35,filter_length='10s',verbose='ERROR').astype(int)
        sensitivity = qrs_detector(fs, ecg, thresh_value=.4, l_freq=5,h_freq=35,filter_length='10s',verbose='ERROR').astype(int)
    primary = interval_metrics(events,sensitivity,fs,4,64)
    late = interval_metrics(events,sensitivity,fs,64,116)
    halves = [interval_metrics(events,sensitivity,fs,64,90), interval_metrics(events,sensitivity,fs,90,116)]
    hr = primary['heart_rate_bpm']
    if primary['event_count'] < 30: reasons.append('fewer_than_30_primary_events')
    if not np.isfinite(hr) or not 40 <= hr <= 160: reasons.append('HR_outside_40_160_or_undefined')
    if not primary['rr_in_range_fraction'] >= .9: reasons.append('RR_in_range_fraction_below_0.90')
    if primary['beat_quality_f1'] < .8: reasons.append('detector_F1_below_0.80')
    if not primary['count_rr_relative_disagreement'] <= .15: reasons.append('count_RR_rate_disagreement_above_0.15')
    mass_value, powers = np.nan, np.array([])
    if finite_mass and not flat_mass:
        mass_value,powers,_,_ = mass_feature(primary_mass,fs)
    if not np.isfinite(mass_value): reasons.append('MASS_feature_nonfinite')
    filt = filtered_ecg(ecg,fs) if finite_ecg and not flat_ecg else None
    templ = [templates(filt,events,fs,a,b) for a,b in [(4,64),(64,90),(90,116)]] if filt is not None else [(None,0)]*3
    morphology = dict(template_counts=[n for _,n in templ],
                      template_peak_to_peak_uv=[float(np.ptp(x)) if x is not None else np.nan for x,_ in templ],
                      application_late_first_correlation=correlation(templ[0][0],templ[1][0]),
                      application_late_second_correlation=correlation(templ[0][0],templ[2][0]),
                      late_half_correlation=correlation(templ[1][0],templ[2][0]))
    row = dict(index=index, record_key=str(index), eligible=not reasons, heart_rate_bpm=hr,
               beat_quality_f1=primary['beat_quality_f1'],mass_log10_power_60_90=mass_value,
               exclusion_reasons=reasons, primary=primary, late_descriptive=late, late_halves_descriptive=halves,
               duration_seconds=len(ecg)/fs, channel_names=names, source_physical_units=units,
               morphology_descriptive=morphology, mass_block_power_uv2=powers.tolist(),
               erbs_primary_rms_uv=float(np.sqrt(np.mean(ecg[2000:32000]**2))),
               mass_primary_rms_uv=float(np.sqrt(np.mean(primary_mass**2))),
               erbs_filtered_event_positive_fraction=float(np.mean(filt[in_window(events,fs,4,64)]>0)) if filt is not None and primary['event_count'] else np.nan)
    return clean(row), dict(primary_events_sample=events.tolist(), sensitivity_events_sample=sensitivity.tolist())
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 paths=read(a.input);assert len(paths)==189;records=[];events={}
 for i,path in enumerate(paths):
  path=Path(path);path=path if path.is_absolute() else a.input.parent/path
  row,peaks=analyze_record(i,path);records.append(row);events[str(i)]=peaks
 dump(a.output/'features.json',{'records':records});dump(a.output/'events.json',events)
 dump(a.output/'eligible_indices.json',[r['index'] for r in records if r['eligible']])
if __name__=='__main__':main()
