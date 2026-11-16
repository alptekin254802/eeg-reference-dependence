"""Enumerate BDF structural support without reading signal samples."""
import argparse,csv,re,math
from pathlib import Path
import mne

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[]
    pattern=re.compile(r'(?P<participant_id>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)_eeg\.bdf$',re.I)
    for path in sorted(a.data.rglob('*.bdf')):
        match=pattern.search(path.name)
        if not match:raise ValueError('BDF filename does not match the released naming pattern: '+path.name)
        row={**match.groupdict(),'filename':path.relative_to(a.data).as_posix(),'structurally_usable':False,'sfreq_hz':'','duration_s':'','physical_channel_names':'','error':''}
        try:
            raw=mne.io.read_raw_bdf(path,preload=False,verbose='ERROR');sf=float(raw.info['sfreq']);duration=raw.n_times/sf
            row.update(sfreq_hz=sf,duration_s=duration,physical_channel_names='|'.join(c for c in raw.ch_names if 'status' not in c.casefold()))
            row['structurally_usable']=bool(path.stat().st_size>0 and math.isfinite(sf) and sf>0 and raw.n_times>0 and math.isfinite(duration) and duration>1 and len(raw.ch_names)>0)
            raw.close()
        except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
        rows.append(row)
    if not rows:raise ValueError('No BDF recordings found')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('Inventoried',len(rows),'recordings')
if __name__=='__main__':main()
