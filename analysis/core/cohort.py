from __future__ import annotations
import csv
from pathlib import Path
import pandas as pd
RELEASE="TDBRAIN V3.1"
def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.casefold().isin({"true", "1", "yes"})


def value(item):
    return "" if pd.isna(item) else item


def common(row: pd.Series) -> dict:
    return {
        "participant_id": row.participant_id,
        "formal_status": row.formal_status,
        "indication_values": row.indication_values,
        "dataset_membership": value(row.dataset_membership),
        "age_years": value(row.age),
        "sex": row.sex,
        "selected_session": value(row.selected_session),
        "selected_session_rule": row.selected_session_rule,
        "observed_session_count": row.observed_session_count,
        "repeated_recording_participant": bool(row.repeated_recording_participant),
        "later_sessions_excluded": bool(row.observed_session_count > 1),
        "primary_state": "restEC",
        "ec_duration_s": value(row.ec_duration_s),
        "sampling_rate_hz": value(row.sampling_rate_hz),
        "physical_channel_count": value(row.physical_channel_count),
        "common_19_channels_available": bool(row.common_19_channels_available),
        "primary_ec_60s_complete": bool(row.primary_ec_60s_complete),
        "release_version": RELEASE,
    }


def main():
    import argparse
    p=argparse.ArgumentParser(description="Select adult formal-diagnosis targets from the participant census.")
    p.add_argument('--census',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();df=pd.read_csv(a.census,low_memory=False);df['age']=pd.to_numeric(df.age,errors='coerce')
    for column in ['repeated_recording_participant','common_19_channels_available','primary_ec_60s_complete','primary_cohort_eligible']:
        df[column]=bool_series(df[column])
    df['formal_status']=df.formal_status.fillna('MISSING').astype(str).str.upper();df['indication_values']=df.indication_values.fillna('').astype(str)
    df=df.sort_values('participant_id',kind='stable').reset_index(drop=True)
    formal=df.loc[df.primary_cohort_eligible];assert formal.loc[formal.formal_status.eq('HEALTHY'),'indication_values'].eq('HEALTHY').all()
    rows=[{'cohort_role':'FORMAL_ADHD_CASE' if r.formal_status=='ADHD' else 'FORMAL_HEALTHY_CONTROL','target_label':r.formal_status,**common(r)} for _,r in formal.iterrows()]
    assert len(rows)==193 and sum(r['target_label']=='ADHD' for r in rows)==56
    assert all(r['selected_session']=='ses-1' for r in rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator=chr(10));writer.writeheader();writer.writerows(rows)
    print('Selected 193 participants: 56 ADHD and 137 controls')
if __name__=='__main__':main()
