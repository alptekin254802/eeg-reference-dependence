from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


METADATA = None
RECORDINGS = None
OUTPUT = None
PRIMARY_AGE_MIN = 18.0
PRIMARY_AGE_MAX = 55.0
COMMON_19 = {
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
}


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def truth(value: Any) -> bool:
    return text(value).casefold() in {"true", "1", "yes"}


def session_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", value)
    return (int(match.group(1)), value) if match else (10**9, value)


def first_valid(rows: list[dict[str, Any]], column: str) -> Any:
    return next((row.get(column) for row in rows if text(row.get(column))), "")


def main() -> None:
    metadata = pd.read_excel(METADATA)
    metadata["__row"] = range(len(metadata))
    recordings = pd.read_csv(RECORDINGS, low_memory=False)

    metadata_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata.to_dict("records"):
        participant_id = text(row.get("TDBRAIN_ID"))
        if participant_id:
            metadata_by_id[participant_id].append(row)

    recordings_by_id_session: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in recordings.to_dict("records"):
        recordings_by_id_session[text(row["participant_id"])][text(row["session"])].append(row)

    output_rows: list[dict[str, Any]] = []
    for participant_id in sorted(metadata_by_id):
        rows = metadata_by_id[participant_id]
        ordered = sorted(
            rows,
            key=lambda row: (
                number(row.get("sessID")) if number(row.get("sessID")) is not None else float("inf"),
                int(row["__row"]),
            ),
        )
        diagnoses = sorted({text(row.get("formal_status")).upper() for row in rows if text(row.get("formal_status"))})
        formal_status = diagnoses[0] if len(diagnoses) == 1 else "AMBIGUOUS" if diagnoses else "MISSING"
        indications = sorted({text(row.get("indication")) for row in rows if text(row.get("indication"))})
        datasets = sorted({text(row.get("Dataset")) for row in rows if text(row.get("Dataset"))})
        age = next((value for row in ordered if (value := number(row.get("age"))) is not None and 0 <= value <= 120), None)
        sex_code = next((int(value) for row in ordered if (value := number(row.get("gender"))) in {0, 1}), None)
        sex = "Female" if sex_code == 0 else "Male" if sex_code == 1 else "Unknown"

        sessions = sorted(recordings_by_id_session.get(participant_id, {}), key=session_key)
        selected_session = sessions[0] if sessions else ""
        selected_records = recordings_by_id_session.get(participant_id, {}).get(selected_session, [])

        def task_record(task: str) -> dict[str, Any] | None:
            return next(
                (
                    row for row in selected_records
                    if text(row.get("task")).casefold() == task.casefold()
                    and truth(row.get("structurally_usable"))
                ),
                None,
            )

        ec = task_record("restEC")
        eo = task_record("restEO")
        ec_channels = set(text(ec.get("physical_channel_names")).split("|")) if ec else set()
        ec_duration = number(ec.get("duration_s")) if ec else None
        eo_duration = number(eo.get("duration_s")) if eo else None
        ec_sfreq = number(ec.get("sfreq_hz")) if ec else None
        common_19 = bool(ec and COMMON_19.issubset(ec_channels))
        primary_ec_complete = bool(ec and ec_duration is not None and ec_duration >= 60 and common_19 and ec_sfreq == 500.0)

        if formal_status == "ADD":
            exclusion = "ADD kept separate from ADHD"
        elif formal_status == "UNKNOWN":
            exclusion = "formal diagnosis UNKNOWN / indication-only"
        elif formal_status == "MISSING":
            exclusion = "missing formal diagnosis (indication-only if indication present)"
        elif formal_status == "AMBIGUOUS":
            exclusion = "ambiguous formal diagnosis"
        elif formal_status not in {"ADHD", "HEALTHY"}:
            exclusion = "other formal diagnosis"
        elif age is None:
            exclusion = "missing/invalid age"
        elif age < PRIMARY_AGE_MIN:
            exclusion = "age below prespecified adult common-support lower bound"
        elif age > PRIMARY_AGE_MAX:
            exclusion = "age above prespecified common-support upper bound"
        elif not selected_session:
            exclusion = "no observed signal session"
        elif not primary_ec_complete:
            exclusion = "earliest-session EC lacks required structural support"
        else:
            exclusion = ""

        selected_metadata_session = number(first_valid(ordered, "sessID"))
        output_rows.append(
            {
                "participant_id": participant_id,
                "formal_status": formal_status,
                "indication_values": ";".join(indications),
                "dataset_membership": ";".join(datasets),
                "age": "" if age is None else age,
                "sex": sex,
                "metadata_row_count": len(rows),
                "metadata_nrSessions": first_valid(ordered, "nrSessions"),
                "observed_session_count": len(sessions),
                "repeated_recording_participant": len(sessions) > 1,
                "selected_session": selected_session,
                "selected_session_rule": "lowest numeric observed file session only",
                "selected_metadata_sessID": "" if selected_metadata_session is None else int(selected_metadata_session),
                "baseline_identity": "earliest observed session; later visits excluded; universal pretreatment status not independently documented",
                "medication_state": "UNKNOWN (official 'drugs' field is time since recreational drugs, not medication)",
                "eo_structurally_usable": bool(eo),
                "ec_structurally_usable": bool(ec),
                "both_states_structurally_usable": bool(eo and ec),
                "eo_duration_s": "" if eo_duration is None else eo_duration,
                "ec_duration_s": "" if ec_duration is None else ec_duration,
                "sampling_rate_hz": "" if ec_sfreq is None else ec_sfreq,
                "physical_channel_count": len(ec_channels) if ec else "",
                "common_19_channels_available": common_19,
                "primary_ec_60s_complete": primary_ec_complete,
                "primary_age_rule": "18 <= age <= 55",
                "primary_cohort_eligible": exclusion == "",
                "exclusion_reason": exclusion,
            }
        )

    fields = list(output_rows[0])
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    eligible = [row for row in output_rows if row["primary_cohort_eligible"]]
    counts = {label: sum(row["formal_status"] == label for row in eligible) for label in ("ADHD", "HEALTHY")}
    assert counts == {"ADHD": 56, "HEALTHY": 137}, counts
    assert all(row["selected_session"] == "ses-1" for row in eligible)
    assert all(row["sampling_rate_hz"] == 500.0 for row in eligible)
    print(f"wrote {OUTPUT} with {len(output_rows)} unique metadata participants")
    print(f"primary EC cohort: ADHD={counts['ADHD']}, HEALTHY={counts['HEALTHY']}, total={len(eligible)}")


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description="Construct the participant census from released metadata and BDF inventory.")
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();METADATA,RECORDINGS,OUTPUT=a.metadata,a.inventory,a.output
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    main()
