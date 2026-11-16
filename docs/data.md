# Data access

Acquire each dataset from its provider and comply with the applicable access agreement. Place the downloaded files under a local `data/` directory or pass another path explicitly to the preparation commands. The repository contains analysis code and group statistics. Waveforms, participant metadata, individual predictions and trained checkpoints are generated locally under the applicable data terms.

| Cohort | Access route | Files and analysis population |
| --- | --- | --- |
| TDBRAIN | [Brainclinics dataset portal](https://www.brainclinics.com/resources/tdbrain-dataset) and [access agreement](https://brainclinics.com/resources/tdbrain-dataset/introduction); ORCID sign-in and provider access process | Version 3.1 BIDS BDF recordings, JSON sidecars and `TDBRAIN_participants_V3.xlsx`. Adults aged 18–55; earliest eligible eyes-closed recording. Expected 193 before signal quality exclusions, 189 afterward (56 ADHD, 133 controls). See [core preparation](core.md). |
| Wroclaw | [RepOD, DOI 10.18150/YHSZR3](https://doi.org/10.18150/YHSZR3) | Pre-task eyes-closed recordings from 16 adults (8 per group). Native physical units remain uncertain. Power ratios require a common scale across groups; provisional model transport does not establish unit-verified transport. |
| Nasrabadi | [IEEE DataPort: EEG data for ADHD / Control children](https://doi.org/10.21227/rzfh-zn36) | The 128-Hz, 19-channel task recordings of 121 children (61 ADHD, 60 controls). Apply the documented channel order and group mapping. Ear-reference linking cannot be established from the released description. |


Use the extraction and metadata schemas in [external analyses](external.md). Cohort labels and inclusion decisions must be obtained from the dataset documentation, not inferred from model outputs. Participant-level files produced by the commands remain local; do not add them to a public repository.
