# tsfm-solar-2

Domain adaptation of time-series foundation models for minute-scale solar irradiance forecasting

## Example and verification data

**Only sample station files are versioned.** The repository ships Qiqihar (QIQ)
parquet under `Data/bsrn_1min/qiq/` as a format example. It does **not** include
the full multi-station `Data/bsrn_1min/` mirror used for training; obtain other
BSRN station-years locally or from BSRN directly. Forecast contexts and targets
for all reserved station-years in the bundled NPZ are independent of which
parquet files are checked in.

```text
Data/
├── bsrn_1min/
│   └── qiq/                    # sample only (2024–2025)
│       ├── qiq_2024.parquet
│       └── qiq_2025.parquet
└── forecast_input/
    └── holdout_all.npz
```

### QIQ sample parquet

The Qiqihar (QIQ) files illustrate the processed, quality-controlled BSRN
station-year format. Each file uses a complete one-minute UTC grid. Missing or
rejected observations remain flagged; they are not interpolated. Parquet schema
metadata describe the station, units, scales, validity flags, and preprocessing.

```python
import pyarrow.parquet as pq

table = pq.read_table("Data/bsrn_1min/qiq/qiq_2025.parquet")
print(table.schema)
```

### Prepared forecast inputs

`Data/forecast_input/holdout_all.npz` contains 8,603,074 forecast origins from 75
reserved station-years. It includes the latest reserved year per station, not
only the five station-years used for the manuscript's main comparison.

The file is a standard compressed NumPy archive; load it with `allow_pickle=False`.
It is about 99.69 MiB on disk and its arrays expand to several GB in memory.

| Array | Shape | Meaning |
| --- | --- | --- |
| `history` | `(N, 60)` | Previous 60 one-minute clear-sky-index values, including the origin |
| `target` | `(N, 15)` | Next 15 one-minute clear-sky-index values |
| `station` | `(N,)` | Uppercase station code |
| `year` | `(N,)` | Source calendar year |
| `origin_utc_minute` | `(N,)` | Forecast origin, in UTC minutes since the Unix epoch |
| `origin_row` | `(N,)` | Origin row in the source station-year Parquet file |
| `source_index` | `(N,)` | Index into the source records in the metadata |
| `metadata_json` | scalar | JSON preparation settings, source records, and dataset fingerprint |

Contexts and targets are clear-sky indices, not irradiance in W m−2. For a
15- or 30-minute context, retain the last 15 or 30 values of each history.
Reconstruct irradiance using clear-sky irradiance at each target timestamp from
the corresponding source Parquet file. Only QIQ source files are included here
as examples; the NPZ itself contains contexts and targets for all 75 stations.

```python
import json
import numpy as np

with np.load("Data/forecast_input/holdout_all.npz", allow_pickle=False) as data:
    metadata = json.loads(data["metadata_json"].item())
    rows = np.flatnonzero(
        (data["station"] == "QIQ") & (data["year"] == 2025)
    )[:256]
    context = data["history"][rows]
    target = data["target"][rows]

print(context.shape, target.shape)
```

Source paths recorded in the NPZ metadata refer to the original preparation
environment. Resolve source files against the local `Data/bsrn_1min/` directory
when using a cloned repository.

### Manuscript verification years

The main five-station comparison uses ASP (2018), QIQ (2025), PAL (2023),
KWA (2002), and SYO (2024), totaling 698,104 origins. The bundled NPZ contains
QIQ (2025), but its latest-year selections for ASP, PAL, KWA, and SYO differ
from this five-station comparison. Reproducing that comparison requires the
specified station-year source files and the corresponding prepared origins.

BSRN source observations should be acknowledged using the relevant station
data citations and the network description: Driemel et al. (2018),
https://doi.org/10.5194/essd-10-1491-2018.

## Code

Run from the project root with Python 3 (the project uses a `tf_timesfm` conda env
with `peft`, PyTorch, and model-specific packages). Training writes under
`Data/models/` (gitignored).

### Input preparation (`1.x`)

| Script | Role |
| --- | --- |
| `Code/1.1.prepare_finetuning_input.py` | Build the pooled training index (`Data/finetuning_data.parquet`; local only). |
| `Code/1.2.prepare_forecast_input.py` | Build or load held-out evaluation NPZ files under `Data/forecast_input/`. |

```bash
python -B Code/1.2.prepare_forecast_input.py --self-test
```

### Fine-tuning (`2.x`)

Both families read `Data/finetuning_data.parquet` from **1.1**, use LoRA, share the
same 15-minute origin grid and station-year validation split, and export adapters
under `Data/models/`. Pass `--train` to run optimization (off by default).

| Script | Role |
| --- | --- |
| `Code/2.1.finetune_chronos.py` | Chronos-2 LoRA on pooled BSRN; outputs `Data/models/chronos-2-finetuned-L{15,30,60}/`. |
| `Code/2.2.finetune_timesfm.py` | TimesFM 2.5 LoRA on the same split; outputs `Data/models/timesfm-2.5-finetuned-L{15,30,60}/`. |
| `Code/2.3.finetune_timesfm_station_balanced.py` | Experimental L=60 TimesFM run with one station-year per site (~6k origins each). |

```bash
python -B Code/2.1.finetune_chronos.py --history-length 60 --num-steps 1000 --train --output-dir Data/models/chronos-2-finetuned-L60
```

### Forecasting (`3.x`)

Five entry points replace the earlier nine-script layout. **3.1** holds shared
Chronos export, batching, and GHI I/O; **3.3** mirrors that for TimesFM and
imports **3.1** as a library. Pretrained, 1k-step, and full-pass BSRN modes are
selected with `--checkpoint-dir` (manifest + adapter), not separate scripts.
Outputs go to `Data/result/` as `{STATION}_{method}_{L}.txt` (gitignored).

| Script | Role |
| --- | --- |
| `Code/3.1.forecast_chronos.py` | Chronos-2 pretrained or LoRA checkpoint → hold-out GHI text exports. |
| `Code/3.2.forecast_persistence.py` | Persistence baseline; reuses **3.1** evaluation and file format. |
| `Code/3.3.forecast_timesfm.py` | TimesFM 2.5 pretrained or adapted checkpoint → same export contract as **3.1**. |
| `Code/3.4.forecast_network_sample.py` | Network-wide L60 sample (persistence + TimesFM pretrained + full-pass) → Parquet. |
| `Code/3.5.forecast_timesfm_station_balanced.py` | **2.3** balanced adapter on the fixed **3.4** cohort → Parquet. |

```bash
python -B Code/3.1.forecast_chronos.py --self-test
python -B Code/3.3.forecast_timesfm.py --self-test
```

### Analysis (`4.x`)

Python metrics and tables run **before** the R figure scripts. They read **`1.2`**
hold-out origins, **`Data/result/`** exports from **3.x** (or Parquet from **3.4**
/ **3.5**), and stream CSV or tables for the matching **5.x** plotter where noted.

| Script | Role |
| --- | --- |
| `Code/4.1.persistence_hurdle.py` | Six curated ramp illustrations; case finder for **5.5**. |
| `Code/4.2.ramp_forecast_errors.py` | Observation-defined ramp cohorts (±0.25 κ); data for **5.6**. |
| `Code/4.3.network_skill.py` | Station RMSE skill from **3.4** network sample → **5.7**. |
| `Code/4.4.timesfm_embeddings.py` | Layer-wise TimesFM embeddings (uses **3.3** loader) → **5.8**. |
| `Code/4.5.station_balanced_skill.py` | Balanced vs full-pass skill on **3.4** cohort → **5.9**. |
| `Code/4.6.tables_forecast.py` | GHI MBE/RMSE/skill LaTeX tables from `Data/result/`. |

```bash
python -B Code/4.1.persistence_hurdle.py --self-test
python -B Code/4.3.network_skill.py --self-test
python -B Code/4.6.tables_forecast.py --self-test
```

### Figures (`5.x`)

R publication drivers (ggplot2; optional **sf** / **rnaturalearth** for **5.1**).
Vector PDFs are written under `tex/` by default (gitignored). Paired scripts call
the **4.x** analysis step above via `system2` / embedded Python.

| Script | Role |
| --- | --- |
| `Code/5.1.map.R` | BSRN station map with Köppen-class colors. |
| `Code/5.2.training_curves.R` | Full-pass training/validation loss curves from `Data/models/*/loss_history.jsonl`. |
| `Code/5.3.RMSE_60.R` | Hold-out κ RMSE vs lead (L=60); Chronos-2 and TimesFM panels. |
| `Code/5.4.training_sample_comparison.R` | Bar chart of hold-out RMSE by method (network sample). |
| `Code/5.5.persistence_hurdle_cases.R` | Multi-panel ramp illustrations via **4.1**. |
| `Code/5.6.ramp_forecast_errors.R` | Ramp gain distributions via **4.2**. |
| `Code/5.7.network_skill_map.R` | Network skill map via **4.3**. |
| `Code/5.8.timesfm_embeddings.R` | Embedding PCA panels via **4.4**. |
| `Code/5.9.station_balanced_skill_map.R` | Balanced-training skill map via **4.5**. |

```bash
Rscript Code/5.2.training_curves.R
Rscript Code/5.6.ramp_forecast_errors.R
```

### Not in the repository

Acquisition and raw BSRN prep: `Code/0.*` (gitignored). Local-only artifacts:
`Data/models/`, `Data/result/`, `Data/finetuning_data.parquet`, full
`Data/bsrn_1min/` (except the versioned QIQ sample), and `tex/`.

Code is licensed under the [MIT license](LICENSE).
