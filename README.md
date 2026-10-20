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

## Code (initial release)

| Script | Role |
| --- | --- |
| `Code/1.1.prepare_finetuning_input.py` | Build the pooled training index (`Data/finetuning_data.parquet` locally; not in git). |
| `Code/1.2.prepare_forecast_input.py` | Build or load held-out evaluation NPZ files under `Data/forecast_input/`. |

Run from the project root with Python 3. Example:

```bash
python -B Code/1.2.prepare_forecast_input.py --self-test
```

Additional pipeline scripts (`0.x` data prep, fine-tuning, forecasting, figures)
may be added later; acquisition scripts under `Code/0.*` are not part of the
public repository.

Code is licensed under the [MIT license](LICENSE).
