"""Station-balanced minus full-data RMSE skill, in percentage points.

CSV goes to stdout for 5.9; no intermediate files. Positive values favor
TimesFM-Balanced. Pair exact origins and observations before calculating scores.
  python -B Code/4.5.station_balanced_skill.py
"""

import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def station_scores():
    network = sibling("balanced_network_metrics", "4.3.network_skill.py")
    forecast = sibling("balanced_pairing", "3.5.forecast_timesfm_station_balanced.py")
    reference, methods = forecast.reference_cohort(
        pq.read_table(ROOT / "Data/result/network_1000_timesfm_L60.parquet"))
    balanced = pq.read_table(ROOT / "Data/result/network_1000_timesfm_balanced_L60.parquet")
    balanced = balanced.sort_by([("InputRow", "ascending")])
    paired = list(forecast.ID_COLUMNS + forecast.OBS_COLUMNS)
    if not balanced.select(paired).equals(reference.select(paired)):
        raise ValueError("Balanced forecasts do not share exact reference origins/observations")
    if set(balanced["Method"].to_pylist()) != {forecast.METHOD}:
        raise ValueError("Unexpected balanced forecast method")
    methods[forecast.METHOD] = balanced
    observed = np.column_stack([reference[c].to_numpy() for c in forecast.OBS_COLUMNS])
    predictions = {name: np.column_stack([part[f"Fcst{h}"].to_numpy() for h in range(1, 16)])
                   for name, part in methods.items()}
    if not np.isfinite(observed).all() or any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("Nonfinite observations/forecasts")
    station_ids = reference["Station"].to_numpy()
    geography = network.station_metadata()
    rows = []
    for station in sorted(set(station_ids)):
        used = station_ids == station
        errors = {name: network.rmse(observed[used], values[used]) for name, values in predictions.items()}
        geo = geography[station]
        full_skill = network.skill_percent(errors["TimesFM-BSRN"], errors["Persistence"])
        balanced_skill = network.skill_percent(errors[forecast.METHOD], errors["Persistence"])
        rows.append(dict(station=station, lat=geo["lat"], lon=geo["lon"], kgc=geo["kgc"],
                         n_origins=int(used.sum()), full_rmse=errors["TimesFM-BSRN"],
                         balanced_rmse=errors[forecast.METHOD], full_skill=full_skill,
                         balanced_skill=balanced_skill, skill_difference_pp=balanced_skill-full_skill))
    return rows


def main():
    rows = station_scores()
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    gains = np.array([r["skill_difference_pp"] for r in rows])
    print(f"Verified {len(rows)} paired stations; balancing improves {int((gains > 0).sum())}; "
          f"median skill difference {np.median(gains):.3f} pp.", file=sys.stderr)


if __name__ == "__main__":
    main()
