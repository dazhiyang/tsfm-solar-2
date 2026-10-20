"""Paired station-level RMSE skill from the sampled network forecasts.

  python -B Code/5.3.network_skill.py --self-test
  python -B Code/5.3.network_skill.py

CSV goes to stdout for script 4.8; no intermediate files are saved.
Skill is 100*(1-RMSE_model/RMSE_persistence), not MSE-based skill.
"""

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("TimesFM", "TimesFM-BSRN")


def rmse(observed, forecast):
    return float(np.sqrt(np.mean(np.square(forecast-observed))))


def skill_percent(error, reference_error):
    return 100*(1-error/reference_error) if reference_error > 0 else float("nan")


def station_metadata():
    package_src = os.environ.get("BSRN_PACKAGE_SRC")
    if not package_src:
        from bsrn.constants import BSRN_STATIONS
        return BSRN_STATIONS
    package = Path(package_src)
    tree = ast.parse((package/"bsrn/constants.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BSRN_STATIONS"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("BSRN_STATIONS metadata not found.")


def station_scores(path):
    table = pq.read_table(path)
    experiment = json.loads(table.schema.metadata[b"experiment"])
    frame = table.to_pandas()
    expected = {r["station"]: r for r in experiment["station_counts"]}
    if set(frame.Station) != set(expected):
        raise ValueError("Exported stations differ from sampling metadata.")
    metadata = station_metadata()
    obs_columns = [f"Obs{h}" for h in range(1, 16)]
    fcst_columns = [f"Fcst{h}" for h in range(1, 16)]
    identity = ["Year", "Time", "OriginRow", "InputRow"]
    rows = []
    for station, group in frame.groupby("Station", sort=True):
        if set(group.Method) != {"Persistence", *MODELS}:
            raise ValueError(f"Incomplete method set: {station}")
        reference = group[group.Method == "Persistence"].sort_values("InputRow")
        if (len(reference) != expected[station]["sampled"]
                or reference.InputRow.duplicated().any()
                or set(reference.Year) != {expected[station]["year"]}):
            raise ValueError(f"Invalid reference origins: {station}")
        observed = reference[obs_columns].to_numpy()
        errors = {"Persistence": rmse(observed, reference[fcst_columns].to_numpy())}
        for model in MODELS:
            data = group[group.Method == model].sort_values("InputRow")
            np.testing.assert_array_equal(data[identity].to_numpy(), reference[identity].to_numpy())
            np.testing.assert_array_equal(data[obs_columns].to_numpy(), observed)
            forecast = data[fcst_columns].to_numpy()
            if not (np.isfinite(observed).all() and np.isfinite(forecast).all()):
                raise ValueError(f"Nonfinite forecasts: {station} {model}")
            errors[model] = rmse(observed, forecast)
        geo = metadata[station]
        for model in MODELS:
            rows.append(dict(station=station, year=expected[station]["year"], n_origins=len(reference),
                             model=model, lat=geo["lat"], lon=geo["lon"], kgc=geo["kgc"],
                             rmse=errors[model], persistence_rmse=errors["Persistence"],
                             skill_percent=skill_percent(errors[model], errors["Persistence"]),
                             skill_change_pp=skill_percent(errors[model], errors["Persistence"])
                             -skill_percent(errors["TimesFM"], errors["Persistence"])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=ROOT/"Data/result/network_1000_timesfm_L60.parquet")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert rmse(np.array([0., 0.]), np.array([2., -2.])) == 2
        assert skill_percent(2, 4) == 50
        assert skill_percent(5, 4) == -25
        assert np.isnan(skill_percent(0, 0))
        print("PASS: RMSE, positive/negative RMSE-based skill, and zero reference.")
        return
    rows = station_scores(args.input_file)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    adapted = [r for r in rows if r["model"] == "TimesFM-BSRN"]
    print(f"Verified {len(adapted)} paired station comparisons; fine-tuning improves RMSE at "
          f"{sum(r['skill_change_pp'] > 0 for r in adapted)}/{len(adapted)} stations.", file=sys.stderr)


if __name__ == "__main__":
    main()
