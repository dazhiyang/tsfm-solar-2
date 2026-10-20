"""Reproduce the six selected single-origin, 75-minute ramp illustrations.

  python -B Code/4.1.persistence_hurdle.py
  python -B Code/4.1.persistence_hurdle.py --self-test

Each instance is exactly 60 observed context values followed by 15 future
observations and the complete 15-value forecasts issued at ONE origin.
No rolling fixed-horizon curves. Script 5.5 imports this finder for plotting.

Read existing L60 hold-out forecasts, checking timestamp and target alignment.
The six SELECTED_ORIGINS below are the user-approved mixed illustration set.
They were selected retrospectively for improvement over pretrained TimesFM and
persistence, with at most one station per group and different dates across
groups. Revalidate the original range, ramp, overshoot and improvement rules,
but do not rerank or replace these fixed origins on subsequent plotting runs.
The later RMSE<=100 and persistence-distance filters have been removed.
Curves and errors use GHI (W m-2), reconstructed with the stored clear-sky
reference. Clear-sky-index variation is used only to identify variable cases.
RMSE is calculated over the 15 future values ONLY, not the observed context.
Cases are deliberately selected USING model results to illustrate improvements,
not to estimate their frequency or demonstrate that phase lag is eliminated.
No model loading, training, saved case files, or new folders.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HOLDOUTS = (("ASP", 2018), ("QIQ", 2025), ("PAL", 2023),
            ("KWA", 2002), ("SYO", 2024))
CONTEXT_MINUTES, FORECAST_MINUTES = 60, 15
N_CASES = 6
# Figure 5: fixed panel order (a)--(f), with user-selected mixed illustrations.
SELECTED_ORIGINS = (
    ("QIQ", 2025, "up", "2025-05-05T02:54:00Z"),
    ("ASP", 2018, "up", "2018-01-09T06:43:00Z"),
    ("KWA", 2002, "up", "2002-09-03T22:23:00Z"),
    ("KWA", 2002, "down", "2002-05-28T20:48:00Z"),
    ("QIQ", 2025, "down", "2025-09-09T04:36:00Z"),
    ("PAL", 2023, "down", "2023-08-30T15:28:00Z"),
)
# Persistence-like pretrained selection retained before stronger-ramp search:
# (a) ASP 2018-01-09 06:43 UTC, up
# (b) QIQ 2025-07-22 23:18 UTC, up
# (c) KWA 2002-12-27 20:52 UTC, up
# (d) KWA 2002-05-28 20:48 UTC, down
# (e) PAL 2023-08-30 15:28 UTC, down
# (f) QIQ 2025-10-11 05:31 UTC, down
MIN_OBSERVED_RANGE = 0.4  # kappa, including the last context observation.
MIN_DIRECTION_CHANGE = 0.25  # Median first five targets minus origin kappa.
MAX_FORECAST_OVERSHOOT = 0.2
MIN_RELATIVE_RMSE_REDUCTION = 0.15
METHODS = ("Persistence", "TimesFM", "TimesFM-BSRN")


def input_module():
    spec = importlib.util.spec_from_file_location(
        "forecast_inputs", ROOT / "Code/1.2.prepare_forecast_input.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_station(helper, station, year):
    examples, metadata = helper.load_eval_examples(
        stations=[station], years=[year], max_origins=0)
    source = pq.read_table(metadata["sources"][0]["path"],
                           columns=["kappa_milli", "ghi_clear_centi"])
    kappa = source["kappa_milli"].to_numpy().astype(float) / 1000
    clear = source["ghi_clear_centi"].to_numpy().astype(float) / 100
    rows = examples["origin_row"].astype(np.int64)
    times = examples["origin_utc_minute"].astype(np.int64)
    indices = rows[:, None] + np.arange(1, FORECAST_MINUTES+1)
    target, reference_clear = kappa[indices], clear[indices]
    if np.any(reference_clear <= 0):
        raise ValueError("Nonpositive clear-sky GHI in eligible targets")
    predictions = {METHODS[0]: kappa[rows, None] * reference_clear}
    for prefix, name in (("timesfm", METHODS[1]), ("timesfm_ft_full", METHODS[2])):
        path = ROOT / "Data/result" / f"{station}_{prefix}_{CONTEXT_MINUTES}.txt"
        frame = pd.read_csv(path, sep="\t")
        np.testing.assert_array_equal(frame.Time.to_numpy(), times)
        np.testing.assert_allclose(frame[[f"Obs{h}" for h in range(1,16)]].to_numpy(),
                                   target * reference_clear, rtol=0, atol=0.006)
        predictions[name] = frame[[f"Fcst{h}" for h in range(1,16)]].to_numpy()
        if not np.isfinite(predictions[name]).all():
            raise ValueError(f"Nonfinite forecasts: {path}")
    return rows, times, kappa, clear, predictions


def select_case(station, rows, times, kappa, clear, predictions, direction=None,
                origin=None, excluded_dates=()):
    target_rows = rows[:, None] + np.arange(1, FORECAST_MINUTES+1)
    target_kappa = kappa[target_rows]
    target = target_kappa * clear[target_rows]
    observed = np.column_stack([kappa[rows], target_kappa])
    rmse = {name: np.sqrt(np.mean((prediction-target)**2, axis=1))
            for name, prediction in predictions.items()}
    gain = rmse[METHODS[1]] - rmse[METHODS[2]]
    eligible = ((np.ptp(observed, axis=1) >= MIN_OBSERVED_RANGE)
                & (np.max(predictions[METHODS[1]] / clear[target_rows], axis=1)
                   <= np.max(observed, axis=1) + MAX_FORECAST_OVERSHOOT)
                & (np.min(predictions[METHODS[1]] / clear[target_rows], axis=1)
                   >= np.min(observed, axis=1) - MAX_FORECAST_OVERSHOOT)
                & (gain >= MIN_RELATIVE_RMSE_REDUCTION * rmse[METHODS[1]])
                & (rmse[METHODS[2]] < rmse[METHODS[0]]))
    change = np.median(target_kappa[:, :5], axis=1) - kappa[rows]
    ghi_change = np.median(target[:, :5], axis=1) - kappa[rows] * clear[rows]
    if direction == "up":
        eligible &= change >= MIN_DIRECTION_CHANGE
    elif direction == "down":
        eligible &= change <= -MIN_DIRECTION_CHANGE
    elif direction is not None:
        raise ValueError("Direction must be up, down, or None")
    if excluded_dates:
        eligible &= ~np.isin(times // 1440, excluded_dates)
    if origin is not None:
        minute = pd.Timestamp(origin).value // 60_000_000_000
        eligible &= times == minute
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        return None
    chosen = int(candidates[np.argmax(gain[candidates])])
    row = int(rows[chosen])
    window = slice(row-CONTEXT_MINUTES+1, row+FORECAST_MINUTES+1)
    observed_window = kappa[window] * clear[window]
    assert len(observed_window) == CONTEXT_MINUTES + FORECAST_MINUTES
    series = {"Observed": observed_window}
    # The origin anchor is observed, not a model prediction; future points are
    # the 15 forecasts from this single row of the saved forecast file.
    series.update({name: np.r_[kappa[row] * clear[row], prediction[chosen]]
                   for name, prediction in predictions.items()})
    return dict(station=station,
                origin=pd.to_datetime(int(times[chosen]), unit="m", utc=True),
                direction=direction, observed_ghi_change=float(ghi_change[chosen]),
                gain=float(gain[chosen]), series=series,
                rmse={name: float(value[chosen]) for name, value in rmse.items()})


def find_cases(n_cases=N_CASES, verbose=True):
    if n_cases != N_CASES:
        raise ValueError("The comparison sheet uses two groups of three cases")
    helper = input_module()
    found = {}
    for station, year in dict.fromkeys((s, y) for s, y, _, _ in SELECTED_ORIGINS):
        if verbose:
            print(f"Loading {station} {year}: fixed single-origin 60+15-minute windows...", flush=True)
        loaded = load_station(helper, station, year)
        for s, y, direction, origin in SELECTED_ORIGINS:
            if (s, y) != (station, year):
                continue
            case = select_case(station, *loaded, direction=direction, origin=origin)
            if case is None:
                raise ValueError(f"Fixed case no longer satisfies the original rules: {station} {origin}")
            case["group"] = "Up ramps" if direction == "up" else "Down ramps"
            found[(station, origin)] = case
    cases = [found[(station, origin)] for station, _, _, origin in SELECTED_ORIGINS]
    if verbose:
        print("\nSelected 75-minute instances (post hoc illustrations):")
        for case in cases:
            print(f"{case['station']} | origin {case['origin']:%Y-%m-%d %H:%M} UTC | "
                  f"{case['group']} | 60 context + 15 forecast values | "
                  f"({case['observed_ghi_change']:+.1f} W m-2)")
            for name in METHODS:
                print(f"  {name:15s} future-only GHI RMSE={case['rmse'][name]:.2f} W m-2")
    return cases


def self_test():
    assert len(SELECTED_ORIGINS) == N_CASES == len(set(SELECTED_ORIGINS))
    assert [d for _, _, d, _ in SELECTED_ORIGINS] == ["up"]*3+["down"]*3
    rng = np.random.default_rng(42)
    kappa = rng.uniform(0, 1, 500)
    rows, times = np.arange(60, 360), np.arange(100, 400)
    target = kappa[rows[:, None] + np.arange(1, 16)]
    persistence = np.repeat(kappa[rows, None], 15, axis=1)
    predictions = {METHODS[0]: persistence, METHODS[1]: persistence,
                   METHODS[2]: target}
    case = select_case("TEST", rows, times, kappa, np.ones_like(kappa), predictions)
    assert case is not None and case["rmse"][METHODS[2]] == 0
    assert len(case["series"]["Observed"]) == 75
    for name in METHODS:
        assert len(case["series"][name]) == 16
        assert case["series"][name][0] == case["series"]["Observed"][59]
    np.testing.assert_array_equal(case["series"][METHODS[2]][1:], case["series"]["Observed"][60:])
    clear = np.linspace(400, 900, len(kappa))
    future_clear = clear[rows[:, None] + np.arange(1, 16)]
    ghi_predictions = {name: values * future_clear for name, values in predictions.items()}
    ghi_case = select_case("TEST", rows, times, kappa, clear, ghi_predictions)
    assert ghi_case is not None and ghi_case["rmse"][METHODS[2]] == 0
    np.testing.assert_array_equal(ghi_case["series"][METHODS[2]][1:],
                                  ghi_case["series"]["Observed"][60:])
    for direction, anchor, future in (("up", 0.2, np.linspace(0.5, 0.9, 15)),
                                      ("down", 0.8, np.linspace(0.5, 0.1, 15))):
        values = np.r_[np.full(60, anchor), future]
        reference_clear = np.full(75, 500.)
        forecasts = {METHODS[0]: np.full((1, 15), anchor * 500),
                     METHODS[1]: np.full((1, 15), anchor * 500),
                     METHODS[2]: future[None, :] * 500}
        found = select_case("TEST", np.array([59]), np.array([100]), values,
                            reference_clear, forecasts, direction=direction)
        assert found is not None and found["direction"] == direction
        opposite = "down" if direction == "up" else "up"
        assert select_case("TEST", np.array([59]), np.array([100]), values,
                           reference_clear, forecasts, direction=opposite) is None
    print("PASS: single-origin 60+15 windows, GHI conversion, future-only RMSE, up/down selection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plot-data", action="store_true",
                        help="Send selected GHI curves as CSV to stdout for R script 5.5; save nothing")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.plot_data:
        writer = csv.DictWriter(sys.stdout, fieldnames=["case", "minute", "method", "ghi"])
        writer.writeheader()
        for index, case in enumerate(find_cases(verbose=False)):
            kind = {"Up ramps": "up", "Down ramps": "down"}[case["group"]]
            label = f"({chr(97+index)}) {case['station']}: {kind}\n{case['origin']:%Y-%m-%d %H:%M} UTC"
            for name, values in case["series"].items():
                first = 1-CONTEXT_MINUTES if name == "Observed" else 0
                for minute, value in enumerate(values, start=first):
                    writer.writerow(dict(case=label, minute=minute, method=name, ghi=float(value)))
    else:
        find_cases()


if __name__ == "__main__":
    main()
