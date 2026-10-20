"""Ramp-forecast errors and fine-tuning improvements (L60).

Import the SAME source-data and forecast loading routines as the ramp examples
(5.1), but do NOT call its case selector.
Each verification case contains 60 context minutes and 15 future values from
ONE forecast origin. RMSE uses ONLY those 15 future GHI values, in W m-2.

Observation-only cohorts: upward ramps require median of the first five future
  kappa values minus origin kappa >= +0.25; downward ramps require <= -0.25.
  No range or GHI-change threshold. The two groups are disjoint.

NO model-error, model-overshoot, fine-tuned-RMSE, or gain filters select the
cohorts. Every model is scored on identical origin/target pairs. Report lower
RMSE than pretrained, lower than persistence, and lower than BOTH as outcomes,
never as selection filters. Include failures and ties in every distribution.

Report cohort prevalence among ALL eligible origins, pooled RMSE (sqrt of
pooled MSE, not average individual RMSE), per-origin RMSE distributions, and
paired gains = comparator RMSE - fine-tuned RMSE. Positive gain favors tuning.
Because adjacent origins overlap, counts are forecast origins, NOT independent
events. These are exploratory, Figure 3-motivated cohorts, not a new independent
test of anticipation. Groups/thresholds are not optimized to obtain gains.

Default: summaries in chat/terminal. --plot-data: stream distributions of
persistence RMSE minus model RMSE to 4.7 for both TimesFM and TimesFM-BSRN,
separately for upward/downward ramps, using ALL origins in each observation-defined
group. Positive means the model is better than persistence. Negative values and
ties are retained; both models use the same origin pairs and smoothing bandwidth.
No model-performance selection is applied. Errors above the plotted axis
range remain in the densities and summaries. No intermediate files, new
folders, or model runs.
"""

from pathlib import Path
import argparse
import csv
import importlib.util
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 60
N = 15
GROUPS = ("Upward ramps", "Downward ramps")
METHODS = ("Persistence", "TimesFM", "TimesFM-BSRN")
MIN_CHANGE_MILLI = 250
BIN_WIDTH = 5.  # W m-2, common density grid.
BANDWIDTH = 15.  # W m-2, identical display smoothing for all models/groups.


def case_module():
    spec = importlib.util.spec_from_file_location("figure3_cases", ROOT / "Code/5.1.persistence_hurdle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.CONTEXT_MINUTES != CONTEXT or module.FORECAST_MINUTES != N:
        raise ValueError("Figure 3 context/forecast settings have changed")
    return module


def observed_groups(rows, kappa, clear):
    """Forecasts deliberately are NOT arguments to this cohort selector."""
    future_rows = rows[:, None]+np.arange(1, N+1)
    target_kappa = kappa[future_rows]
    codes = np.rint(np.column_stack((kappa[rows], target_kappa))*1000).astype(np.int32)
    net_milli = np.median(codes[:, 1:6], axis=1)-codes[:, 0]
    target_ghi = target_kappa*clear[future_rows]
    masks = {
        GROUPS[0]: net_milli >= MIN_CHANGE_MILLI,
        GROUPS[1]: net_milli <= -MIN_CHANGE_MILLI,
    }
    return masks, target_ghi


def origin_rmse(target, predictions):
    return np.column_stack([np.sqrt(np.mean((predictions[name]-target)**2, axis=1)) for name in METHODS])


def outcomes(rmse):
    persistence, pretrained, tuned = rmse.T
    return dict(
        better_pretrained=tuned < pretrained,
        better_persistence=tuned < persistence,
        better_both=(tuned < pretrained) & (tuned < persistence),
    )


def density_curves(arrays, nonnegative):
    """All values retained; same bandwidth; boundary reflection only at RMSE=0."""
    low = 0. if nonnegative else np.floor((min(x.min() for x in arrays)-5*BANDWIDTH)/BIN_WIDTH)*BIN_WIDTH
    high = np.ceil((max(x.max() for x in arrays)+5*BANDWIDTH)/BIN_WIDTH)*BIN_WIDTH
    edges = np.arange(low, high+BIN_WIDTH/2, BIN_WIDTH)
    curves = []
    for values in arrays:
        counts, _ = np.histogram(values, bins=edges)
        assert counts.sum() == len(values)
        curve = gaussian_filter1d(counts.astype(float), BANDWIDTH/BIN_WIDTH,
                                  mode="reflect" if nonnegative else "constant")
        curve /= len(values)*BIN_WIDTH
        np.testing.assert_allclose(curve.sum()*BIN_WIDTH, 1.)
        curves.append(curve)
    return (edges[:-1]+edges[1:])/2, curves


def self_test():
    rows = np.array([59, 139])
    kappa, clear = np.full(160, 0.2), np.full(160, 500.)
    kappa[60:75] = np.linspace(0.5, 0.9, N)
    kappa[139] = 0.8
    kappa[140:155] = np.linspace(0.5, 0.1, N)
    masks, target = observed_groups(rows, kappa, clear)
    np.testing.assert_array_equal(masks[GROUPS[0]], [True, False])
    np.testing.assert_array_equal(masks[GROUPS[1]], [False, True])
    assert not (masks[GROUPS[0]] & masks[GROUPS[1]]).any()
    # Changes confined to minutes 11--15 must NOT enter the new ramp groups.
    late_only = np.full(75, 0.2)
    late_only[70:75] = 0.9
    late_masks, _ = observed_groups(np.array([59]), late_only, np.full(75, 500.))
    assert not any(mask.any() for mask in late_masks.values())
    # Exactly +0.25 in the first five future minutes qualifies inclusively.
    late_only[60:65] = 0.45
    boundary_masks, _ = observed_groups(np.array([59]), late_only, np.full(75, 500.))
    assert boundary_masks[GROUPS[0]][0]
    dim_masks, _ = observed_groups(rows, kappa, clear*0.01)
    for group in GROUPS:
        np.testing.assert_array_equal(masks[group], dim_masks[group])
    prediction = dict(zip(METHODS, (target+100, target+50, target)))
    rmse = origin_rmse(target, prediction)
    np.testing.assert_allclose(rmse, [[100, 50, 0], [100, 50, 0]])
    assert all(outcome.all() for outcome in outcomes(rmse).values())
    failures = outcomes(np.array([[10., 20., 30.], [10., 10., 10.]]))
    assert not any(outcome.any() for outcome in failures.values())
    # Gain sign and pooled-vs-mean RMSE are explicitly tested.
    np.testing.assert_array_equal(rmse[:, 1]-rmse[:, 2], [50, 50])
    np.testing.assert_array_equal(rmse[:, 0]-rmse[:, 2], [100, 100])
    np.testing.assert_array_equal(np.array([[10., 20., 30.], [10., 10., 10.]])[:, 0]
                                  - np.array([[10., 20., 30.], [10., 10., 10.]])[:, 2], [-20, 0])
    assert np.sqrt(np.mean(np.array([3., 4.])**2)) != np.mean([3., 4.])
    density_curves([rmse[:, 0], rmse[:, 1], rmse[:, 2]], True)
    density_curves([np.array([-20., 0., 30.]), np.array([-5., 8.])], False)
    print("Tests passed: observation-only groups, up/down signs, 15-value RMSE, failures/ties retained, paired gains, pooled RMSE, densities.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plot-data", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    cases = case_module()
    helper = cases.input_module()
    pooled = {group: [] for group in GROUPS}
    total, union_count = 0, 0
    for station, year in cases.HOLDOUTS:
        rows, _, kappa, clear, predictions = cases.load_station(helper, station, year)
        masks, target = observed_groups(rows, kappa, clear)
        rmse = origin_rmse(target, predictions)
        union = np.logical_or.reduce(list(masks.values()))
        total += len(rows)
        union_count += union.sum()
        for group, mask in masks.items():
            pooled[group].append(rmse[mask])
        print(f"{station}: {len(rows):,} eligible origins; "+", ".join(f"{g}={m.sum():,}" for g, m in masks.items()), file=sys.stderr, flush=True)
    arrays = {group: np.concatenate(parts) for group, parts in pooled.items()}
    print(f"Distinct-origin union: {union_count:,}/{total:,} ({union_count/total:.2%}). Groups are disjoint; adjacent forecast windows overlap.", file=sys.stderr)
    records = []
    for group, rmse in arrays.items():
        if not len(rmse):
            raise ValueError(f"Empty cohort: {group}")
        frequency = {name: np.mean(mask) for name, mask in outcomes(rmse).items()}
        print(f"{group}: n={len(rmse):,} ({len(rmse)/total:.2%} of all origins); "+", ".join(f"{key}={value:.2%}" for key, value in frequency.items()), file=sys.stderr)
        for i, method in enumerate(METHODS):
            x = rmse[:, i]
            print(f"  {method}: pooled RMSE={np.sqrt(np.mean(x*x)):.3f}; mean origin RMSE={x.mean():.3f}; median={np.median(x):.3f}", file=sys.stderr)
        for i, method in enumerate(METHODS[:2]):
            gain = rmse[:, i]-rmse[:, 2]
            print(f"  Gain vs {method}: mean={gain.mean():.3f}; median={np.median(gain):.3f}; improved={np.mean(gain>0):.2%}; worsened={np.mean(gain<0):.2%}; ties={np.mean(gain==0):.2%}", file=sys.stderr)
        for i, method in enumerate(METHODS[1:], start=1):
            records.append((group, method, rmse[:, 0]-rmse[:, i]))
    if not args.plot_data:
        return
    writer = csv.writer(sys.stdout)
    writer.writerow(["group", "method", "value", "density", "n"])
    grid, curves = density_curves([record[2] for record in records], False)
    for (group, method, values), curve in zip(records, curves):
        for value, probability in zip(grid, curve):
            writer.writerow([group, method, value, probability, len(values)])


if __name__ == "__main__":
    main()
