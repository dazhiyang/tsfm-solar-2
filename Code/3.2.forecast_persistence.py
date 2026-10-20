"""Clear-sky-index persistence on the shared NPZ inputs from script 1.2.

Run with Python, preferably with -B.
  python -B Code/3.2.forecast_persistence.py --stations CAB
  python -B Code/3.2.forecast_persistence.py --koppen-holdout-stations
  python -B Code/3.2.forecast_persistence.py --self-test

Without explicit --years, the latest available year at each station is used.

Repeats the last observed clear-sky index at the forecast origin for all 15
leads. With identical origins from script 1.2, the forecast does not depend on
whether one later slices 15-, 30-, or 60-minute history suffixes.

Prints the same lead-wise and station summaries as script 3.1. Writes
CAB_pers_15.txt, CAB_pers_30.txt, and CAB_pers_60.txt (one row per origin;
Time, Obs1..Obs15, Fcst1..Fcst15 in W m-2). Forecast is identical across L.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np

FORECAST_LENGTH = 15
METHOD = "pers"


def input_module():
    path = Path(__file__).with_name("1.2.prepare_forecast_input.py")
    spec = importlib.util.spec_from_file_location("bsrn_forecast_input", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics_module():
    path = Path(__file__).with_name("3.1.forecast_chronos.py")
    spec = importlib.util.spec_from_file_location("bsrn_chronos_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forecast(histories):
    """Repeat each origin's last observed clear-sky index for all 15 leads."""
    histories = np.asarray(histories, dtype=np.float32)
    if histories.ndim != 2 or histories.shape[1] < 1:
        raise ValueError("Expected a nonempty (origins, history) array.")
    if not np.isfinite(histories).all() or np.any(histories < 0):
        raise ValueError("Histories must be finite and nonnegative.")
    return np.repeat(histories[:, -1:], FORECAST_LENGTH, axis=1)


def assert_history_suffix_invariant(examples, helper):
    """Persistence at t uses kappa(t); suffixes of one 60-minute row must agree."""
    base = forecast(examples["history"])
    for length in helper.HISTORY_LENGTHS:
        alt = forecast(helper.histories_for(examples, length))
        if not np.array_equal(base, alt):
            raise ValueError(
                f"Persistence changed for history length {length}; "
                "origins may not share a common 60-minute context."
            )


def self_test():
    histories = np.array([[0.2, 0.5, 0.9], [1.0, 2.0, 2.0]], dtype=np.float32)
    prediction = forecast(histories)
    np.testing.assert_array_equal(
        prediction,
        np.array([[0.9] * 15, [2.0] * 15], dtype=np.float32),
    )
    helper = input_module()
    full = np.zeros((2, 60), dtype=np.float32)
    full[0, -4:] = [0.1, 0.2, 0.3, 0.4]
    full[1, -6:] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    examples = {"history": full, "target": np.zeros((2, 15), dtype=np.float32)}
    assert_history_suffix_invariant(examples, helper)
    print("Persistence checks passed.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    helper = input_module()
    helper.add_eval_input_arguments(parser)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Directory for GHI text outputs (default: same as script 3.1).",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    has_file = args.input_file is not None
    has_stations = bool(args.stations) or args.koppen_holdout_stations
    if has_file == has_stations:
        parser.error("Provide exactly one of --input-file or --stations/--koppen-holdout-stations.")
    if has_file and not args.input_file.is_file():
        parser.error(f"Missing input file: {args.input_file}")
    if args.max_origins < 0 or args.seed < 0 or args.stride_minutes < 1:
        parser.error("Use nonnegative --max-origins/--seed and positive --stride-minutes.")
    if args.years is not None and args.stations and len(args.years) != len(args.stations):
        parser.error("Provide one --years value for each --stations entry.")
    chronos = metrics_module()
    examples, input_metadata = helper.load_eval_examples(
        input_file=args.input_file,
        data_root=args.data_root,
        stations=args.stations,
        years=args.years,
        koppen_holdout_stations=args.koppen_holdout_stations,
        max_origins=args.max_origins,
        seed=args.seed,
        stride_minutes=args.stride_minutes,
        non_overlapping=args.non_overlapping,
    )
    for source in input_metadata["sources"]:
        print(
            f"Eval input: {source['station']} {source['year']}: "
            f"{source['selected_count']:,} origins",
            flush=True,
        )

    started = perf_counter()
    assert_history_suffix_invariant(examples, helper)
    prediction = forecast(examples["history"])
    elapsed = perf_counter() - started

    report = chronos.evaluate(
        prediction,
        examples["target"],
        examples["station"],
        examples["history"],
    )
    chronos.print_report("Persistence", report)
    print(f"Computed {len(prediction):,} forecasts in {elapsed:.3f}s", flush=True)
    if args.no_save:
        return

    result_dir = args.result_dir if args.result_dir is not None else chronos.RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        for station in set(examples["station"]):
            for length in chronos.HISTORY_LENGTHS:
                path = result_dir / f"{station}_{METHOD}_{length}.txt"
                if path.exists():
                    parser.error(f"Refusing to overwrite {path}; use --overwrite or --no-save.")
    written = []
    for length in chronos.HISTORY_LENGTHS:
        written.extend(
            chronos.write_ghi_results(
                result_dir, examples, input_metadata, METHOD, prediction, length,
                data_root=args.data_root,
            )
        )
    print(f"Wrote {', '.join(written)}", flush=True)


if __name__ == "__main__":
    main()
