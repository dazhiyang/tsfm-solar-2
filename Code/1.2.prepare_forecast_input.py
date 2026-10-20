"""Prepare common BSRN forecast examples; print a summary unless --output-file.

Examples (run from the project root with the tf_timesfm interpreter):
  python -B Code/1.2.prepare_forecast_input.py
  python -B Code/1.2.prepare_forecast_input.py --stations CAB --years 2025 --max-origins 0 \\
    --output-file Data/forecast_input/cab_2025.npz
  python -B Code/1.2.prepare_forecast_input.py --self-test

Saved evaluation inputs live under Data/forecast_input/ (not under bsrn_1min/).

The default CLI is still a small CAB pilot sample. For held-out evaluation matching
script 1.1 test policy, use ``select_holdout_station_years`` (latest calendar year
per station, including Köppen codes) or run the forecasting scripts with ``--stations``.
Omit --years to select each station's latest available file. --max-origins 0 selects
all eligible origins after other filters; a positive count caps the sample PER
station-year. --non-overlapping keeps origins whose full 60+15-minute windows
do not share any minute (greedy scan in time order).
Saving requires an explicit approved .npz filename in an existing directory.
No directories are created and existing files are never overwritten.

NPZ contract (allow_pickle=False):
  history: float32 (N, 60), clear-sky index at t-59,...,t
  target: float32 (N, 15), clear-sky index at t+1,...,t+15
  station, year: (N,) source identifiers
  origin_utc_minute: int64 (N,), minutes since 1970-01-01 00:00 UTC
  origin_row: int64 (N,), original Parquet row of t
  source_index: int32 (N,), index into metadata.sources
  metadata_json: scalar JSON string with settings, source hashes and dataset ID

Use histories_for(examples, L) to obtain 15/30/60-minute histories as suffixes.
The selected files are held-out candidates only; training preparation must
explicitly exclude them. This script does not decide the final ten stations.
"""

import argparse
import calendar
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data" / "bsrn_1min"
FORECAST_INPUT_DIR = PROJECT_ROOT / "Data" / "forecast_input"
PILOT_STATION = "CAB"
PILOT_YEAR = 2025
HISTORY_LENGTHS = (15, 30, 60)
COMMON_HISTORY_LENGTH = max(HISTORY_LENGTHS)
FORECAST_LENGTH = 15
RANDOM_SEED = 42
MAX_ORIGINS = 8
ARRAY_KEYS = ("history", "target", "station", "year", "origin_utc_minute",
              "origin_row", "source_index")


def eligible_origins(minutes, flags, history_length=60, forecast_length=15):
    """Return ROW indices t for fully valid, consecutive, same-UTC-day windows."""
    minutes, flags = np.asarray(minutes), np.asarray(flags)
    if (minutes.ndim != 1 or flags.ndim != 1 or len(minutes) != len(flags)
            or not np.issubdtype(minutes.dtype, np.integer)
            or not np.issubdtype(flags.dtype, np.integer)):
        raise ValueError("minutes/flags must be equally sized 1-D integer arrays.")
    if history_length < 1 or forecast_length < 1:
        raise ValueError("History and forecast lengths must be positive.")
    span = history_length + forecast_length
    if len(minutes) < span:
        return np.empty(0, dtype=np.int64)
    minutes = minutes.astype(np.int64, copy=False)
    starts = np.arange(len(minutes) - span + 1, dtype=np.int64)
    ends = starts + span - 1
    invalid = np.r_[0, np.cumsum(flags != 0, dtype=np.int64)]
    breaks = np.r_[0, np.cumsum(np.diff(minutes) != 1, dtype=np.int64)]
    keep = ((invalid[ends + 1] == invalid[starts])
            & (breaks[ends] == breaks[starts])
            & (minutes[starts] // 1440 == minutes[ends] // 1440))
    return starts[keep] + history_length - 1


def non_overlapping_origins(origins, history_length=60, forecast_length=15):
    """Greedy earliest-first subset with disjoint history+target minute spans.

    For origin row t the used rows are t-history_length+1 through t+forecast_length.
    The next origin must start strictly after the previous target ends.
    """
    origins = np.sort(np.unique(np.asarray(origins, dtype=np.int64)))
    if len(origins) == 0:
        return origins
    span = history_length + forecast_length
    selected = []
    index = 0
    while index < len(origins):
        origin = int(origins[index])
        selected.append(origin)
        index = int(np.searchsorted(origins, origin + span, side="left"))
    return np.asarray(selected, dtype=np.int64)


def finetune_index_module():
    path = Path(__file__).with_name("1.1.prepare_finetuning_input.py")
    spec = importlib.util.spec_from_file_location("bsrn_finetune_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def available_station_years(data_root, station):
    """Sorted calendar years with a parquet file for one station."""
    station = station.upper()
    pattern = re.compile(rf"{re.escape(station.lower())}_(\d{{4}})\.parquet")
    years = sorted(
        int(match.group(1))
        for path in (Path(data_root) / station.lower()).glob("*.parquet")
        if (match := pattern.fullmatch(path.name))
    )
    if not years:
        raise ValueError(f"No processed files found for {station}.")
    return years


def select_station_years(data_root, stations, years=None):
    """Resolve requested files; missing years are not silently substituted."""
    stations = [station.upper() for station in stations]
    if (not stations or len(set(stations)) != len(stations)
            or any(not re.fullmatch(r"[A-Z0-9]+", code) for code in stations)):
        raise ValueError("Specify unique alphanumeric station codes.")
    if years is not None and len(years) != len(stations):
        raise ValueError("Provide one --years value for each station, in order.")
    selected = []
    for number, station in enumerate(stations):
        if years is not None:
            year = int(years[number])
        else:
            year = max(available_station_years(data_root, station))
        path = Path(data_root) / station.lower() / f"{station.lower()}_{year}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append((station, year))
    return selected


def select_holdout_station_years(data_root, stations, years=None):
    """Latest calendar year per station (1.1 ``--test-last-year-per-station`` eval set)."""
    return select_station_years(data_root, stations, years=years)


def resolve_eval_station_list(stations, koppen_holdout_stations=False):
    """Merge explicit stations with optional default Köppen full hold-outs."""
    chosen = list(dict.fromkeys(code.upper() for code in (stations or [])))
    if koppen_holdout_stations:
        chosen = list(dict.fromkeys(chosen + list(finetune_index_module().KOPPEN_HOLDOUT_STATIONS)))
    if not chosen:
        raise ValueError("Specify --stations and/or --koppen-holdout-stations.")
    return chosen


def add_eval_input_arguments(parser):
    """Arguments for 1.3/1.4: either --input-file or held-out station-years from disk."""
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Saved .npz from this script (mutually exclusive with --stations).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="BSRN parquet root when building hold-out examples in memory.",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        help="Station codes to evaluate (hold-out years per script 1.1 policy).",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Optional one year per station; overrides automatic hold-out years.",
    )
    parser.add_argument(
        "--koppen-holdout-stations",
        action="store_true",
        help="Add ASP, QIQ, PAL, KWA, SYO (latest year each) to the station list.",
    )
    parser.add_argument(
        "--max-origins",
        type=int,
        default=0,
        help="Cap origins per station-year (0 = all eligible).",
    )
    parser.add_argument("--stride-minutes", type=int, default=1)
    parser.add_argument(
        "--non-overlapping",
        action="store_true",
        help="Disjoint 60+15-minute windows (greedy scan).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)


def load_eval_examples(
    *,
    input_file=None,
    data_root=DATA_ROOT,
    stations=None,
    years=None,
    koppen_holdout_stations=False,
    max_origins=0,
    seed=RANDOM_SEED,
    stride_minutes=1,
    non_overlapping=False,
):
    """Load a saved NPZ or build in-memory examples for held-out station-years."""
    if input_file is not None:
        examples = load_examples(input_file)
        metadata = json.loads(examples["metadata_json"].item())
        return examples, metadata
    station_list = resolve_eval_station_list(stations, koppen_holdout_stations)
    station_years = select_holdout_station_years(data_root, station_list, years=years)
    examples = prepare_examples(
        station_years,
        data_root,
        max_origins,
        seed,
        stride_minutes,
        non_overlapping,
    )
    metadata = json.loads(examples["metadata_json"].item())
    return examples, metadata


def source_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_id(examples, metadata):
    """Fingerprint values, origins and preparation settings, not just a filename."""
    metadata = {key: value for key, value in metadata.items() if key != "dataset_id"}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for key in ARRAY_KEYS:
        array = np.asarray(examples[key])
        digest.update(f"{key}|{array.dtype.str}|{array.shape}".encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def prepare_examples(station_years, data_root=DATA_ROOT, max_origins=MAX_ORIGINS,
                     seed=RANDOM_SEED, stride_minutes=1, non_overlapping=False):
    """Prepare paired examples, with uniform sampling after the cadence filter.

    A positive cap applies separately to each station-year; it does NOT yield a
    uniform pooled-archive sample. Selected origins are stored in temporal order.
    Cadence is anchored to UTC minute-of-year, not positions in a filtered table.
    Non-overlapping selection runs after cadence filtering and before any cap.
    """
    if max_origins < 0 or seed < 0 or stride_minutes < 1:
        raise ValueError("Use a nonnegative cap/seed and positive stride.")
    if not station_years or len(set(station_years)) != len(station_years):
        raise ValueError("Specify at least one unique station-year.")
    data_root = Path(data_root)
    rng = np.random.default_rng(seed)
    pieces = {key: [] for key in ARRAY_KEYS}
    sources = []
    for source_index, (station, year) in enumerate(station_years):
        station, year = station.upper(), int(year)
        if not re.fullmatch(r"[A-Z0-9]+", station) or not 1900 <= year <= 2200:
            raise ValueError("Invalid station code or year.")
        path = data_root / station.lower() / f"{station.lower()}_{year}.parquet"
        before = path.stat()
        table = pq.read_table(path, columns=["minute_of_year", "kappa_milli", "kappa_flag"])
        types = {"minute_of_year": pa.uint32(), "kappa_milli": pa.uint16(),
                 "kappa_flag": pa.uint8()}
        if any(table.schema.field(key).type != kind or table[key].null_count
               for key, kind in types.items()):
            raise ValueError(f"Unexpected schema or null observations: {path}")
        minutes = table["minute_of_year"].to_numpy().astype(np.int64)
        codes = table["kappa_milli"].to_numpy()
        flags = table["kappa_flag"].to_numpy()
        year_minutes = (366 if calendar.isleap(year) else 365) * 1440
        if (not len(minutes) or minutes[0] < 0 or minutes[-1] >= year_minutes
                or np.any(np.diff(minutes) <= 0)):
            raise ValueError(f"Minute indices must be unique, increasing and within {year}: {path}")
        if np.any((flags == 0) & (codes == 65535)):
            raise ValueError(f"Invalid storage sentinel marked valid: {path}")
        origins = eligible_origins(minutes, flags, COMMON_HISTORY_LENGTH, FORECAST_LENGTH)
        eligible_count = len(origins)
        origins = origins[minutes[origins] % stride_minutes == 0]
        cadence_count = len(origins)
        if not cadence_count:
            raise ValueError(f"No eligible origins at the requested cadence: {path}")
        if non_overlapping:
            origins = non_overlapping_origins(
                origins, COMMON_HISTORY_LENGTH, FORECAST_LENGTH)
        nonoverlap_count = len(origins)
        if not nonoverlap_count:
            raise ValueError(f"No non-overlapping origins remain: {path}")
        if max_origins and len(origins) > max_origins:
            origins = np.sort(rng.choice(origins, max_origins, replace=False))
        history_rows = origins[:, None] + np.arange(1 - COMMON_HISTORY_LENGTH, 1)
        target_rows = origins[:, None] + np.arange(1, FORECAST_LENGTH + 1)
        epoch_minute = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() // 60)
        count = len(origins)
        pieces["history"].append(codes[history_rows].astype(np.float32) / 1000)
        pieces["target"].append(codes[target_rows].astype(np.float32) / 1000)
        pieces["station"].append(np.full(count, station, dtype="U8"))
        pieces["year"].append(np.full(count, year, dtype=np.int32))
        pieces["origin_utc_minute"].append(minutes[origins] + epoch_minute)
        pieces["origin_row"].append(origins)
        pieces["source_index"].append(np.full(count, source_index, dtype=np.int32))
        checksum = source_hash(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError(f"Source changed during preparation: {path}")
        sources.append(dict(station=station, year=year, path=str(path.resolve()),
                            sha256=checksum, size_bytes=after.st_size,
                            mtime_ns=after.st_mtime_ns, n_rows=len(minutes),
                            eligible_count=eligible_count, cadence_count=cadence_count,
                            nonoverlap_count=nonoverlap_count, selected_count=count))
    examples = {key: np.concatenate(value) for key, value in pieces.items()}
    common_window_span = COMMON_HISTORY_LENGTH + FORECAST_LENGTH
    if non_overlapping:
        sampling = ("non-overlapping "
                    + ("all" if max_origins == 0 else "capped per station-year"))
    else:
        sampling = "all" if max_origins == 0 else "capped separately per station-year"
    metadata = dict(format_version=1, target="stored_clear_sky_index", kappa_scale=1000,
                    history_lengths=list(HISTORY_LENGTHS), common_history_length=60,
                    forecast_length=15, common_window_span=common_window_span,
                    timezone="UTC", validity_flag=0,
                    consecutive_minutes=True, same_day_history_and_target=True,
                    seed=seed, stride_minutes=stride_minutes, non_overlapping=non_overlapping,
                    max_origins_per_station_year=max_origins, sampling=sampling,
                    split_status="selected held-out candidates; training must exclude these files",
                    sources=sources)
    metadata["dataset_id"] = dataset_id(examples, metadata)
    examples["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    validate_examples(examples)
    return examples


def histories_for(examples, length):
    """Return a view; do not resample origins or pad across missing observations."""
    if length not in HISTORY_LENGTHS:
        raise ValueError(f"History length must be one of {HISTORY_LENGTHS}.")
    return examples["history"][:, -length:]


def validate_examples(examples):
    """Validate saved-array consistency; validity/continuity are checked at source."""
    metadata = json.loads(str(np.asarray(examples["metadata_json"]).item()))
    n = len(examples["history"])
    if (metadata["format_version"] != 1 or metadata["common_history_length"] != 60
            or metadata["forecast_length"] != 15 or metadata["timezone"] != "UTC"
            or not n or examples["history"].shape != (n, 60)
            or examples["target"].shape != (n, 15)):
        raise ValueError("Unexpected forecasting input format or shape.")
    for key in ARRAY_KEYS[2:]:
        if examples[key].shape != (n,):
            raise ValueError(f"Inconsistent example count: {key}")
    for key in ("history", "target"):
        if not np.isfinite(examples[key]).all() or np.any(examples[key] < 0):
            raise ValueError(f"Nonfinite or negative {key} values.")
    for key in ("year", "origin_utc_minute", "origin_row", "source_index"):
        if not np.issubdtype(examples[key].dtype, np.integer):
            raise ValueError(f"Expected integer {key}.")
    times = examples["origin_utc_minute"]
    if np.any((times - 59) // 1440 != (times + 15) // 1440):
        raise ValueError("History and target cross a UTC day boundary.")
    indices = examples["source_index"]
    if np.any(indices < 0) or np.any(indices >= len(metadata["sources"])):
        raise ValueError("Unknown source index.")
    pairs = [(s["station"], s["year"]) for s in metadata["sources"]]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate station-year sources.")
    for i, source in enumerate(metadata["sources"]):
        mask = indices == i
        rows = examples["origin_row"][mask]
        start = int(datetime(source["year"], 1, 1, tzinfo=timezone.utc).timestamp() // 60)
        stop = int(datetime(source["year"] + 1, 1, 1, tzinfo=timezone.utc).timestamp() // 60)
        if (int(mask.sum()) != source["selected_count"]
                or np.any(examples["station"][mask] != source["station"])
                or np.any(examples["year"][mask] != source["year"])
                or np.any(rows < 59) or np.any(rows + 15 >= source["n_rows"])
                or np.any(times[mask] - 59 < start) or np.any(times[mask] + 15 >= stop)
                or np.any(np.diff(times[mask]) <= 0) or np.any(np.diff(rows) <= 0)):
            raise ValueError(f"Inconsistent source/origin metadata: {source['station']}")
        if metadata.get("non_overlapping") and len(rows) > 1:
            span = int(metadata.get("common_window_span", 75))
            if np.any(rows[1:] - rows[:-1] < span):
                raise ValueError(
                    f"Non-overlapping origins overlap for {source['station']}."
                )
    if metadata["dataset_id"] != dataset_id(examples, metadata):
        raise ValueError("Dataset fingerprint does not match the stored examples.")
    return metadata


def load_examples(path):
    """Load the shared input without pickle; validate before any model uses it."""
    with np.load(path, allow_pickle=False) as saved:
        examples = {key: saved[key] for key in (*ARRAY_KEYS, "metadata_json")}
    validate_examples(examples)
    return examples


def self_test():
    """Pure in-memory checks against a direct, independent window enumeration."""
    rng = np.random.default_rng(17)
    for history in (1, 3, 60):
        for horizon in (1, 15):
            for scenario in range(4):
                minutes = 1370 + np.arange(300, dtype=np.int64)
                flags = np.zeros(300, dtype=np.uint8)
                if scenario & 1:
                    flags[rng.choice(300, 15, replace=False)] = 4
                if scenario & 2:
                    minutes[150:] += 2
                expected = []
                for t in range(history - 1, len(minutes) - horizon):
                    sl = slice(t - history + 1, t + horizon + 1)
                    if (np.all(flags[sl] == 0) and np.all(np.diff(minutes[sl]) == 1)
                            and minutes[t - history + 1] // 1440 == minutes[t + horizon] // 1440):
                        expected.append(t)
                np.testing.assert_array_equal(eligible_origins(minutes, flags, history, horizon),
                                              np.asarray(expected, dtype=np.int64))
    np.testing.assert_array_equal(eligible_origins(np.arange(75), np.zeros(75, np.uint8)), [59])
    assert not len(eligible_origins(np.arange(74), np.zeros(74, np.uint8)))
    assert not len(eligible_origins(np.arange(1400, 1475), np.zeros(75, np.uint8)))
    dense = np.arange(59, 300, dtype=np.int64)
    picked = non_overlapping_origins(dense)
    np.testing.assert_array_equal(picked, np.arange(59, 300, 75, dtype=np.int64))
    sparse = np.array([59, 60, 134, 200], dtype=np.int64)
    np.testing.assert_array_equal(non_overlapping_origins(sparse), np.array([59, 134]))
    print("Eligibility checks passed (flags, timestamp gaps, UTC midnight, exact boundaries, non-overlap).")


def discover_station_codes(data_root):
    """Three-letter codes from lowercase subfolders under the parquet root."""
    root = Path(data_root)
    return sorted(
        path.name.upper()
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--stations", nargs="+", default=[PILOT_STATION])
    parser.add_argument(
        "--all-stations",
        action="store_true",
        help="Use every station directory under --data-root (overrides default CAB pilot).",
    )
    parser.add_argument(
        "--holdout-years",
        action="store_true",
        help="Latest calendar year per station (script 1.1 test-year policy).",
    )
    parser.add_argument("--years", nargs="+", type=int,
                        help="One year per station; default: latest available year of each.")
    parser.add_argument("--max-origins", type=int, default=MAX_ORIGINS,
                        help="Maximum per station-year, default 8 (pilot); 0 means all.")
    parser.add_argument("--stride-minutes", type=int, default=1)
    parser.add_argument(
        "--non-overlapping",
        action="store_true",
        help="Keep disjoint 60+15-minute windows (greedy temporal scan).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-file", type=Path,
                        help="Explicit .npz filename; without this, prepare in memory only.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.max_origins < 0 or args.seed < 0 or args.stride_minutes < 1:
        parser.error("Use nonnegative --max-origins/--seed and positive --stride-minutes.")
    if args.output_file is not None:
        if args.output_file.suffix.lower() != ".npz" or not args.output_file.parent.is_dir():
            parser.error("Output must be a .npz file in an existing directory.")
        if args.output_file.exists():
            parser.error("Output already exists; it will not be overwritten.")
    if args.all_stations:
        stations = discover_station_codes(args.data_root)
        if not stations:
            parser.error(f"No station folders under {args.data_root}")
    else:
        stations = args.stations
    if args.years is not None and len(args.years) != len(stations):
        parser.error("Provide one --years value for each station.")
    if args.holdout_years:
        selected = select_holdout_station_years(args.data_root, stations, args.years)
    else:
        selected = select_station_years(args.data_root, stations, args.years)
    examples = prepare_examples(
        selected,
        args.data_root,
        args.max_origins,
        args.seed,
        args.stride_minutes,
        args.non_overlapping,
    )
    metadata = json.loads(examples["metadata_json"].item())
    for source in metadata["sources"]:
        line = (f"{source['station']} {source['year']}: {source['eligible_count']:,} eligible; "
                f"{source['cadence_count']:,} after cadence")
        if metadata.get("non_overlapping"):
            line += f"; {source['nonoverlap_count']:,} non-overlapping"
        print(f"{line}; {source['selected_count']:,} selected")
    print(f"Common histories: {examples['history'].shape}; targets: {examples['target'].shape}")
    print(f"Dataset ID: {metadata['dataset_id']}")
    if args.output_file is None:
        print("In-memory preparation only. No files or folders created.")
    else:
        with args.output_file.open("xb") as stream:
            np.savez_compressed(stream, **examples)
        print(f"Saved {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
