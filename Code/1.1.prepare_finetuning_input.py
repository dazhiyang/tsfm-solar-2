"""Build a pooled BSRN fine-tuning index with explicit train/validation splits.

Each indexed origin is the end of a 60-minute history with a valid 15-minute
target, confined to one UTC calendar day. Shorter histories (15 and 30 minutes)
are suffixes of the same 60-minute context at downstream scripts.

Test station-years are excluded entirely from this index; held-out evaluation
inputs are prepared separately in ``1.2.prepare_forecast_input.py``.

Split policies:
  * Test: ``--test-station`` (all years), ``--test-station-year``, and/or
    ``--test-last-year-per-station`` (each station's latest calendar year held out for 1.2).
  * Validation: 30% of origins per station-year (default with ``--test-last-year-per-station``),
    overridable via ``--validation-fraction``, or ``--no-validation-split``.
  * Legacy: explicit ``--validation-station-year`` (whole files).

Example (Köppen hold-outs + last-year test + 30% validation):

  python -B Code/1.1.prepare_finetuning_input.py \\
    --koppen-holdout-stations --test-last-year-per-station \\
    --overwrite --output-file Data/finetuning_data.parquet
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit


# Analysis parameters
HISTORY_LENGTHS = (15, 30, 60)
MAX_HISTORY_LENGTH = max(HISTORY_LENGTHS)
FORECAST_LENGTH = 15
COMMON_WINDOW_LENGTH = MAX_HISTORY_LENGTH + FORECAST_LENGTH
MINUTES_PER_DAY = 24 * 60
KAPPA_SCALE = 1000
FLAG_VALID = 0
DEFAULT_RANDOM_SEED = 42

# Full-station hold-outs for unseen evaluation (one station per major Köppen group).
KOPPEN_HOLDOUT_STATIONS = ("ASP", "QIQ", "PAL", "KWA", "SYO")
KOPPEN_HOLDOUT_INFO = {
    "ASP": {"koppen": "B", "name": "Alice Springs"},
    "QIQ": {"koppen": "D", "name": "Qiqihar"},
    "PAL": {"koppen": "C", "name": "Palaiseau (SIRTA)"},
    "KWA": {"koppen": "A", "name": "Kwajalein"},
    "SYO": {"koppen": "E", "name": "Syowa"},
}

DEFAULT_VALIDATION_FRACTION = 0.3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data" / "bsrn_1min"

REQUIRED_COLUMNS = ("minute_of_year", "kappa_milli", "kappa_flag")
REQUIRED_TYPES = {
    "minute_of_year": pa.uint32(),
    "kappa_milli": pa.uint16(),
    "kappa_flag": pa.uint8(),
}


@njit
def valid_starts(flags, length):
    """Return starts of windows whose every minute has flag zero."""
    n = len(flags) - length + 1
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    output = np.empty(n, dtype=np.int64)
    invalid = 0
    for offset in range(length):
        invalid += flags[offset] != 0
    count = 0
    for start in range(n):
        if invalid == 0:
            output[count] = start
            count += 1
        if start == n - 1:
            break
        invalid -= flags[start] != 0
        invalid += flags[start + length] != 0
    return output[:count]


@njit
def same_day_mask(starts, minute_of_year, window_length):
    """Keep only windows whose endpoints share one UTC calendar day."""
    mask = np.zeros(len(starts), dtype=np.bool_)
    for index in range(len(starts)):
        start = starts[index]
        day_start = minute_of_year[start] // MINUTES_PER_DAY
        day_end = minute_of_year[start + window_length - 1] // MINUTES_PER_DAY
        mask[index] = day_start == day_end
    return mask


def parse_station_codes(raw_codes, label):
    """Normalize three-letter station codes for full-station holdout."""
    if not raw_codes:
        return set()
    normalized = set()
    for code in raw_codes:
        station_code = str(code).upper()
        if len(station_code) != 3 or not station_code.isalpha():
            raise ValueError(f"{label}: invalid station code {code!r}.")
        normalized.add(station_code)
    return normalized


def parse_station_year_pairs(raw_pairs, label):
    """Normalize (STATION, YEAR) CLI pairs into a set of uppercase keys."""
    if not raw_pairs:
        return set()
    normalized = set()
    for station, year_text in raw_pairs:
        station_code = str(station).upper()
        if len(station_code) != 3 or not station_code.isalpha():
            raise ValueError(f"{label}: invalid station code {station!r}.")
        try:
            year = int(year_text)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}: invalid year {year_text!r}.") from error
        normalized.add((station_code, year))
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory with lowercase station subfolders.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Parquet index path; parent directory must already exist.",
    )
    parser.add_argument(
        "--test-station-year",
        nargs=2,
        action="append",
        metavar=("STATION", "YEAR"),
        default=[],
        help="Held-out test file to exclude from this index (repeatable).",
    )
    parser.add_argument(
        "--test-station",
        action="append",
        metavar="STATION",
        default=[],
        help="Held-out test station: every year excluded from train/validation (repeatable).",
    )
    parser.add_argument(
        "--koppen-holdout-stations",
        action="store_true",
        help=(
            "Add default Köppen hold-outs: ASP (B), QIQ (D), PAL (C), KWA (A), SYO (E). "
            "Combines with --test-station."
        ),
    )
    parser.add_argument(
        "--test-last-year-per-station",
        action="store_true",
        help=(
            "Hold out each station's latest calendar year for script 1.2 (excluded from index). "
            "Skipped for stations already fully held out via --test-station."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help=f"Fraction of origins per station-year for validation (default {DEFAULT_VALIDATION_FRACTION}).",
    )
    parser.add_argument(
        "--validation-station-year",
        nargs=2,
        action="append",
        metavar=("STATION", "YEAR"),
        default=[],
        help="Validation file reserved from the pooled training pool (repeatable).",
    )
    parser.add_argument(
        "--no-validation-split",
        action="store_true",
        help="Explicitly allow zero validation files (no implicit validation rule).",
    )
    parser.add_argument(
        "--station",
        nargs="+",
        help="Restrict discovery to these three-letter station codes.",
    )
    parser.add_argument(
        "--year",
        nargs="+",
        type=int,
        help="Restrict discovery to these calendar years.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many station-year files after sorting.",
    )
    parser.add_argument(
        "--train-subsample-fraction",
        type=float,
        default=1.0,
        help="Keep this fraction of training origins per file (validation untouched).",
    )
    parser.add_argument(
        "--max-train-origins",
        type=int,
        help="After indexing, randomly cap the training split to this many origins.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Seed for per-file training subsampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan inputs and print counts without writing output.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    if not args.data_root.is_dir():
        raise ValueError(f"Data root does not exist: {args.data_root}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if not 0 < args.train_subsample_fraction <= 1.0:
        raise ValueError("--train-subsample-fraction must lie in (0, 1].")
    if args.max_train_origins is not None and args.max_train_origins < 1:
        raise ValueError("--max-train-origins must be positive.")
    validation_keys = parse_station_year_pairs(
        args.validation_station_year, "validation"
    )
    val_modes = (
        args.validation_fraction is not None,
        bool(validation_keys),
        args.no_validation_split,
    )
    if sum(val_modes) > 1:
        raise ValueError(
            "Choose one validation mode: --validation-fraction, "
            "--validation-station-year, or --no-validation-split."
        )
    if args.test_last_year_per_station and not any(val_modes):
        args.validation_fraction = DEFAULT_VALIDATION_FRACTION
    elif not any(val_modes):
        raise ValueError(
            "Specify --test-last-year-per-station, --validation-fraction, "
            "--validation-station-year, or --no-validation-split."
        )
    if args.validation_fraction is not None and not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must lie in (0, 1).")
    if validation_keys and args.no_validation_split:
        raise ValueError(
            "Pass either --validation-station-year entries or --no-validation-split, not both."
        )
    if args.output_file.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            f"Output exists: {args.output_file}. Pass --overwrite to replace it."
        )
    if not args.dry_run and not args.output_file.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: {args.output_file.parent}"
        )


def discover_station_years(data_root, stations=None, years=None, limit=None):
    """Return station/year/path records from existing Parquet filenames."""
    records = []
    for station_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for path in sorted(station_dir.glob("*.parquet")):
            stem = path.stem
            if "_" not in stem:
                continue
            station_token, year_token = stem.rsplit("_", 1)
            try:
                year = int(year_token)
            except ValueError:
                continue
            station_code = station_token.upper()
            records.append(
                {
                    "station": station_code,
                    "year": year,
                    "file_path": str(path.resolve()),
                }
            )
    if not records:
        raise ValueError(f"No station-year parquet files found under {data_root}.")
    frame = pd.DataFrame(records).sort_values(["station", "year"]).reset_index(drop=True)
    if stations:
        selected = {code.upper() for code in stations}
        frame = frame.loc[frame["station"].isin(selected)]
    if years:
        selected_years = {int(value) for value in years}
        frame = frame.loc[frame["year"].isin(selected_years)]
    if limit is not None:
        frame = frame.head(limit)
    if frame.empty:
        raise ValueError("No station-year files matched the requested filters.")
    if frame.duplicated(["station", "year"]).any():
        raise ValueError("Duplicate station-year paths were discovered.")
    return frame.reset_index(drop=True)


def expand_test_keys(records, test_station_codes, test_station_years):
    """Union explicit test years with every year of each held-out test station."""
    keys = set(test_station_years)
    if not test_station_codes:
        return keys
    for row in records.itertuples(index=False):
        if row.station in test_station_codes:
            keys.add((row.station, int(row.year)))
    return keys


def add_last_year_test_keys(records, test_keys, full_station_holdouts):
    """Add each partial station's latest calendar year to the held-out test set."""
    keys = set(test_keys)
    for station, group in records.groupby("station", sort=True):
        station = str(station)
        if station in full_station_holdouts:
            continue
        keys.add((station, int(group["year"].max())))
    return keys


def pool_after_test_exclusions(records, test_keys):
    """Station-years that remain for train/validation indexing."""
    keep = [
        (str(station), int(year)) not in test_keys
        for station, year in zip(records["station"], records["year"])
    ]
    pool = records.loc[keep].copy()
    if pool.empty:
        raise ValueError("No station-years remain after test exclusions.")
    return pool.reset_index(drop=True)


def assign_origin_validation_fraction(index_frame, fraction, seed):
    """Randomly label a fraction of origins per station-year as validation."""
    if index_frame.empty:
        return index_frame
    rng = np.random.default_rng(seed)
    split = index_frame["split"].to_numpy(copy=True)
    for positions in index_frame.groupby(["station", "year"], sort=True).groups.values():
        idx = np.fromiter(positions, dtype=np.int64)
        n = len(idx)
        if n < 2:
            continue
        n_val = min(n - 1, max(1, int(round(n * fraction))))
        val_local = rng.choice(n, size=n_val, replace=False)
        split[idx[val_local]] = "validation"
    out = index_frame.copy()
    out["split"] = split
    return out


def validation_keys_from_index(index_frame):
    """Station-years that contribute at least one validation origin."""
    val = index_frame.loc[index_frame["split"] == "validation"]
    if val.empty:
        return set()
    return {
        (str(station), int(year))
        for station, year in zip(val["station"], val["year"])
    }


def partition_records(records, test_keys, validation_keys):
    """Assign split labels; test files are dropped from the returned frame."""
    overlap = test_keys & validation_keys
    if overlap:
        formatted = ", ".join(f"{station}-{year}" for station, year in sorted(overlap))
        raise ValueError(f"Test and validation splits overlap: {formatted}.")
    keys = list(zip(records["station"], records["year"]))
    split = []
    keep = []
    for station, year in keys:
        key = (station, year)
        if key in test_keys:
            keep.append(False)
            continue
        keep.append(True)
        if key in validation_keys:
            split.append("validation")
        else:
            split.append("train")
    partitioned = records.loc[keep].copy()
    partitioned["split"] = split
    return partitioned.reset_index(drop=True)


def assert_station_year_schema(path):
    """Require the compact one-minute schema produced by 0.3.prepare_BSRN."""
    schema = pq.read_schema(path)
    for name, dtype in REQUIRED_TYPES.items():
        if name not in schema.names:
            raise ValueError(f"Missing column {name!r} in {path}.")
        if schema.field(name).type != dtype:
            raise ValueError(f"Unexpected dtype for {name!r} in {path}.")
    metadata = schema.metadata or {}
    if metadata.get(b"year") is not None:
        file_year = int(metadata[b"year"].decode())
        if file_year != int(path.stem.rsplit("_", 1)[1]):
            raise ValueError(f"Metadata year mismatch in {path}.")


def load_station_year_arrays(path):
    """Read minute grid, flags, and verify the one-minute UTC year grid."""
    table = pq.read_table(path, columns=list(REQUIRED_COLUMNS))
    minute_of_year = table.column("minute_of_year").to_numpy(zero_copy_only=False)
    flags = table.column("kappa_flag").to_numpy(zero_copy_only=False)
    if len(minute_of_year) == 0:
        raise ValueError(f"Empty station-year file: {path}")
    if minute_of_year[0] != 0 or np.any(np.diff(minute_of_year) != 1):
        raise ValueError(f"Incomplete minute grid in {path}")
    return minute_of_year.astype(np.int64), flags.astype(np.uint8)


def eligible_origins(minute_of_year, flags):
    """Return origin indices (end of 60-minute history) for valid 75-minute windows."""
    window_starts = valid_starts(flags, COMMON_WINDOW_LENGTH)
    if len(window_starts) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint32)
    keep = same_day_mask(window_starts, minute_of_year, COMMON_WINDOW_LENGTH)
    window_starts = window_starts[keep]
    if len(window_starts) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint32)
    origins = window_starts + np.int64(MAX_HISTORY_LENGTH - 1)
    origin_minute = minute_of_year[origins].astype(np.uint32)
    return origins, origin_minute


def per_file_rng(seed, station, year):
    """Derive a stable generator for one station-year training subsample."""
    token = f"{seed}|{station}|{year}".encode()
    word = int.from_bytes(token[:8].ljust(8, b"\0"), "little", signed=False)
    return np.random.default_rng(word & np.uint32.max)


def subsample_training_indices(origin_count, fraction, rng):
    if fraction >= 1.0:
        return np.arange(origin_count, dtype=np.int64)
    if origin_count == 0:
        return np.empty(0, dtype=np.int64)
    mask = rng.random(origin_count) < fraction
    return np.flatnonzero(mask)


def index_station_year(row, train_subsample_fraction, seed):
    """Build index rows for one station-year file."""
    path = Path(row.file_path)
    assert_station_year_schema(path)
    minute_of_year, flags = load_station_year_arrays(path)
    origins, origin_minute = eligible_origins(minute_of_year, flags)
    if len(origins) == 0:
        return pd.DataFrame(
            columns=[
                "station",
                "year",
                "file_path",
                "origin_index",
                "origin_minute_of_year",
                "split",
            ]
        )
    if row.split == "train" and train_subsample_fraction < 1.0:
        rng = per_file_rng(seed, row.station, int(row.year))
        keep = subsample_training_indices(len(origins), train_subsample_fraction, rng)
        origins = origins[keep]
        origin_minute = origin_minute[keep]
    return pd.DataFrame(
        {
            "station": row.station,
            "year": np.int32(row.year),
            "file_path": row.file_path,
            "origin_index": origins.astype(np.int32),
            "origin_minute_of_year": origin_minute,
            "split": row.split,
        }
    )


def cap_training_origins(index_frame, max_train, seed):
    """Uniformly subsample the training split to at most max_train rows."""
    if max_train is None:
        return index_frame
    train_mask = index_frame["split"].to_numpy() == "train"
    n_train = int(train_mask.sum())
    if n_train <= max_train:
        return index_frame
    positions = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    keep_positions = np.sort(rng.choice(positions, max_train, replace=False))
    keep = np.zeros(len(index_frame), dtype=bool)
    keep[keep_positions] = True
    keep |= ~train_mask
    return index_frame.loc[keep].reset_index(drop=True)


def prepare_training_index(partitioned_records, train_subsample_fraction, seed):
    """Index eligible origins for every training or validation station-year."""
    frames = []
    for row in partitioned_records.itertuples(index=False):
        frames.append(
            index_station_year(row, train_subsample_fraction, seed)
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "station",
                "year",
                "file_path",
                "origin_index",
                "origin_minute_of_year",
                "split",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def split_summary(index_frame):
    """Summarize indexed origins by split and station-year."""
    if index_frame.empty:
        return pd.DataFrame(columns=["split", "station", "year", "n_origins"])
    grouped = (
        index_frame.groupby(["split", "station", "year"], as_index=False)
        .size()
        .rename(columns={"size": "n_origins"})
        .sort_values(["split", "station", "year"])
    )
    return grouped


def build_metadata(
    args,
    test_keys,
    validation_keys,
    records,
    partitioned,
    index_frame,
    *,
    validation_rule,
    test_station_codes,
    excluded_single_year_stations,
):
    """Attach reproducible split configuration to the parquet file."""
    return {
        "script": Path(__file__).name,
        "max_history_length": MAX_HISTORY_LENGTH,
        "history_lengths": list(HISTORY_LENGTHS),
        "forecast_length": FORECAST_LENGTH,
        "common_window_length": COMMON_WINDOW_LENGTH,
        "kappa_scale": KAPPA_SCALE,
        "kappa_valid_flag": FLAG_VALID,
        "random_seed": args.random_seed,
        "train_subsample_fraction": args.train_subsample_fraction,
        "max_train_origins": args.max_train_origins,
        "validation_rule": validation_rule,
        "validation_fraction": args.validation_fraction,
        "test_last_year_per_station": args.test_last_year_per_station,
        "test_stations": sorted(test_station_codes),
        "koppen_holdout_stations": {
            code: KOPPEN_HOLDOUT_INFO[code]
            for code in sorted(test_station_codes)
            if code in KOPPEN_HOLDOUT_INFO
        },
        "excluded_single_year_stations": excluded_single_year_stations,
        "test_station_years": sorted(
            [{"station": s, "year": y} for s, y in test_keys],
            key=lambda item: (item["station"], item["year"]),
        ),
        "validation_station_years": sorted(
            [{"station": s, "year": y} for s, y in validation_keys],
            key=lambda item: (item["station"], item["year"]),
        ),
        "n_station_years_discovered": int(len(records)),
        "n_station_years_indexed": int(len(partitioned)),
        "n_origins_total": int(len(index_frame)),
        "n_origins_train": int((index_frame["split"] == "train").sum()),
        "n_origins_validation": int((index_frame["split"] == "validation").sum()),
    }


def write_index(index_frame, output_file, metadata):
    """Write the index atomically without creating parent directories."""
    table = pa.Table.from_pandas(index_frame, preserve_index=False)
    encoded = {
        key.encode(): json.dumps(value, sort_keys=True).encode()
        for key, value in metadata.items()
    }
    table = table.replace_schema_metadata(encoded)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, output_file)


def resolve_test_station_codes(args):
    codes = parse_station_codes(args.test_station, "test")
    if args.koppen_holdout_stations:
        codes |= set(KOPPEN_HOLDOUT_STATIONS)
    return codes


def main():
    args = parse_args()
    test_station_codes = resolve_test_station_codes(args)

    records = discover_station_years(
        args.data_root,
        stations=args.station,
        years=args.year,
        limit=args.limit,
    )
    test_keys = expand_test_keys(
        records,
        test_station_codes,
        parse_station_year_pairs(args.test_station_year, "test"),
    )
    if args.test_last_year_per_station:
        test_keys = add_last_year_test_keys(records, test_keys, test_station_codes)
    discovered_stations = set(records["station"])
    missing_test_stations = test_station_codes - discovered_stations
    if missing_test_stations:
        formatted = ", ".join(sorted(missing_test_stations))
        raise ValueError(f"Test stations not found under data root: {formatted}")
    missing_test = test_keys - {
        (row.station, int(row.year)) for row in records.itertuples(index=False)
    }
    if missing_test:
        formatted = ", ".join(f"{s}-{y}" for s, y in sorted(missing_test))
        raise ValueError(f"Test station-years not found under data root: {formatted}")

    pool = pool_after_test_exclusions(records, test_keys)
    excluded_single_year_stations = []
    explicit_validation = parse_station_year_pairs(args.validation_station_year, "validation")
    if args.no_validation_split:
        partitioned = pool.copy()
        partitioned["split"] = "train"
        validation_keys = set()
        validation_rule = "none"
    elif args.validation_fraction is not None:
        partitioned = pool.copy()
        partitioned["split"] = "train"
        validation_keys = set()
        validation_rule = f"fraction_{args.validation_fraction:g}"
    else:
        validation_keys = explicit_validation
        missing_validation = validation_keys - {
            (str(row.station), int(row.year)) for row in pool.itertuples(index=False)
        }
        if missing_validation:
            formatted = ", ".join(f"{s}-{y}" for s, y in sorted(missing_validation))
            raise ValueError(
                f"Validation station-years not in train pool after test exclusions: {formatted}"
            )
        partitioned = partition_records(pool, set(), validation_keys)
        validation_rule = "explicit"

    index_frame = prepare_training_index(
        partitioned,
        args.train_subsample_fraction,
        args.random_seed,
    )
    if args.validation_fraction is not None:
        index_frame = assign_origin_validation_fraction(
            index_frame, args.validation_fraction, args.random_seed
        )
        validation_keys = validation_keys_from_index(index_frame)
    index_frame = cap_training_origins(
        index_frame, args.max_train_origins, args.random_seed
    )
    metadata = build_metadata(
        args,
        test_keys,
        validation_keys,
        records,
        partitioned,
        index_frame,
        validation_rule=validation_rule,
        test_station_codes=test_station_codes,
        excluded_single_year_stations=excluded_single_year_stations,
    )
    summary = split_summary(index_frame)

    print(f"Discovered station-years: {len(records)}", flush=True)
    if test_station_codes:
        print(f"Held-out test stations: {', '.join(sorted(test_station_codes))}", flush=True)
    print(f"Excluded test station-years: {len(test_keys)}", flush=True)
    print(f"Validation rule: {validation_rule}", flush=True)
    if excluded_single_year_stations:
        print(
            f"Excluded single-year stations ({len(excluded_single_year_stations)}): "
            f"{', '.join(excluded_single_year_stations)}",
            flush=True,
        )
    print(f"Indexed station-years: {len(partitioned)}", flush=True)
    print(f"Training origins: {metadata['n_origins_train']}", flush=True)
    print(f"Validation origins: {metadata['n_origins_validation']}", flush=True)
    if not summary.empty:
        print(summary.to_string(index=False), flush=True)

    if args.dry_run:
        print("Dry run: no output written.", flush=True)
        return

    write_index(index_frame, args.output_file, metadata)
    print(f"Wrote {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
