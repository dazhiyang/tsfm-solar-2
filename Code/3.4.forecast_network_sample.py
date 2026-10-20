"""Sample common network hold-out origins and forecast with three L60 methods.

  python -B Code/3.4.forecast_network_sample.py --self-test
  python -B Code/3.4.forecast_network_sample.py --stations ABS --samples-per-station 3 --no-save
  python -B Code/3.4.forecast_network_sample.py

Default: up to 1000 origins per station, uniformly without replacement, seed42.
Stations with fewer origins use all available origins. Samples can overlap in
time. Pretrained TimesFM, full-pass TimesFM-BSRN and persistence use identical
origins. No training, model downloads, directories, or per-station files.
One Parquet export retains Station, Year, Time (UTC epoch minutes), OriginRow,
InputRow, Method, Obs1..15 and Fcst1..15 in GHI (W m-2, two decimal places).
Existing output is never overwritten. --no-save writes no results.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HISTORY_LENGTH = 60
FORECAST_LENGTH = 15
METHODS = ("Persistence", "TimesFM", "TimesFM-BSRN")


def sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_indices(source_indices, sources, count, seed, stations=None):
    """Station/year-specific seeds make selection independent of station order."""
    selected, counts = [], []
    wanted = set(stations) if stations else {s["station"] for s in sources}
    if not wanted.issubset({s["station"] for s in sources}):
        raise ValueError("Requested station is absent from the network input.")
    if len({s["station"] for s in sources}) != len(sources):
        raise ValueError("Expected one reserved year per station.")
    for index, source in enumerate(sources):
        station, year = source["station"], int(source["year"])
        if station not in wanted:
            continue
        available = np.flatnonzero(source_indices == index)
        if len(available) != source["selected_count"] or not len(available):
            raise ValueError(f"Inconsistent available-origin count: {station}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, year, *station.encode()]))
        chosen = np.sort(rng.choice(available, min(count, len(available)), replace=False))
        selected.append(chosen)
        counts.append(dict(station=station, year=year, available=len(available), sampled=len(chosen)))
    return np.concatenate(selected), counts


def self_test():
    sources = [dict(station="AAA", year=2020, selected_count=5),
               dict(station="BBB", year=2021, selected_count=2)]
    indices = np.array([0]*5 + [1]*2)
    a, counts = sample_indices(indices, sources, 3, 42)
    b, _ = sample_indices(indices, sources, 3, 42)
    c, _ = sample_indices(indices, sources, 3, 42, ["AAA"])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:3], c)
    assert len(a) == len(set(a)) == 5 and counts[1]["sampled"] == 2
    print("PASS: reproducible sampling, no duplicates, station filtering, and small stations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", type=Path, default=ROOT/"Data/forecast_input/network_holdout.npz")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT/"Data/models/timesfm-2.5-full-L60")
    parser.add_argument("--output-file", type=Path, default=ROOT/"Data/result/network_1000_timesfm_L60.parquet")
    parser.add_argument("--samples-per-station", type=int, default=1000)
    parser.add_argument("--stations", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="cpu")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.samples_per_station, args.batch_size, args.threads) < 1 or args.seed < 0:
        parser.error("Counts must be positive and seed nonnegative.")
    if not args.no_save and (args.output_file.exists() or not args.output_file.parent.is_dir()
                            or args.output_file.suffix != ".parquet"):
        parser.error("Output must be a new .parquet file in an existing directory.")
    baseline = sibling("network_timesfm", "3.3.forecast_timesfm.py")
    adapted = baseline
    common = baseline.chronos_module()
    helper = common.input_module()
    examples = helper.load_examples(args.input_file)
    metadata = json.loads(examples["metadata_json"].item())
    rows, counts = sample_indices(examples["source_index"], metadata["sources"],
                                 args.samples_per_station, args.seed,
                                 [s.upper() for s in args.stations] if args.stations else None)
    sample = {key: examples[key][rows].copy() for key in helper.ARRAY_KEYS}
    del examples
    gc.collect()
    manifest = adapted.read_manifest(args.checkpoint_dir, "full")
    if manifest["history_length"] != HISTORY_LENGTH:
        raise ValueError("This experiment requires a full-pass L60 checkpoint.")
    reserved = {(r["station"], int(r["year"])) for r in manifest["test_station_years"]}
    if any((c["station"], c["year"]) not in reserved for c in counts):
        raise ValueError("Selected station-year was not excluded from adaptation.")
    if set(sample["station"]) & set(metadata.get("excluded_case_study_stations", [])):
        raise ValueError("Network cohort unexpectedly includes five-station case-study data.")
    clear = np.empty((len(rows), FORECAST_LENGTH), dtype=np.float64)
    for index in np.unique(sample["source_index"]):
        source = metadata["sources"][int(index)]
        source_path = common.resolve_source_path(source)
        if helper.source_hash(source_path) != source["sha256"]:
            raise ValueError(f"Source changed: {source['path']}")
        values = pq.read_table(source_path, columns=["ghi_clear_centi"])["ghi_clear_centi"].to_numpy()
        used = sample["source_index"] == index
        values = values[sample["origin_row"][used, None] + np.arange(1, 16)]
        if np.any(values == common.INVALID_CLEAR_SKY):
            raise ValueError("Invalid clear-sky reference.")
        clear[used] = values.astype(np.float64)/common.CLEAR_SKY_SCALE
    print(f"Selected {len(rows)} origins from {len(counts)} stations; L60, next15 values.", flush=True)
    print("Stations below requested count:", [c for c in counts if c["sampled"] < args.samples_per_station], flush=True)
    import torch
    from timesfm import ForecastConfig

    torch.set_num_threads(args.threads)
    observed = np.round(sample["target"].astype(np.float64)*clear, common.GHI_DECIMALS)
    tables, summary = [], {}
    for method in METHODS:
        started = perf_counter()
        if method == "Persistence":
            prediction = np.repeat(sample["history"][:, -1:], 15, axis=1)
        else:
            pipeline = (baseline.load_pretrained_model(args.device) if method == "TimesFM"
                        else adapted.load_finetuned_model(args.checkpoint_dir, manifest, args.device))
            pipeline.compile(ForecastConfig(max_context=60, max_horizon=15, per_core_batch_size=args.batch_size))
            prediction, _ = common.forecast(baseline.MedianAdapter(pipeline), sample["history"],
                                            60, args.batch_size, progress=False)
            del pipeline
            gc.collect()
            if args.device == "mps":
                torch.mps.empty_cache()
        forecast = np.round(prediction.astype(np.float64)*clear, common.GHI_DECIMALS)
        if not np.isfinite(forecast).all():
            raise ValueError(f"Nonfinite forecasts: {method}")
        data = dict(Station=sample["station"], Year=sample["year"], Time=sample["origin_utc_minute"],
                    OriginRow=sample["origin_row"], InputRow=rows, Method=[method]*len(rows))
        data.update({f"Obs{h+1}": observed[:, h] for h in range(15)})
        data.update({f"Fcst{h+1}": forecast[:, h] for h in range(15)})
        tables.append(pa.table(data))
        summary[method] = dict(rmse=float(np.sqrt(np.mean((forecast-observed)**2))),
                               mbe=float(np.mean(forecast-observed)), elapsed_seconds=perf_counter()-started)
        print(method, json.dumps(summary[method]), flush=True)
    for method in METHODS:
        summary[method]["rmse_skill"] = 1-summary[method]["rmse"]/summary["Persistence"]["rmse"]
    print("POOLED_GHI", json.dumps(summary), flush=True)
    if args.no_save:
        print("No-save run complete; no outputs written.", flush=True)
        return
    provenance = dict(input_file=str(args.input_file.resolve()), input_dataset_id=metadata["dataset_id"],
                      context_minutes=60, forecast_minutes=15, seed=args.seed,
                      requested_per_station=args.samples_per_station, sampling="uniform_without_replacement_per_station",
                      station_counts=counts, methods=list(METHODS), ghi_decimals=common.GHI_DECIMALS,
                      time_unit="UTC minutes since 1970-01-01", checkpoint=str(args.checkpoint_dir.resolve()),
                      checkpoint_manifest_sha256=hashlib.sha256((args.checkpoint_dir/"training_manifest.json").read_bytes()).hexdigest(),
                      base_model=manifest["base_model"], base_revision=manifest["base_revision"], metrics=summary)
    table = pa.concat_tables(tables).replace_schema_metadata({b"experiment": json.dumps(provenance).encode()})
    with args.output_file.open("xb") as stream:
        pq.write_table(table, stream, compression="zstd")
    saved = pq.read_table(args.output_file)
    assert saved.equals(table, check_metadata=True)
    print(f"VERIFIED {args.output_file}: {len(rows)} origins, {saved.num_rows} method rows, "
          f"{args.output_file.stat().st_size/1024**2:.1f} MiB.", flush=True)


if __name__ == "__main__":
    main()
