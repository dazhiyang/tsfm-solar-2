"""Forecast with 2.3's station-balanced L60 adapter on EXACT Discussion 3 origins.

  python -B Code/3.5.forecast_timesfm_station_balanced.py --self-test
  python -B Code/3.5.forecast_timesfm_station_balanced.py \
    --checkpoint-dir PATH --max-origins 3 --no-save
  python -B Code/3.5.forecast_timesfm_station_balanced.py --checkpoint-dir PATH

The existing network_1000_timesfm_L60.parquet defines the evaluation cohort.
InputRow identifiers select the original contexts from network_holdout.npz;
origins are NOT resampled. Station, year, timestamp, source row and observed GHI
must agree with the saved Discussion 3 results. Uses the pinned adapter loader,
native median predictions and horizon-specific clear-sky GHI reconstruction.

Save only the new method, TimesFM-Balanced, in one distinct Parquet file in the
existing Data/result directory, with the same columns as 3.4. Existing comparison
results are read-only. No directories, training or model downloads are created.
--max-origins is for no-save smoke tests ONLY; saved results use the full cohort.
The balanced checkpoint must first be trained separately using 2.3.
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
METHOD = "TimesFM-Balanced"
REFERENCE_METHODS = ("Persistence", "TimesFM", "TimesFM-BSRN")
ID_COLUMNS = ("Station", "Year", "Time", "OriginRow", "InputRow")
OBS_COLUMNS = tuple(f"Obs{h}" for h in range(1, 16))


def sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_cohort(table):
    """Verify that every saved comparator uses exactly the same ordered pairs."""
    methods = table["Method"].to_numpy()
    cohorts = {m: table.filter(pa.array(methods == m)).sort_by([("InputRow", "ascending")])
               for m in REFERENCE_METHODS}
    reference = cohorts["Persistence"]
    if not reference.num_rows or len(np.unique(reference["InputRow"])) != reference.num_rows:
        raise ValueError("Reference cohort is empty or contains duplicate origins")
    paired_columns = list(ID_COLUMNS + OBS_COLUMNS)
    for name, part in cohorts.items():
        if not part.select(paired_columns).equals(reference.select(paired_columns)):
            raise ValueError(f"Saved comparator is not paired with persistence: {name}")
    return reference, cohorts


def balanced_manifest(path, loader):
    manifest = loader.read_manifest(path, "balanced")
    if (manifest.get("train_mode") not in {
            "one_year_per_station_balanced", "all_years_station_balanced",
            "one_year_per_station_one_pass"} or
            manifest.get("history_length") != 60 or
            manifest.get("checkpoint_format") != "peft_adapter"):
        raise ValueError("Expected a completed L60 station-balanced adapter from 2.3")
    return manifest


def self_test():
    data = dict(Station=["AAA", "BBB"], Year=[2020, 2021], Time=[1, 2],
                OriginRow=[59, 74], InputRow=[4, 9])
    data.update({name: [100.0, 200.0] for name in OBS_COLUMNS})
    table = pa.concat_tables([pa.table(dict(data, Method=[m, m])) for m in REFERENCE_METHODS])
    cohort, _ = reference_cohort(table)
    assert cohort["InputRow"].to_pylist() == [4, 9]
    bad = table.set_column(table.schema.get_field_index("Obs1"), "Obs1",
                           pa.array([100., 200., 100., 201., 100., 200.]))
    try:
        reference_cohort(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched observations were not rejected")
    print("PASS: exact cohort reuse, comparator pairing and mismatched-observation rejection")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", type=Path, default=ROOT/"Data/forecast_input/network_holdout.npz")
    parser.add_argument("--reference-file", type=Path, default=ROOT/"Data/result/network_1000_timesfm_L60.parquet")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT/"Data/models/timesfm-2.5-balanced-L60")
    parser.add_argument("--output-file", type=Path, default=ROOT/"Data/result/network_1000_timesfm_balanced_L60.parquet")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.batch_size, args.threads) < 1 or args.max_origins < 0:
        parser.error("Positive batch/thread counts and nonnegative origin cap required")
    if args.max_origins and not args.no_save:
        parser.error("A partial cohort is allowed only with --no-save")
    if not args.no_save and (args.output_file.exists() or not args.output_file.parent.is_dir()
                            or args.output_file.suffix != ".parquet"):
        parser.error("Output must be a new .parquet file in an existing directory")

    baseline = sibling("balanced_forecast_base", "3.3.forecast_timesfm.py")
    adapted = baseline
    common = baseline.chronos_module()
    helper = common.input_module()
    manifest = balanced_manifest(args.checkpoint_dir, adapted)
    saved_reference = pq.read_table(args.reference_file)
    provenance = json.loads(saved_reference.schema.metadata[b"experiment"])
    cohort, comparators = reference_cohort(saved_reference)
    if args.max_origins:
        cohort = cohort.slice(0, args.max_origins)
        comparators = {m: t.slice(0, args.max_origins) for m, t in comparators.items()}
    rows = cohort["InputRow"].to_numpy().astype(np.int64)
    examples = helper.load_examples(args.input_file)
    metadata = json.loads(examples["metadata_json"].item())
    if metadata["dataset_id"] != provenance["input_dataset_id"]:
        raise ValueError("Input differs from the original Discussion 3 dataset")
    if rows.min() < 0 or rows.max() >= len(examples["history"]):
        raise ValueError("Reference InputRow is outside the saved input")
    sample = {key: examples[key][rows].copy() for key in helper.ARRAY_KEYS}
    del examples
    gc.collect()
    for column, key in (("Station", "station"), ("Year", "year"),
                        ("Time", "origin_utc_minute"), ("OriginRow", "origin_row")):
        np.testing.assert_array_equal(cohort[column].to_numpy(), sample[key])
    reserved = {(r["station"], int(r["year"])) for r in manifest["test_station_years"]}
    if not set(zip(sample["station"], sample["year"])).issubset(reserved):
        raise ValueError("Evaluation includes station-years not reserved from training")
    clear = np.empty((len(rows), 15), dtype=np.float64)
    for index in np.unique(sample["source_index"]):
        source = metadata["sources"][int(index)]
        source_path = common.resolve_source_path(source)
        if helper.source_hash(source_path) != source["sha256"]:
            raise ValueError(f"Source changed: {source['path']}")
        codes = pq.read_table(source_path, columns=["ghi_clear_centi"])["ghi_clear_centi"].to_numpy()
        used = sample["source_index"] == index
        values = codes[sample["origin_row"][used, None] + np.arange(1, 16)]
        if np.any(values == common.INVALID_CLEAR_SKY):
            raise ValueError("Invalid clear-sky reference")
        clear[used] = values.astype(np.float64)/common.CLEAR_SKY_SCALE
    observed = np.round(sample["target"].astype(np.float64)*clear, common.GHI_DECIMALS)
    np.testing.assert_array_equal(observed, np.column_stack([cohort[c].to_numpy() for c in OBS_COLUMNS]))

    import torch
    from timesfm import ForecastConfig

    torch.set_num_threads(args.threads)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    print(f"Reusing {len(rows):,} Discussion 3 origins; forecasting only {METHOD}.", flush=True)
    started = perf_counter()
    pipeline = adapted.load_finetuned_model(args.checkpoint_dir, manifest, args.device)
    pipeline.compile(ForecastConfig(max_context=60, max_horizon=15, per_core_batch_size=args.batch_size))
    prediction, _ = common.forecast(baseline.MedianAdapter(pipeline), sample["history"],
                                    60, args.batch_size, progress=False)
    forecast = np.round(prediction.astype(np.float64)*clear, common.GHI_DECIMALS)
    if forecast.shape != observed.shape or not np.isfinite(forecast).all():
        raise ValueError("Invalid forecast shape or nonfinite forecast")
    summary = {}
    for method, part in comparators.items():
        values = np.column_stack([part[f"Fcst{h}"].to_numpy() for h in range(1, 16)])
        summary[method] = dict(rmse=float(np.sqrt(np.mean((values-observed)**2))))
    summary[METHOD] = dict(rmse=float(np.sqrt(np.mean((forecast-observed)**2))),
                           mbe=float(np.mean(forecast-observed)), elapsed_seconds=perf_counter()-started)
    for values in summary.values():
        values["rmse_skill"] = 1-values["rmse"]/summary["Persistence"]["rmse"]
    print("OVERALL_GHI", json.dumps(summary), flush=True)
    if args.no_save:
        print("No-save forecast complete; existing results untouched.")
        return
    data = {column: cohort[column] for column in ID_COLUMNS + OBS_COLUMNS}
    data["Method"] = [METHOD]*len(rows)
    data.update({f"Fcst{h+1}": forecast[:, h] for h in range(15)})
    output_metadata = dict(input_dataset_id=metadata["dataset_id"], reference_file=str(args.reference_file.resolve()),
                           reference_sha256=helper.source_hash(args.reference_file),
                           checkpoint=str(args.checkpoint_dir.resolve()),
                           checkpoint_manifest_sha256=hashlib.sha256((args.checkpoint_dir/"training_manifest.json").read_bytes()).hexdigest(),
                           base_model=manifest["base_model"], base_revision=manifest["base_revision"],
                           context_minutes=60, forecast_minutes=15, methods=[METHOD],
                           sampling="exact InputRow cohort from Discussion 3; no resampling",
                           time_unit="UTC minutes since 1970-01-01", metrics=summary)
    table = pa.table(data).select(saved_reference.column_names)
    table = table.replace_schema_metadata({b"experiment": json.dumps(output_metadata).encode()})
    with args.output_file.open("xb") as stream:
        pq.write_table(table, stream, compression="zstd")
    if not pq.read_table(args.output_file).equals(table, check_metadata=True):
        raise RuntimeError("Saved forecast failed exact read-back")
    print(f"Saved and verified {len(rows):,} forecasts: {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
