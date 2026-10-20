"""TimesFM L60 station-balanced fine-tuning, using one year per station.

Select the year closest to 6000 training origins (latest year breaks ties).
Reuse the full-pass day-blocked split and fixed validation origins. By default,
inspect counts only. Training requires --train and a new --output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = ROOT / "Data/models/timesfm-2.5-full-L60/training_manifest.json"
SEED = 42
N_VALIDATION = 2048
HISTORY_LENGTH = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
EVALUATION_STEPS = 1000
TARGET_ORIGINS = 6000


def timesfm_module():
    spec = importlib.util.spec_from_file_location(
        "timesfm_balanced_base", Path(__file__).with_name("2.2.finetune_timesfm.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_years(train, target=TARGET_ORIGINS):
    counts = train.groupby(["station", "year"]).size().reset_index(name="count")
    counts["distance"] = (counts["count"] - target).abs()
    chosen = counts.sort_values(
        ["station", "distance", "year"], ascending=[True, True, False]
    ).drop_duplicates("station")
    return (train.merge(chosen[["station", "year"]], on=["station", "year"],
                        how="inner", validate="many_to_one")
            .sort_values(["station", "year", "origin_minute_of_year"])
            .reset_index(drop=True))


def origin_fingerprint(frame):
    records = frame[["station", "year", "origin_minute_of_year", "origin_index"]]
    return hashlib.sha256(records.to_csv(index=False).encode()).hexdigest()


def self_test():
    import pandas as pd
    frame = pd.DataFrame(dict(station=["AAA", "AAA", "BBB", "BBB", "BBB", "BBB"],
                              year=[2020, 2021, 2021, 2022, 2022, 2022],
                              origin_minute_of_year=[0, 15, 30, 45, 60, 75],
                              origin_index=[59, 74, 89, 104, 119, 134]))
    train = select_years(frame, target=2)
    assert train.groupby("station").year.first().to_dict() == {"AAA": 2021, "BBB": 2022}
    timesfm = timesfm_module()
    parts = list(timesfm.seeded_batches(len(train), 3, SEED))
    batches = np.concatenate(parts)
    repeat = np.concatenate(list(timesfm.seeded_batches(len(train), 3, SEED)))
    assert np.array_equal(batches, repeat)
    assert np.array_equal(np.sort(batches), np.arange(len(train)))
    assert [len(p) for p in parts] == [3, 1]
    assert select_years(frame, target=1).groupby("station").year.first().to_dict() == {
        "AAA": 2021, "BBB": 2021}
    assert origin_fingerprint(train) != origin_fingerprint(train.iloc[::-1])
    print("PASS: year selection, tie break, reproducible pass, partial batch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.train and (args.output_dir is None or args.output_dir.exists()):
        parser.error("Training requires an explicit new --output-dir")
    timesfm = timesfm_module()
    base = timesfm.chronos_finetune_module()
    reference = json.loads(REFERENCE_MANIFEST.read_text())
    for key, value in dict(completed=True, history_length=60, random_seed=SEED,
                           validation_seed=SEED, n_val_series=N_VALIDATION,
                           validation_rule="station_year_utc_day_fraction",
                           validation_fraction=0.3, origin_cadence_minutes=15,
                           cadence_offset_minutes=0).items():
        if reference.get(key) != value:
            raise ValueError(f"Reference manifest differs for {key}")
    index_file = base.DEFAULT_INDEX
    if Path(reference["index_file"]).resolve() != index_file.resolve():
        raise ValueError("Reference used a different index")
    frame, metadata = base.load_cadence_index(index_file)
    train, val = base.blocked_split(frame, SEED)
    del frame
    if (len(train) != reference["n_train_index_rows"] or
            len(val) != reference["n_validation_pool_rows"] or
            metadata["test_station_years"] != reference["test_station_years"]):
        raise ValueError("Index no longer reproduces the reference split/exclusions")
    n_validation_pool = len(val)
    val = base.sample_index_rows(val, N_VALIDATION, SEED)
    train = select_years(train)
    selected = train.groupby("station", sort=True).agg(year=("year", "first"), count=("year", "size"))
    num_steps = math.ceil(len(train) / BATCH_SIZE)
    if set(selected.index) & set(base.index_module().KOPPEN_HOLDOUT_STATIONS):
        raise ValueError("A held-out station is present in training")
    print(f"L60; {len(train):,} origins from {len(selected)} stations; {num_steps} updates")
    print(f"Validation: {len(val)} origins; fingerprint {origin_fingerprint(val)}")
    if not args.train:
        print("Inspection only: no training started and no files saved.")
        return
    cache = base.KappaCache(base.index_module())
    val_h, val_t = timesfm.index_to_arrays(base, val, cache, HISTORY_LENGTH, SEED, None)
    pipeline = timesfm.load_pretrained_pipeline(args.device)
    timesfm.finetune(
        pipeline, histories=None, targets=None, val_histories=val_h, val_targets=val_t,
        history_length=HISTORY_LENGTH, output_dir=args.output_dir,
        batch_size=BATCH_SIZE, num_steps=num_steps, learning_rate=LEARNING_RATE,
        seed=SEED, train_heads_only=False, smoke_test=False, index_file=index_file.resolve(),
        index_metadata=metadata, n_train_index_rows=len(train), train_frame=train,
        cache=cache, base=base, evaluation_steps=EVALUATION_STEPS,
        batch_indices=timesfm.seeded_batches(len(train), BATCH_SIZE, SEED),
        save_adapter_only=True,
        manifest_extra=dict(
            script=Path(__file__).name, train_mode="one_year_per_station_one_pass",
            sampling="one_shuffled_pass_without_replacement",
            year_selection="closest_training_origin_count_latest_year_tiebreak",
            target_origins_per_station=TARGET_ORIGINS,
            selected_station_years=[dict(station=s, year=int(y), n_origins=int(n))
                                    for (s, y), n in train.groupby(["station", "year"]).size().items()],
            n_training_stations=len(selected), climate_balancing=False,
            origin_cadence_minutes=15, cadence_offset_minutes=0,
            validation_rule="station_year_utc_day_fraction", validation_fraction=0.3,
            validation_seed=SEED, n_validation_pool_rows=n_validation_pool,
            validation_origin_fingerprint=origin_fingerprint(val),
            reference_manifest=str(REFERENCE_MANIFEST), training_logging_steps=100,
            starting_checkpoint="pinned_pretrained_base_not_BSRN_adapter",
        ),
    )


if __name__ == "__main__":
    main()
