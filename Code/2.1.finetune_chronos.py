"""Fine-tune one shared Chronos-2 model on the pooled index from script 1.1.

Index and outputs (under Data/):
  Data/finetuning_data.parquet              — train/validation row index (script 1.1)
  Data/forecast_input/*.npz                 — held-out evaluation (script 1.2)

Model paths:
  Data/models/chronos-2-base/              — pinned base weights (optional local copy)
  Data/models/chronos-2-finetuned-L{15,30,60}/  — per history length (--output-dir)
  .../finetuned-ckpt/                         — final LoRA adapter

Training is not started by default. Run without --train to inspect index counts.

  python -B Code/2.1.finetune_chronos.py
  python -B Code/2.1.finetune_chronos.py --download-base   # once, needs network
  python -B Code/2.1.finetune_chronos.py --history-length 15 --num-steps 1000 --train --output-dir PATH

Both limited and full-pass runs use the 15-minute origin grid, station-year UTC
day-blocked validation split, and the same seeded permutation without replacement.
Omit --num-steps for one full pass. Validation uses 2048 fixed origins by default.

LoRA requires the peft package in tf_timesfm.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
from transformers import TrainerCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "Data" / "finetuning_data.parquet"
BASE_MODEL_DIR = PROJECT_ROOT / "Data" / "models" / "chronos-2-base"
FINETUNED_OUTPUT_DIR = PROJECT_ROOT / "Data" / "models" / "chronos-2-finetuned-L30"
FINETUNED_CKPT_NAME = "finetuned-ckpt"
FINETUNED_CHECKPOINT_DIR = FINETUNED_OUTPUT_DIR / FINETUNED_CKPT_NAME
TRAINING_MANIFEST = FINETUNED_OUTPUT_DIR / "training_manifest.json"
MODEL_NAME = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
HISTORY_LENGTHS = (15, 30, 60)
FORECAST_LENGTH = 15
MAX_HISTORY_LENGTH = 60
KAPPA_SCALE = 1000
FINETUNE_MODE = "lora"
DEVICE = "mps"
RANDOM_SEED = 42
STREAM_TRAIN_THRESHOLD = 250_000
MAX_MATERIALIZE_TRAIN = 2_000_000
CADENCE = 15
VALIDATION_FRACTION = 0.30
# Measured 5,000-step L15 pilot, including validation/checkpoint overhead.
PILOT_SECONDS_PER_STEP = 812.6935 / 5000


def index_module():
    path = Path(__file__).with_name("1.1.prepare_finetuning_input.py")
    spec = importlib.util.spec_from_file_location("finetune_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_index(index_file):
    """Load the parquet index and JSON metadata written by script 1.1."""
    index_file = Path(index_file)
    table = pq.read_table(index_file)
    frame = table.to_pandas()
    metadata = {}
    if table.schema.metadata:
        for key, value in table.schema.metadata.items():
            try:
                metadata[key.decode()] = json.loads(value.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                metadata[key.decode()] = value.decode()
    required = {"station", "year", "file_path", "origin_index", "origin_minute_of_year", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Index missing columns: {required - set(frame.columns)}")
    return frame, metadata


def split_frame(frame, split):
    part = frame.loc[frame["split"] == split].reset_index(drop=True)
    if part.empty:
        raise ValueError(f"No rows with split={split!r}.")
    return part


def load_cadence_index(path):
    """Load eligible origins on the common 15-minute grid in bounded batches."""
    parquet = pq.ParquetFile(path)
    metadata = {
        key.decode(): json.loads(value.decode())
        for key, value in (parquet.schema_arrow.metadata or {}).items()
        if key != b"pandas"
    }
    for key, expected in (
        ("common_window_length", 75), ("max_history_length", 60),
        ("forecast_length", 15), ("kappa_scale", 1000),
        ("train_subsample_fraction", 1), ("max_train_origins", None),
    ):
        if key not in metadata or metadata[key] != expected:
            raise ValueError(f"Index metadata {key!r} must be {expected!r}.")
    if not metadata.get("test_station_years"):
        raise ValueError("Index must record its held-out station-years.")
    columns = ["station", "year", "file_path", "origin_index", "origin_minute_of_year"]
    parts = []
    for batch in parquet.iter_batches(batch_size=500_000, columns=columns):
        minutes = batch.column(columns.index("origin_minute_of_year")).to_numpy()
        keep = minutes % CADENCE == 0
        if keep.any():
            parts.append(batch.filter(keep).to_pandas())
    if not parts:
        raise ValueError("No eligible origins on the 15-minute grid.")
    frame = pd.concat(parts, ignore_index=True)
    frame["station"] = frame["station"].str.upper()
    test_keys = {(item["station"].upper(), int(item["year"]))
                 for item in metadata["test_station_years"]}
    keys = set(frame[["station", "year"]].itertuples(index=False, name=None))
    if keys & test_keys or set(frame.station) & set(metadata.get("test_stations", [])):
        raise ValueError("Held-out verification data found in the fine-tuning index.")
    if frame.duplicated(["file_path", "origin_index"]).any():
        raise ValueError("Duplicate training origins in index.")
    return frame, metadata


def blocked_split(frame, seed):
    """Deterministically reserve about 30% of each station-year's UTC days."""
    frame = frame.copy()
    frame["day"] = frame.origin_minute_of_year.to_numpy() // 1440
    validation = np.zeros(len(frame), dtype=bool)
    for (station, year), indices in frame.groupby(["station", "year"], sort=True).indices.items():
        days = frame.iloc[indices].day.to_numpy()
        unique_days = np.unique(days)
        n_val = min(len(unique_days) - 1, round(len(unique_days) * VALIDATION_FRACTION))
        digest = hashlib.sha256(f"{seed}:{station}:{year}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        selected = rng.choice(unique_days, size=n_val, replace=False)
        validation[indices] = np.isin(days, selected)
    train = frame.loc[~validation].reset_index(drop=True)
    val = frame.loc[validation].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("Blocked split requires nonempty training and validation pools.")
    return train, val


class KappaCache:
    """Lazy per-file kappa_milli arrays for indexed origins."""

    def __init__(self, validator):
        self._validator = validator
        self._arrays = {}

    def kappa(self, file_path):
        path = Path(file_path)
        if path not in self._arrays:
            self._validator.assert_station_year_schema(path)
            table = pq.read_table(path, columns=["kappa_milli"])
            self._arrays[path] = table["kappa_milli"].to_numpy().astype(np.uint16, copy=False)
        return self._arrays[path]


def example_window(codes, origin_index, history_length, forecast_length=FORECAST_LENGTH):
    """Return (history, target) clear-sky index vectors for one indexed origin."""
    if history_length not in HISTORY_LENGTHS:
        raise ValueError(f"history_length must be one of {HISTORY_LENGTHS}.")
    origin = int(origin_index)
    start = origin - MAX_HISTORY_LENGTH + 1
    stop = origin + forecast_length
    if start < 0 or stop >= len(codes):
        raise ValueError("Origin window falls outside the station-year file.")
    window = codes[start : stop + 1].astype(np.float32) / KAPPA_SCALE
    history = window[:MAX_HISTORY_LENGTH][-history_length:]
    target = window[MAX_HISTORY_LENGTH:]
    if len(history) != history_length or len(target) != forecast_length:
        raise ValueError("Unexpected history/target length.")
    if not np.isfinite(history).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite kappa in indexed window.")
    return history, target


def chronos_series(history, target):
    """One univariate series for Chronos fit: context + held-out future in `target`."""
    return {"target": np.concatenate([history, target]).astype(np.float32, copy=False)}


def sample_index_rows(frame, max_series, seed):
    if max_series is None or max_series >= len(frame):
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(frame), size=int(max_series), replace=False)
    return frame.iloc[positions].reset_index(drop=True)


def index_to_chronos_inputs(frame, cache, history_length, seed, max_series=None):
    rows = sample_index_rows(frame, max_series, seed)
    inputs = []
    for _, row in rows.iterrows():
        codes = cache.kappa(row["file_path"])
        history, target = example_window(codes, row["origin_index"], history_length)
        inputs.append(chronos_series(history, target))
    if not inputs:
        raise ValueError("No training series materialized from index.")
    return inputs


def should_stream_train(n_train_rows, stream_train, no_stream_train):
    if no_stream_train:
        return False
    if stream_train:
        return True
    return n_train_rows > STREAM_TRAIN_THRESHOLD


def resolve_train_mode(n_train_rows, max_train_series, stream_train, no_stream_train):
    """Choose in-memory materialization vs index streaming for training."""
    if should_stream_train(n_train_rows, stream_train, no_stream_train):
        if max_train_series is not None:
            print(
                "--max-train-series ignored in streaming mode (each step samples the full train index).",
                flush=True,
            )
        return "stream"
    cap = max_train_series if max_train_series is not None else n_train_rows
    if cap > MAX_MATERIALIZE_TRAIN:
        raise ValueError(
            f"Refusing to materialize {cap:,} training series in memory "
            f"(limit {MAX_MATERIALIZE_TRAIN:,}). Use --stream-train or lower --max-train-series."
        )
    return "materialize"


class KappaIndexedBatchDataset(IterableDataset):
    """Chronos2Trainer-compatible iterable dataset; samples index rows per batch."""

    def __init__(
        self,
        frame,
        cache,
        history_length,
        batch_size,
        seed,
        num_output_patches,
        mode,
    ):
        from chronos.chronos2.dataset import DatasetMode

        super().__init__()
        if mode not in (DatasetMode.TRAIN, DatasetMode.VALIDATION):
            raise ValueError("mode must be TRAIN or VALIDATION.")
        if len(frame) == 0:
            raise ValueError("Indexed frame is empty.")
        self._frame = frame.reset_index(drop=True)
        self._cache = cache
        self._history_length = history_length
        self.batch_size = int(batch_size)
        self._num_output_patches = int(num_output_patches)
        self._mode = mode
        self._rng = np.random.default_rng(seed)
        self._file_paths = self._frame["file_path"].to_numpy()
        self._origin_indices = self._frame["origin_index"].to_numpy(dtype=np.int64)

    def _row_tensors(self, row_idx):
        import torch

        codes = self._cache.kappa(self._file_paths[row_idx])
        history, target = example_window(codes, self._origin_indices[row_idx], self._history_length)
        series = torch.from_numpy(np.concatenate([history, target])).view(1, -1)
        context = series[:, : self._history_length]
        future = series[:, self._history_length : self._history_length + FORECAST_LENGTH]
        return context, future

    def _batch_dict(self, row_indices):
        import torch

        contexts, futures = [], []
        for row_idx in row_indices:
            context, future = self._row_tensors(int(row_idx))
            contexts.append(context)
            futures.append(future)
        batch_size = len(contexts)
        return {
            "context": torch.cat(contexts, dim=0),
            "future_target": torch.cat(futures, dim=0),
            "future_covariates": torch.full((batch_size, FORECAST_LENGTH), float("nan")),
            "group_ids": torch.arange(batch_size),
            "num_output_patches": self._num_output_patches,
        }

    def __iter__(self):
        from chronos.chronos2.dataset import DatasetMode

        n_rows = len(self._frame)
        if self._mode == DatasetMode.TRAIN:
            while True:
                row_indices = self._rng.integers(0, n_rows, size=self.batch_size)
                batch = self._batch_dict(row_indices)
                yield batch
        else:
            for start in range(0, n_rows, self.batch_size):
                stop = min(start + self.batch_size, n_rows)
                batch = self._batch_dict(np.arange(start, stop))
                yield batch


class SeededNoReplacementDataset(KappaIndexedBatchDataset):
    """Finite seeded permutation shared by limited and full-pass runs."""

    def __init__(self, *args, **kwargs):
        self._shuffle_seed = kwargs["seed"]
        super().__init__(*args, **kwargs)

    def __len__(self):
        return math.ceil(len(self._frame) / self.batch_size)

    def __iter__(self):
        order = np.random.default_rng(self._shuffle_seed).permutation(len(self._frame))
        for start in range(0, len(order), self.batch_size):
            yield self._batch_dict(order[start:start + self.batch_size])


class LossHistory(TrainerCallback):
    """Persist each loss event immediately so interrupted runs retain a curve."""

    def __init__(self, path, history_length):
        self.path = Path(path)
        self.history_length = int(history_length)

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()
        with self.path.open("x", encoding="utf-8"):
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs and "eval_loss" not in logs:
            return
        event = dict(
            history_length=self.history_length,
            step=int(state.global_step),
            elapsed_seconds=time.perf_counter() - self.started,
            event="validation" if "eval_loss" in logs else "training",
            **logs,
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")


def synthetic_chronos_inputs(n_series, history_length, seed):
    rng = np.random.default_rng(seed)
    inputs = []
    for _ in range(n_series):
        history = rng.uniform(0.05, 1.0, size=history_length).astype(np.float32)
        target = np.clip(history[-1] + rng.normal(0, 0.02, size=FORECAST_LENGTH), 0.0, 1.2).astype(
            np.float32
        )
        inputs.append(chronos_series(history, target))
    return inputs


def training_batches(frame, cache, history_length, batch_size, seed, limit=None):
    """Yield (histories, targets) float32 batches from an index subset."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    rows = frame if limit is None else frame.head(limit)
    order = np.arange(len(rows))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    histories, targets = [], []
    for position in order:
        row = rows.iloc[int(position)]
        codes = cache.kappa(row["file_path"])
        history, target = example_window(codes, row["origin_index"], history_length)
        histories.append(history)
        targets.append(target)
        if len(histories) == batch_size:
            yield np.stack(histories), np.stack(targets)
            histories, targets = [], []
    if histories:
        yield np.stack(histories), np.stack(targets)


def base_model_dir_ready(base_dir):
    """True if a local Chronos-2 snapshot looks complete."""
    base_dir = Path(base_dir)
    config = base_dir / "config.json"
    if not config.is_file():
        return False
    try:
        payload = json.loads(config.read_text())
    except json.JSONDecodeError:
        return False
    return "chronos_config" in payload


def resolve_pretrained_source(base_dir=BASE_MODEL_DIR):
    """Prefer Data/models/chronos-2-base when present; else Hub id + revision."""
    if base_model_dir_ready(base_dir):
        return str(base_dir.resolve()), True
    return MODEL_NAME, False


def download_base_weights(base_dir=BASE_MODEL_DIR):
    """Copy pinned chronos-2 weights into Data/models/chronos-2-base (creates dirs)."""
    from huggingface_hub import snapshot_download

    base_dir = Path(base_dir)
    if base_model_dir_ready(base_dir):
        print(f"Base model already present: {base_dir.resolve()}", flush=True)
        return base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
        local_dir=str(base_dir),
        local_dir_use_symlinks=False,
    )
    if not base_model_dir_ready(base_dir):
        raise RuntimeError(f"Download finished but {base_dir} is not a valid Chronos-2 folder.")
    print(f"Base model saved: {base_dir.resolve()}", flush=True)
    return base_dir


def load_base_pipeline(device=DEVICE, base_dir=BASE_MODEL_DIR):
    """Load base Chronos-2 from Data/models when available, else Hub cache."""
    import torch
    from chronos import Chronos2Pipeline

    source, local_only = resolve_pretrained_source(base_dir)
    kwargs = {
        "revision": None if local_only else MODEL_REVISION,
        "device_map": device,
        "torch_dtype": torch.float32,
        "local_files_only": local_only,
    }
    pipeline = Chronos2Pipeline.from_pretrained(source, **kwargs)
    pipeline.model.train()
    return pipeline


def write_training_manifest(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def check_dependencies_and_device():
    """Report whether LoRA (peft) and the requested device are available."""
    import torch

    report = {"device": DEVICE, "mps_available": torch.backends.mps.is_available(), "peft": False}
    try:
        import peft  # noqa: F401

        report["peft"] = True
    except ImportError:
        pass
    if FINETUNE_MODE == "lora" and not report["peft"]:
        raise RuntimeError("LoRA requires peft; install only with explicit approval.")
    if DEVICE == "mps" and not report["mps_available"]:
        raise RuntimeError("MPS unavailable for training trial.")
    return report


def run_lora_trainer(
    pipeline,
    train_dataset,
    eval_dataset,
    *,
    history_length,
    output_dir,
    finetuned_ckpt_name,
    batch_size,
    num_steps,
    learning_rate,
    evaluation_steps=100,
    extra_callbacks=None,
):
    """LoRA fine-tune with pre-built Chronos2Trainer datasets (streaming or materialized)."""
    import torch
    import torch.cuda
    from peft import LoraConfig, get_peft_model
    from transformers.trainer_callback import PrinterCallback
    from transformers.training_args import TrainingArguments

    from chronos import Chronos2Model
    from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback

    config = deepcopy(pipeline.model.config)
    model = Chronos2Model(config).to(pipeline.model.device)
    model.load_state_dict(pipeline.model.state_dict())
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    )
    model = get_peft_model(model, lora_config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_type = torch.device(model.device).type
    use_cpu = device_type == "cpu"
    has_sm80 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    optim = "adamw_torch" if device_type in ("cpu", "mps") else "adamw_torch_fused"

    training_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        optim=optim,
        logging_strategy="steps",
        logging_steps=100,
        disable_tqdm=False,
        report_to="none",
        max_steps=num_steps,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        tf32=has_sm80 and not use_cpu,
        bf16=has_sm80 and not use_cpu,
        save_only_model=True,
        prediction_loss_only=True,
        save_total_limit=1,
        save_strategy="no",
        save_steps=None,
        eval_strategy="no",
        eval_steps=None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        use_cpu=use_cpu,
    )
    callbacks = []
    if eval_dataset is not None:
        training_kwargs["save_strategy"] = "steps"
        training_kwargs["save_steps"] = evaluation_steps
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = evaluation_steps
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["label_names"] = ["future_target"]
        callbacks.append(EvaluateAndSaveFinalStepCallback())
    callbacks.extend(extra_callbacks or [])

    training_args = TrainingArguments(**training_kwargs)
    if not use_cpu:
        training_args._n_gpu = 1

    trainer = Chronos2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    trainer.pop_callback(PrinterCallback)
    trainer.train()

    model.chronos_config.context_length = max(model.chronos_config.context_length, history_length)
    patch = pipeline.model_output_patch_size
    model.chronos_config.max_output_patches = max(
        model.chronos_config.max_output_patches,
        math.ceil(FORECAST_LENGTH / patch),
    )
    model.config.chronos_config = model.chronos_config.__dict__

    from chronos import Chronos2Pipeline

    finetuned_pipeline = Chronos2Pipeline(model=model)
    finetuned_path = output_dir / finetuned_ckpt_name
    finetuned_pipeline.save_pretrained(finetuned_path)
    return finetuned_pipeline, finetuned_path


def finetune(
    pipeline,
    *,
    history_length,
    index_file,
    index_metadata,
    train_frame=None,
    train_inputs=None,
    validation_inputs=None,
    cache=None,
    train_mode="materialize",
    n_train_index_rows=0,
    seed=RANDOM_SEED,
    output_dir=FINETUNED_OUTPUT_DIR,
    finetuned_ckpt_name=FINETUNED_CKPT_NAME,
    batch_size=64,
    num_steps=1000,
    learning_rate=1e-5,
    smoke_test=False,
):
    """Run LoRA training; train_mode 'stream' samples κ windows from the index each step."""
    if FINETUNE_MODE != "lora":
        raise ValueError(f"Only LoRA is supported; got {FINETUNE_MODE!r}.")
    check_dependencies_and_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    num_output_patches = math.ceil(FORECAST_LENGTH / pipeline.model_output_patch_size)
    if train_mode == "stream":
        if train_frame is None or cache is None:
            raise ValueError("Streaming training requires train_frame and cache.")
        train_dataset = KappaIndexedBatchDataset(
            train_frame,
            cache,
            history_length,
            batch_size,
            seed,
            num_output_patches,
            DatasetMode.TRAIN,
        )
        train_desc = f"{len(train_frame):,} index rows (streamed)"
    else:
        if not train_inputs:
            raise ValueError("Materialized training requires train_inputs.")
        train_dataset = Chronos2Dataset.convert_inputs(
            inputs=train_inputs,
            context_length=history_length,
            prediction_length=FORECAST_LENGTH,
            batch_size=batch_size,
            output_patch_size=pipeline.model_output_patch_size,
            min_past=history_length,
            mode=DatasetMode.TRAIN,
        )
        train_desc = f"{len(train_inputs):,} materialized series"

    eval_dataset = None
    if validation_inputs:
        eval_dataset = Chronos2Dataset.convert_inputs(
            inputs=validation_inputs,
            context_length=history_length,
            prediction_length=FORECAST_LENGTH,
            batch_size=batch_size,
            output_patch_size=pipeline.model_output_patch_size,
            mode=DatasetMode.VALIDATION,
        )

    print(
        f"LoRA fit: {train_desc}, "
        f"{len(validation_inputs) if validation_inputs else 0} validation, "
        f"steps={num_steps}, batch_size={batch_size}, mode={train_mode}",
        flush=True,
    )

    finetuned, ckpt_dir = run_lora_trainer(
        pipeline,
        train_dataset,
        eval_dataset,
        history_length=history_length,
        output_dir=output_dir,
        finetuned_ckpt_name=finetuned_ckpt_name,
        batch_size=batch_size,
        num_steps=num_steps,
        learning_rate=learning_rate,
    )
    adapter_config = ckpt_dir / "adapter_config.json"
    if not adapter_config.is_file():
        raise RuntimeError(f"Expected LoRA adapter at {adapter_config}; got no adapter_config.json.")
    manifest = {
        "finetune_mode": FINETUNE_MODE,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "history_length": history_length,
        "forecast_length": FORECAST_LENGTH,
        "index_file": str(Path(index_file).resolve()) if index_file else None,
        "dataset_id": index_metadata.get("dataset_id") if index_metadata else None,
        "n_train_index_rows": int(n_train_index_rows),
        "train_mode": train_mode,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "smoke_test": smoke_test,
        "checkpoint_dir": str(ckpt_dir.resolve()),
    }
    write_training_manifest(output_dir / "training_manifest.json", manifest)
    print(f"LoRA checkpoint: {ckpt_dir.resolve()}", flush=True)
    return finetuned


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--history-length", type=int, choices=HISTORY_LENGTHS, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-val-series", type=int, default=2048)
    parser.add_argument("--evaluation-steps", type=int, default=1000)
    parser.add_argument("--num-steps", type=int,
                        help="Seeded no-replacement update budget (default: one full pass).")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--base-model-dir", type=Path, default=BASE_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--download-base", action="store_true")
    args = parser.parse_args()
    counts = [args.batch_size, args.max_val_series, args.evaluation_steps]
    if args.num_steps is not None:
        counts.append(args.num_steps)
    if min(counts) < 1 or args.seed < 0 or args.learning_rate <= 0:
        parser.error("Counts and learning rate must be positive; seed must be nonnegative.")
    if args.train and args.output_dir is None:
        parser.error("Training requires an explicit new --output-dir.")
    if args.train and args.output_dir.exists():
        parser.error("Output path exists; choose a new path to preserve previous results.")
    return args


def main():
    args = parse_args()
    if args.download_base:
        download_base_weights(args.base_model_dir)
        return
    if args.preflight:
        print(check_dependencies_and_device())
        return
    print("Reading and thinning the existing index; no new index file...", flush=True)
    frame, metadata = load_cadence_index(args.index_file)
    train, val = blocked_split(frame, args.seed)
    del frame
    full_pass_steps = math.ceil(len(train) / args.batch_size)
    steps = min(args.num_steps, full_pass_steps) if args.num_steps is not None else full_pass_steps
    presented_origins = min(steps * args.batch_size, len(train))
    print(f"Training origins: {len(train):,}; validation pool: {len(val):,}")
    print(f"History: {args.history_length}; forecast: 15 one-minute values; cadence: 15 minutes")
    if steps == full_pass_steps:
        print(f"One pass: {steps:,} updates, batch {args.batch_size} (last batch retained)")
    else:
        print(f"Limited run: {steps:,}/{full_pass_steps:,} updates, "
              f"{presented_origins:,} distinct origin presentations")
    print(f"Pilot-scaled estimate: {steps * PILOT_SECONDS_PER_STEP / 3600:.2f} hours; not guaranteed")
    if not args.train:
        print("Inspection only: no training started and no files saved.")
        return

    import torch
    from chronos.chronos2.dataset import DatasetMode

    check_dependencies_and_device()
    torch.manual_seed(args.seed)
    pipeline = load_base_pipeline(device=DEVICE, base_dir=args.base_model_dir)
    cache = KappaCache(index_module())
    n_validation_pool_rows = len(val)
    val = sample_index_rows(val, args.max_val_series, args.seed)
    common = dict(cache=cache, history_length=args.history_length,
                  batch_size=args.batch_size, seed=args.seed,
                  num_output_patches=math.ceil(FORECAST_LENGTH / pipeline.model_output_patch_size))
    train_dataset = SeededNoReplacementDataset(train, mode=DatasetMode.TRAIN, **common)
    val_dataset = KappaIndexedBatchDataset(val, mode=DatasetMode.VALIDATION, **common)
    train_mode = ("one_pass_without_replacement" if steps == full_pass_steps
                  else "fixed_updates_without_replacement")
    manifest = dict(
        script=Path(__file__).name, index_file=str(args.index_file.resolve()),
        model_name=MODEL_NAME, model_revision=MODEL_REVISION, finetune_mode="lora",
        history_length=args.history_length, forecast_length=FORECAST_LENGTH,
        origin_cadence_minutes=CADENCE, cadence_offset_minutes=0,
        validation_rule="station_year_utc_day_fraction", validation_fraction=VALIDATION_FRACTION,
        validation_seed=args.seed, n_validation_pool_rows=n_validation_pool_rows,
        n_validation_series=len(val), test_station_years=metadata["test_station_years"],
        n_train_index_rows=len(train), n_presented_origins=presented_origins,
        train_mode=train_mode, sampling="seeded_permutation_without_replacement",
        num_steps=steps, batch_size=args.batch_size, learning_rate=args.learning_rate,
        evaluation_steps=args.evaluation_steps, random_seed=args.seed, completed=False,
        training_logging_steps=100, loss_history_file="loss_history.jsonl",
        loss_metric="Chronos objective on clear-sky index (not GHI RMSE)",
        checkpoint_selection="minimum_validation_loss",
        checkpoint_dir=str((args.output_dir / FINETUNED_CKPT_NAME).resolve()),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "training_manifest.json"
    write_training_manifest(manifest_path, manifest)
    started = time.perf_counter()
    run_lora_trainer(
        pipeline, train_dataset, val_dataset, history_length=args.history_length,
        output_dir=args.output_dir, finetuned_ckpt_name=FINETUNED_CKPT_NAME,
        batch_size=args.batch_size, num_steps=steps, learning_rate=args.learning_rate,
        evaluation_steps=args.evaluation_steps,
        extra_callbacks=[LossHistory(args.output_dir / "loss_history.jsonl", args.history_length)],
    )
    manifest.update(completed=True, elapsed_seconds=time.perf_counter() - started)
    write_training_manifest(manifest_path, manifest)
    print(f"Finished in {manifest['elapsed_seconds'] / 3600:.2f} hours.")


if __name__ == "__main__":
    main()
