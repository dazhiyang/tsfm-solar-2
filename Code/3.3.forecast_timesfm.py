"""TimesFM 2.5 forecasting with pretrained weights or a local adapted checkpoint.

  python -B Code/3.3.forecast_timesfm.py --stations QIQ --years 2025 --max-origins 32 --no-save
  python -B Code/3.3.forecast_timesfm.py --checkpoint-dir Data/models/timesfm-2.5-full-L60 --stations QIQ --years 2025
  python -B Code/3.3.forecast_timesfm.py --self-test

Without --checkpoint-dir use cached pinned pretrained weights. Adapted models
use their manifest context length. Output prefixes are timesfm (pretrained),
timesfm_ft (1000 updates), and timesfm_ft_full (full pass). Supports both
LoRA adapters and earlier merged checkpoints. Existing exports require
--overwrite; --no-save prints verification without saving forecasts.
The native median forecast, masked padding, flip invariance, and positivity
handling are shared across all modes. Default station-years match the paper.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "google/timesfm-2.5-200m-pytorch"
MODEL_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
HOLDOUT_STATIONS = ("ASP", "QIQ", "PAL", "KWA", "SYO")
HOLDOUT_YEARS = (2018, 2025, 2023, 2002, 2024)
HISTORY_LENGTHS = (15, 30, 60)
FORECAST_LENGTH = 15
MEDIAN_CHANNEL = 5  # Native output: point head, then quantiles 0.1,...,0.9.


def chronos_module():
    spec = importlib.util.spec_from_file_location(
        "forecast_common", Path(__file__).with_name("3.1.forecast_chronos.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pretrained_model(device):
    import torch
    from huggingface_hub import hf_hub_download
    from timesfm import TimesFM_2p5_200M_torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; allow GPU access or explicitly use --device cpu.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose an available device.")
    weights = hf_hub_download(
        MODEL_NAME, "model.safetensors", revision=MODEL_REVISION, local_files_only=True
    )
    pipeline = TimesFM_2p5_200M_torch()
    pipeline.model.device = torch.device(device)
    pipeline.model.device_count = 1
    # The installed from_pretrained() forces a config download even for local
    # weights. Its public checkpoint loader avoids that and disables torch.compile
    # (not required for inference, and not assumed to work with this MPS setup).
    pipeline.model.load_checkpoint(weights, torch_compile=False)
    pipeline.model.eval()
    config = pipeline.model.config
    if not np.isclose(config.quantiles[MEDIAN_CHANNEL - 1], 0.5):
        raise ValueError("Unexpected TimesFM quantile layout; cannot select median safely.")
    return pipeline


class MedianAdapter:
    """Adapt native TimesFM forecasts to the shared 3.1 median/batching interface."""

    quantiles = (0.5,)

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model = pipeline.model

    def predict(self, inputs, *, prediction_length, context_length, batch_size, cross_learning):
        import torch

        if cross_learning or prediction_length != FORECAST_LENGTH:
            raise ValueError("Expected independent univariate 15-minute forecasts.")
        if any(len(row) != context_length for row in inputs):
            raise ValueError("History length does not match the requested context.")
        n = len(inputs)
        # Native timesfm 2.0.0 pads partial batches with float64 arrays, which
        # promotes the whole batch before its MPS transfer. Pad in float32 here
        # so the native padding path is not entered; keep only real origins.
        padded = [np.asarray(row, dtype=np.float32) for row in inputs]
        missing = (-n) % self.pipeline.global_batch_size
        padded.extend(np.zeros(context_length, dtype=np.float32) for _ in range(missing))
        with torch.inference_mode():
            _, quantiles = self.pipeline.forecast(horizon=prediction_length, inputs=padded)
        quantiles = np.asarray(quantiles)
        if quantiles.shape != (len(padded), FORECAST_LENGTH, 10):
            raise ValueError(f"Unexpected TimesFM forecast shape: {quantiles.shape}")
        median = quantiles[:n, :, MEDIAN_CHANNEL].astype(np.float32)
        return [torch.from_numpy(row.copy()).reshape(1, 1, FORECAST_LENGTH) for row in median]


def self_test(common):
    import torch

    class FakeTimesFM:
        model = SimpleNamespace(device=torch.device("cpu"), eval=lambda: None)
        global_batch_size = 3

        def forecast(self, horizon, inputs):
            assert horizon == 15
            assert len(inputs) % self.global_batch_size == 0
            assert all(row.dtype == np.float32 for row in inputs)
            q = np.zeros((len(inputs), horizon, 10), dtype=np.float32)
            for i, row in enumerate(inputs):
                q[i, :, MEDIAN_CHANNEL] = row[-1] + 2
            return np.full((len(inputs), horizon), -999), q

    for length in HISTORY_LENGTHS:
        histories = np.arange(7 * length, dtype=np.float32).reshape(7, length)
        prediction, timing = common.forecast(
            MedianAdapter(FakeTimesFM()), histories, length, batch_size=3, progress=False
        )
        np.testing.assert_array_equal(prediction, np.repeat(histories[:, -1:] + 2, 15, axis=1))
        assert timing["n_origins"] == 7
    print("PASS: L15/L30/L60, explicit median (not point head), all origins, partial batches; no files saved.")


def sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_manifest(checkpoint_dir, expected_mode=None):
    manifest = json.loads((checkpoint_dir / "training_manifest.json").read_text())
    if not manifest.get("completed") or manifest.get("finetune_mode") != "lora":
        raise ValueError("A completed LoRA checkpoint is required.")
    if manifest.get("history_length") not in (15, 30, 60) or manifest.get("forecast_length") != 15:
        raise ValueError("Checkpoint does not match the 15-value forecast task.")
    if expected_mode == "budget" and manifest.get("num_steps") != 1000:
        raise ValueError("Use the full-pass forecasting script for this checkpoint.")
    if expected_mode == "full" and manifest.get("train_mode") != "one_pass_without_replacement":
        raise ValueError("A one-pass checkpoint is required.")
    return manifest


def load_finetuned_model(checkpoint_dir, manifest, device):
    training = sibling("timesfm_training", "2.2.finetune_timesfm.py")
    if (manifest["base_model"], manifest["base_revision"]) != (training.MODEL_NAME, training.MODEL_REVISION):
        raise ValueError("Checkpoint base identity differs from the pinned loader.")
    if manifest.get("checkpoint_format") == "peft_adapter":
        return training.load_adapter_pipeline(checkpoint_dir, device)
    # Earlier full-pass runs saved merged weights without a checkpoint_format.
    if manifest.get("weights_file", "model.safetensors") != "model.safetensors":
        raise ValueError("Unrecognized checkpoint format.")
    from safetensors.torch import load_file

    pipeline = training.load_pretrained_pipeline(device)
    pipeline.model.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")), strict=True)
    pipeline.model.eval()
    return pipeline



def main():
    common = chronos_module()
    helper = common.input_module()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    helper.add_eval_input_arguments(parser)
    parser.add_argument("--history-lengths", type=int, nargs="+", choices=HISTORY_LENGTHS,
                        default=None)
    parser.add_argument("--result-dir", type=Path, default=common.RESULT_DIR)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(common)
        return
    manifest = read_manifest(args.checkpoint_dir) if args.checkpoint_dir else None
    if manifest is not None:
        trained_length = int(manifest["history_length"])
        if args.history_lengths is not None and args.history_lengths != [trained_length]:
            parser.error("An adapted checkpoint requires its trained context length.")
        args.history_lengths = [trained_length]
    args.history_lengths = args.history_lengths or list(HISTORY_LENGTHS)
    method = ("timesfm" if manifest is None else
              "timesfm_ft" if manifest.get("num_steps") == 1000 else "timesfm_ft_full")
    if args.batch_size < 1 or args.threads < 1 or args.stride_minutes < 1:
        parser.error("Batch size, threads and stride must be positive.")
    if args.max_origins < 0 or args.seed < 0:
        parser.error("Origin cap and seed must be nonnegative.")
    if len(set(args.history_lengths)) != len(args.history_lengths):
        parser.error("History lengths must be unique.")
    if not args.no_save:
        args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.input_file is not None and (args.stations or args.koppen_holdout_stations or args.years):
        parser.error("Use --input-file or station/year arguments, not both.")
    if args.input_file is None and not args.stations and not args.koppen_holdout_stations:
        if args.years:
            parser.error("--years requires --stations.")
        args.stations, args.years = list(HOLDOUT_STATIONS), list(HOLDOUT_YEARS)
    if args.years and args.stations and len(args.years) != len(args.stations):
        parser.error("Provide one year for each station.")

    import torch
    from timesfm import ForecastConfig

    torch.set_num_threads(args.threads)
    started = perf_counter()
    pipeline = (load_pretrained_model(args.device) if manifest is None else
                load_finetuned_model(args.checkpoint_dir, manifest, args.device))
    common.synchronize(pipeline.model.device)
    print(f"TimesFM 2.5 at {MODEL_REVISION}; device={pipeline.model.device}; "
          f"load={perf_counter() - started:.3f}s", flush=True)
    examples, metadata = helper.load_eval_examples(
        input_file=args.input_file, data_root=args.data_root, stations=args.stations,
        years=args.years, koppen_holdout_stations=args.koppen_holdout_stations,
        max_origins=args.max_origins, seed=args.seed, stride_minutes=args.stride_minutes,
        non_overlapping=args.non_overlapping,
    )
    if not args.no_save and not args.overwrite:
        for station in set(examples["station"]):
            for length in args.history_lengths:
                path = args.result_dir / f"{station}_{method}_{length}.txt"
                if path.exists():
                    parser.error(f"Refusing to overwrite {path}; use --overwrite or --no-save.")
    for source in metadata["sources"]:
        print(f"Eval input: {source['station']} {source['year']}: {source['selected_count']:,} origins", flush=True)
    persistence = np.repeat(examples["history"][:, -1:], FORECAST_LENGTH, axis=1)
    common.print_report("Persistence", common.evaluate(
        persistence, examples["target"], examples["station"], examples["history"]
    ))
    for length in args.history_lengths:
        pipeline.compile(ForecastConfig(
            max_context=length, max_horizon=FORECAST_LENGTH, per_core_batch_size=args.batch_size
        ))
        prediction, timing = common.forecast(
            MedianAdapter(pipeline), helper.histories_for(examples, length), length, args.batch_size
        )
        print(f"L={length}: warm-up={timing['warmup_seconds']:.3f}s; "
              f"inference={timing['inference_seconds']:.3f}s; "
              f"amortized={1000 * timing['seconds_per_origin']:.3f}ms/origin", flush=True)
        common.print_report(f"{method}_L{length}", common.evaluate(
            prediction, examples["target"], examples["station"], examples["history"]
        ))
        if args.no_save:
            continue
        files = common.write_ghi_results(args.result_dir, examples, metadata, method, prediction,
                                        length, data_root=args.data_root)
        print(f"Wrote {', '.join(files)}", flush=True)


if __name__ == "__main__":
    main()
