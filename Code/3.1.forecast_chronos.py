"""Chronos-2 forecasting with pretrained weights or a local LoRA checkpoint.

  python -B Code/3.1.forecast_chronos.py --stations QIQ --years 2025 --max-origins 32 --no-save
  python -B Code/3.1.forecast_chronos.py --stations QIQ --years 2025 --checkpoint-dir Data/models/chronos-2-full-L60
  python -B Code/3.1.forecast_chronos.py --self-test

Without --checkpoint-dir use the cached pinned pretrained model. A local
checkpoint may be a training directory or its finetuned-ckpt subdirectory.
Adapted forecasts use the context length recorded in the training manifest.
Output names retain chronos, chronos_ft (1000 updates), and chronos_ft_full
(full pass) prefixes. Existing exports require --overwrite; --no-save only
prints verification. Station years should be supplied explicitly for the paper.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "Data" / "result"
CLEAR_SKY_SCALE = 100
INVALID_CLEAR_SKY = int(np.iinfo(np.uint32).max)
GHI_DECIMALS = 2

MODEL_NAME = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
HISTORY_LENGTHS = (15, 30, 60)
FORECAST_LENGTH = 15
BATCH_SIZE = 64
DEVICE = "cpu"
VARIABILITY_THRESHOLD = 0.05


def input_module():
    """Reuse script 1.2's input loader, validation and history suffixes."""
    path = Path(__file__).with_name("1.2.prepare_forecast_input.py")
    spec = importlib.util.spec_from_file_location("bsrn_forecast_input", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synchronize(device):
    import torch

    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_pretrained_model(device=DEVICE):
    """Load only the pinned cached full Chronos-2, in float32 evaluation mode."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    from chronos import Chronos2Pipeline

    if device not in ("mps", "cuda", "cpu"):
        raise ValueError("Choose mps, cuda, or cpu explicitly.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable. Run outside the sandbox with GPU permission, "
                           "or explicitly choose --device cpu.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose an available device explicitly.")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, device_map=device,
        torch_dtype=torch.float32, local_files_only=True)
    pipeline.model.eval()
    return pipeline


def forecast(pipeline, histories, history_length, batch_size=BATCH_SIZE, progress=True):
    """Return (median predictions, timing); warm the first batch separately.

    Measured inference includes batching, transfers and output checks. A final
    partial batch can have first-use overhead even after full-batch warm-up.
    Per-origin time is amortized throughput, not single-request latency.
    """
    import torch

    histories = np.asarray(histories, dtype=np.float32)
    if (history_length not in HISTORY_LENGTHS or histories.ndim != 2
            or histories.shape[1] != history_length or len(histories) == 0
            or batch_size < 1 or not np.isfinite(histories).all()
            or np.any(histories < 0)):
        raise ValueError("Expected finite nonnegative (N, L) histories and a positive batch size.")
    quantiles = np.asarray(pipeline.quantiles)
    median_indices = np.flatnonzero(np.isclose(quantiles, 0.5, rtol=0, atol=1e-8))
    if len(median_indices) != 1:
        raise ValueError("The model must expose exactly one 0.5 quantile.")
    median_index = int(median_indices[0])
    device = pipeline.model.device
    pipeline.model.eval()

    def predict_batch(batch):
        outputs = pipeline.predict(
            [row.copy() for row in batch], prediction_length=FORECAST_LENGTH,
            context_length=history_length, batch_size=batch_size, cross_learning=False)
        if len(outputs) != len(batch):
            raise ValueError("Chronos returned an unexpected number of forecasts.")
        if any(tuple(item.shape) != (1, len(quantiles), FORECAST_LENGTH) for item in outputs):
            raise ValueError("Expected univariate (1, quantiles, 15) outputs.")
        result = torch.stack(outputs)[:, 0, median_index, :].detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite model forecast; no origins will be silently dropped.")
        return result

    synchronize(device)
    started = perf_counter()
    predict_batch(histories[:batch_size])
    synchronize(device)
    warmup_seconds = perf_counter() - started
    predictions = np.empty((len(histories), FORECAST_LENGTH), dtype=np.float32)
    synchronize(device)
    started = perf_counter()
    for batch_number, start in enumerate(range(0, len(histories), batch_size), 1):
        stop = min(start + batch_size, len(histories))
        predictions[start:stop] = predict_batch(histories[start:stop])
        if progress and batch_number % 100 == 0:
            print(f"L={history_length}: {stop:,}/{len(histories):,} origins", flush=True)
    synchronize(device)
    seconds = perf_counter() - started
    timing = dict(warmup_seconds=warmup_seconds, inference_seconds=seconds,
                  seconds_per_origin=seconds / len(histories), n_origins=len(histories),
                  batch_size=batch_size, device=str(device))
    return predictions, timing


def _errors(error):
    mae = np.abs(error).mean(axis=0)
    mse = np.square(error).mean(axis=0)
    return dict(n_origins=len(error), mae=float(mae.mean()), rmse=float(np.sqrt(mse.mean())),
                mae_by_lead=mae.tolist(), rmse_by_lead=np.sqrt(mse).tolist())


def evaluate(predictions, targets, station_ids, common_histories=None):
    """Lead/station errors; equal-station RMSE is sqrt(mean station MSE).

    Optional variability classes use the COMMON 60-minute past history for all
    input lengths, so stratum membership cannot change with model configuration.
    No future information is used for classification. Empty strata are explicit.
    """
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    station_ids = np.asarray(station_ids)
    if (predictions.ndim != 2 or predictions.shape != targets.shape
            or predictions.shape[1] != FORECAST_LENGTH or not len(targets)
            or station_ids.shape != (len(targets),)
            or not np.isfinite(predictions).all() or not np.isfinite(targets).all()):
        raise ValueError("Expected matching finite (N, 15) forecasts/targets and N station IDs.")
    error = predictions.astype(np.float64) - targets.astype(np.float64)
    classes = {"all": np.ones(len(targets), dtype=bool)}
    if common_histories is not None:
        common_histories = np.asarray(common_histories, dtype=np.float64)
        if common_histories.shape != (len(targets), 60) or not np.isfinite(common_histories).all():
            raise ValueError("Variability classification requires finite common (N, 60) histories.")
        variable = np.std(common_histories, axis=1) > VARIABILITY_THRESHOLD
        classes.update(low_variability=~variable, variable=variable)
    report = {}
    for label, mask in classes.items():
        if not np.any(mask):
            report[label] = dict(n_origins=0, pooled=None, equal_station=None, by_station={})
            continue
        station_metrics = {str(station): _errors(error[mask & (station_ids == station)])
                           for station in np.unique(station_ids[mask])}
        values = list(station_metrics.values())
        equal = dict(n_origins=int(mask.sum()), n_stations=len(values),
                     mae=float(np.mean([v["mae"] for v in values])),
                     rmse=float(np.sqrt(np.mean([v["rmse"] ** 2 for v in values]))),
                     mae_by_lead=np.mean([v["mae_by_lead"] for v in values], axis=0).tolist(),
                     rmse_by_lead=np.sqrt(np.mean(
                         np.square([v["rmse_by_lead"] for v in values]), axis=0)).tolist())
        report[label] = dict(n_origins=int(mask.sum()), pooled=_errors(error[mask]),
                             equal_station=equal, by_station=station_metrics)
    return report


def resolve_source_path(source, data_root=None):
    """Resolve archived source metadata against the current data directory."""
    original = Path(source["path"])
    if original.is_file():
        return original
    root = Path(data_root) if data_root is not None else PROJECT_ROOT / "Data/bsrn_1min"
    station = source["station"].lower()
    candidate = root / station / f"{station}_{source['year']}.parquet"
    if not candidate.is_file():
        raise FileNotFoundError(f"Source data required for GHI conversion: {candidate}")
    return candidate


def write_ghi_results(result_dir, examples, input_metadata, method, forecast_kappa, history_length,
                      data_root=None):
    """Write {STATION}_{method}_{L}.txt — one row/origin; Time, Obs1..15, Fcst1..15 (W m-2)."""
    result_dir = Path(result_dir)
    forecast_kappa = np.asarray(forecast_kappa, dtype=np.float64)
    n = len(examples["history"])
    if forecast_kappa.shape != (n, FORECAST_LENGTH):
        raise ValueError(f"Expected forecast shape ({n}, {FORECAST_LENGTH}).")
    leads = np.arange(1, FORECAST_LENGTH + 1, dtype=np.int64)
    clear = {}
    for i, source in enumerate(input_metadata["sources"]):
        centi = pq.read_table(resolve_source_path(source, data_root),
                              columns=["ghi_clear_centi"])["ghi_clear_centi"].to_numpy()
        used = examples["source_index"] == i
        if np.any(used):
            row_idx = np.unique(
                examples["origin_row"][used][:, None] + leads
            )
            if np.any(centi[row_idx] == INVALID_CLEAR_SKY):
                raise ValueError(f"Invalid clear-sky for forecast rows in {source['path']}.")
        clear[i] = centi.astype(np.float64) / CLEAR_SKY_SCALE
    lead_names = [f"Obs{j}" for j in leads] + [f"Fcst{j}" for j in leads]
    header = "Time\t" + "\t".join(lead_names)
    written = []
    for station in sorted({str(s) for s in examples["station"]}):
        mask = examples["station"] == station
        src = int(np.unique(examples["source_index"][mask])[0])
        clear_leads = clear[src][examples["origin_row"][mask][:, None] + leads]
        obs = examples["target"][mask] * clear_leads
        fcst = forecast_kappa[mask] * clear_leads
        times = examples["origin_utc_minute"][mask].reshape(-1, 1)
        table = np.hstack((times, obs, fcst))
        path = result_dir / f"{station}_{method}_{history_length}.txt"
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            stream.write(header + "\n")
            for row in table:
                stream.write(f"{int(row[0])}\t")
                stream.write("\t".join(f"{value:.{GHI_DECIMALS}f}" for value in row[1:]))
                stream.write("\n")
        tmp.replace(path)
        written.append(path.name)
    return written


def print_report(label, report):
    """Print full lead-wise and per-station summaries without creating a report file."""
    for query_class, values in report.items():
        if values["n_origins"] == 0:
            print(f"{label} {query_class}: no origins")
            continue
        pooled, equal = values["pooled"], values["equal_station"]
        print(f"{label} {query_class}: N={values['n_origins']:,}; "
              f"pooled MAE={pooled['mae']:.6f}, RMSE={pooled['rmse']:.6f}; "
              f"equal-station MAE={equal['mae']:.6f}, RMSE={equal['rmse']:.6f}")
        print("  Leads +1..+15 MAE: " + ", ".join(f"{x:.6f}" for x in pooled["mae_by_lead"]))
        print("  Leads +1..+15 RMSE: " + ", ".join(f"{x:.6f}" for x in pooled["rmse_by_lead"]))
        for station, metrics in values["by_station"].items():
            print(f"  {station}: N={metrics['n_origins']:,}; "
                  f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")


def self_test():
    """Check explicit quantile selection, partial batches and metric weighting in memory."""
    from types import SimpleNamespace
    import torch

    class FakePipeline:
        quantiles = [0.9, 0.5, 0.1]

        def __init__(self):
            self.model = SimpleNamespace(device=torch.device("cpu"), eval=lambda: None)
            self.calls = []

        def predict(self, inputs, **kwargs):
            self.calls.append(kwargs)
            # Deliberate crossing: median over outputs would select the WRONG value.
            return [torch.tensor(np.repeat(
                (np.array([9, 20, -9], np.float32) + row[-1])[None, :, None], 15, axis=2))
                    for row in inputs]

    for length in HISTORY_LENGTHS:
        histories = np.arange(7 * length, dtype=np.float32).reshape(7, length)
        pipeline = FakePipeline()
        prediction, timing = forecast(pipeline, histories, length, batch_size=3, progress=False)
        np.testing.assert_array_equal(prediction, np.repeat(histories[:, -1:] + 20, 15, axis=1))
        assert timing["n_origins"] == 7 and len(pipeline.calls) == 4
        assert all(c["cross_learning"] is False and c["prediction_length"] == 15
                   and c["context_length"] == length for c in pipeline.calls)
    prediction = np.zeros((7, 15))
    prediction[0] = 2
    metrics = evaluate(prediction, np.zeros_like(prediction), np.array(["A"] + ["B"] * 6),
                       np.zeros((7, 60)))
    assert np.isclose(metrics["all"]["pooled"]["mae"], 2 / 7)
    assert np.isclose(metrics["all"]["equal_station"]["mae"], 1)
    assert np.isclose(metrics["all"]["equal_station"]["rmse"], np.sqrt(2))
    assert metrics["variable"]["n_origins"] == 0
    print("Checks passed: median quantile, all origins, partial batches, station weighting, empty strata.")


PEFT_IMPORT_ALLOWLIST = ["chronos.chronos2.model"]


def read_training_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing training manifest: {path}")
    return json.loads(path.read_text())


def load_finetuned_model(
    checkpoint_dir,
    manifest_path,
    device=DEVICE,
    evaluation_history_lengths=None,
):
    """Load LoRA checkpoint (merged by Chronos) and verify manifest vs evaluation."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    from chronos import Chronos2Pipeline

    checkpoint_dir = Path(checkpoint_dir)
    adapter_config = checkpoint_dir / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(f"No LoRA adapter at {checkpoint_dir} (expected adapter_config.json).")

    manifest = read_training_manifest(manifest_path)
    train_length = int(manifest["history_length"])
    if manifest.get("finetune_mode") != "lora":
        raise ValueError(f"Expected LoRA manifest; got finetune_mode={manifest.get('finetune_mode')!r}.")

    if evaluation_history_lengths is not None:
        lengths = sorted(set(int(x) for x in evaluation_history_lengths))
        if train_length not in lengths:
            raise ValueError(
                f"Checkpoint trained with history_length={train_length}; "
                f"evaluation requested {lengths}."
            )
        if lengths != [train_length]:
            print(
                f"Note: fine-tuned model trained at L={train_length} only; "
                f"skipping other requested lengths {lengths}.",
                flush=True,
            )

    if device not in ("mps", "cuda", "cpu"):
        raise ValueError("Choose mps, cuda, or cpu explicitly.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; choose --device cpu explicitly.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose an available device explicitly.")

    pipeline = Chronos2Pipeline.from_pretrained(
        str(checkpoint_dir.resolve()),
        device_map=device,
        torch_dtype=torch.float32,
        local_files_only=True,
        import_allowlist=PEFT_IMPORT_ALLOWLIST,
    )
    pipeline.model.eval()
    return pipeline, manifest



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    helper = input_module()
    helper.add_eval_input_arguments(parser)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=RESULT_DIR,
        help="Directory for GHI text outputs (must already exist).",
    )
    parser.add_argument("--history-lengths", type=int, nargs="+", choices=HISTORY_LENGTHS,
                        default=None)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
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
    manifest = None
    if args.checkpoint_dir is not None:
        if (args.checkpoint_dir / "finetuned-ckpt").is_dir():
            args.checkpoint_dir = args.checkpoint_dir / "finetuned-ckpt"
        args.manifest = args.manifest or args.checkpoint_dir.parent / "training_manifest.json"
        manifest = read_training_manifest(args.manifest)
        trained_length = int(manifest["history_length"])
        if args.history_lengths is not None and args.history_lengths != [trained_length]:
            parser.error("An adapted checkpoint requires its trained context length.")
        args.history_lengths = [trained_length]
    elif args.manifest is not None:
        parser.error("--manifest requires --checkpoint-dir.")
    args.history_lengths = args.history_lengths or list(HISTORY_LENGTHS)
    method = ("chronos" if manifest is None else
              "chronos_ft" if manifest.get("num_steps") == 1000 else "chronos_ft_full")
    if args.batch_size < 1 or args.threads < 1 or len(set(args.history_lengths)) != len(args.history_lengths):
        parser.error("Counts must be positive and history lengths unique.")
    if args.max_origins < 0 or args.seed < 0 or args.stride_minutes < 1:
        parser.error("Use nonnegative --max-origins/--seed and positive --stride-minutes.")
    if args.years is not None and args.stations and len(args.years) != len(args.stations):
        parser.error("Provide one --years value for each --stations entry.")
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
    import torch

    torch.set_num_threads(args.threads)
    started = perf_counter()
    pipeline = (load_pretrained_model(args.device) if manifest is None else
                load_finetuned_model(args.checkpoint_dir, args.manifest, args.device)[0])
    synchronize(pipeline.model.device)
    load_seconds = perf_counter() - started
    print(f"{MODEL_NAME} at {MODEL_REVISION}; device={pipeline.model.device}; "
          f"model load={load_seconds:.3f}s", flush=True)
    persistence = np.repeat(examples["history"][:, -1:], FORECAST_LENGTH, axis=1)
    baseline = evaluate(persistence, examples["target"], examples["station"], examples["history"])
    print_report("Persistence", baseline)
    reports, timings = {"persistence": baseline}, {}
    if not args.no_save:
        args.result_dir.mkdir(parents=True, exist_ok=True)
        if not args.overwrite:
            for station in set(examples["station"]):
                for length in args.history_lengths:
                    path = args.result_dir / f"{station}_{method}_{length}.txt"
                    if path.exists():
                        parser.error(f"Refusing to overwrite {path}; use --overwrite or --no-save.")
    for length in args.history_lengths:
        prediction, timing = forecast(pipeline, helper.histories_for(examples, length),
                                      length, args.batch_size)
        label = f"{method}_L{length}"
        report = evaluate(prediction, examples["target"], examples["station"], examples["history"])
        reports[label], timings[str(length)] = report, timing
        print(f"L={length}: warm-up={timing['warmup_seconds']:.3f}s; "
              f"inference={timing['inference_seconds']:.3f}s; "
              f"amortized={1000 * timing['seconds_per_origin']:.3f}ms/origin", flush=True)
        print_report(label, report)
        if args.no_save:
            continue
        files = write_ghi_results(
            args.result_dir, examples, input_metadata, method, prediction, length,
            data_root=args.data_root,
        )
        print(f"Wrote {', '.join(files)}", flush=True)


if __name__ == "__main__":
    main()
