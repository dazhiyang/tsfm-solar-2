"""Fine-tune TimesFM 2.5 on the same pooled BSRN split as Chronos script 2.1.

Both limited and full-pass runs use the common 15-minute origin grid, the same
station-year day-blocked validation split, and one seeded permutation without
replacement. Omit --num-steps for one full pass or set an explicit budget.
Training is off by default.

  python -B Code/2.2.finetune_timesfm.py
  python -B Code/2.2.finetune_timesfm.py --history-length 15 --num-steps 1000 --train --output-dir PATH

Outputs (only with --train):
  Data/models/timesfm-2.5-finetuned-L{15,30,60}/adapter_model.safetensors
  .../adapter_config.json (LoRA; requires the pinned pretrained model)
  .../training_manifest.json
  .../loss_history.jsonl

Native TimesFM has no Chronos-style trainer; this script optimises median-quantile
MSE on κ with a gradient-enabled decode path and the same flip-invariance and
positivity handling as pretrained forecasting in 3.3. Adaptation uses LoRA on
attention projections (peft). torch.compile is disabled for
training. The exported adapter weights are
selected by minimum validation MSE, not automatically the final update. Losses
are flushed to JSONL every 100 updates and at the final step. These MSE values
are not numerically comparable to Chronos's training objective.
The CLI exports adapter-only LoRA weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "Data" / "finetuning_data.parquet"
MODEL_NAME = "google/timesfm-2.5-200m-pytorch"
MODEL_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
HISTORY_LENGTHS = (15, 30, 60)
FORECAST_LENGTH = 15
MEDIAN_INDEX = 5
RANDOM_SEED = 42
DEVICE = "mps"
DEFAULT_MAX_TRAIN_SERIES = None
DEFAULT_MAX_VAL_SERIES = 2048
DEFAULT_NUM_STEPS = 1000


def chronos_finetune_module():
    path = Path(__file__).with_name("2.1.finetune_chronos.py")
    spec = importlib.util.spec_from_file_location("chronos_finetune", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_size(context_length: int) -> int:
    return math.ceil(context_length / 32) * 32


def pad_batch_context(histories: np.ndarray, context_length: int, device):
    """Left-pad κ histories to a fixed context (multiple of patch size 32)."""
    import torch

    context = patch_size(context_length)
    batch = histories.shape[0]
    values = torch.zeros(batch, context, dtype=torch.float32, device=device)
    masks = torch.ones(batch, context, dtype=torch.bool, device=device)
    for row, series in enumerate(histories):
        length = int(series.shape[0])
        if length > context:
            series = series[-context:]
            length = context
        pad = context - length
        values[row, pad:] = torch.from_numpy(series.astype(np.float32, copy=False))
        masks[row, pad:] = False
    return values, masks


def decode_with_grad(module, horizon: int, inputs, masks):
    """TimesFM decode without torch.no_grad (for median MSE on κ)."""
    import torch
    from timesfm.torch import util

    revin = util.revin
    batch_size, context = inputs.shape[0], inputs.shape[1]
    num_decode_steps = (horizon - 1) // module.o
    num_input_patches = context // module.p
    decode_cache_size = num_input_patches + num_decode_steps * module.m

    patched_inputs = torch.reshape(inputs, (batch_size, -1, module.p))
    patched_masks = torch.reshape(masks, (batch_size, -1, module.p))

    n = torch.zeros(batch_size, device=inputs.device)
    mu = torch.zeros(batch_size, device=inputs.device)
    sigma = torch.zeros(batch_size, device=inputs.device)
    patch_mu, patch_sigma = [], []
    for i in range(num_input_patches):
        (n, mu, sigma), _ = util.update_running_stats(
            n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    last_n, last_mu, last_sigma = n, mu, sigma
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)

    decode_caches = [
        util.DecodeCache(
            next_index=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
            num_masked=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
            key=torch.zeros(batch_size, decode_cache_size, module.h, module.hd, device=inputs.device),
            value=torch.zeros(batch_size, decode_cache_size, module.h, module.hd, device=inputs.device),
        )
        for _ in range(module.x)
    ]

    normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
    (_, _, normed_outputs, _), decode_caches = module(normed_inputs, patched_masks, decode_caches)
    renormed_outputs = torch.reshape(
        revin(normed_outputs, context_mu, context_sigma, reverse=True),
        (batch_size, -1, module.o, module.q),
    )

    ar_outputs = []
    last_renormed_output = renormed_outputs[:, -1, :, module.aridx]
    for _ in range(num_decode_steps):
        new_patched_input = torch.reshape(last_renormed_output, (batch_size, module.m, module.p))
        new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool)
        n, mu, sigma = last_n, last_mu, last_sigma
        new_mus, new_sigmas = [], []
        for i in range(module.m):
            (n, mu, sigma), _ = util.update_running_stats(
                n, mu, sigma, new_patched_input[:, i], new_mask[:, i]
            )
            new_mus.append(mu)
            new_sigmas.append(sigma)
        last_n, last_mu, last_sigma = n, mu, sigma
        new_mu = torch.stack(new_mus, dim=1)
        new_sigma = torch.stack(new_sigmas, dim=1)
        new_normed_input = revin(new_patched_input, new_mu, new_sigma, reverse=False)
        (_, _, new_normed_output, _), decode_caches = module(new_normed_input, new_mask, decode_caches)
        new_renormed_output = torch.reshape(
            revin(new_normed_output, new_mu, new_sigma, reverse=True),
            (batch_size, module.m, module.o, module.q),
        )
        ar_outputs.append(new_renormed_output[:, -1, ...])
        last_renormed_output = new_renormed_output[:, -1, :, module.aridx]

    if num_decode_steps > 0:
        ar_renormed_outputs = torch.stack(ar_outputs, dim=1)
        to_cat = [renormed_outputs[:, -1, ...], ar_renormed_outputs.reshape(batch_size, -1, module.q)]
        full = torch.cat(to_cat, dim=1)
    else:
        full = renormed_outputs[:, -1, ...]
    return full[:, :horizon, MEDIAN_INDEX]


def forecast_with_grad(module, horizon, inputs, masks):
    """Median matching 3.5's default flip invariance and positivity inference."""
    import torch

    positive = decode_with_grad(module, horizon, inputs, masks)
    negative = decode_with_grad(module, horizon, -inputs, masks)
    forecast = (positive - negative) / 2
    is_positive = torch.all(inputs >= 0, dim=-1, keepdim=True)
    return torch.where(is_positive, forecast.clamp_min(0), forecast)


def index_to_arrays(base, frame, cache, history_length, seed, max_series):
    rows = base.sample_index_rows(frame, max_series, seed)
    histories, targets = [], []
    for _, row in rows.iterrows():
        codes = cache.kappa(row["file_path"])
        history, target = base.example_window(codes, row["origin_index"], history_length)
        histories.append(history)
        targets.append(target)
    if not histories:
        raise ValueError("No training windows materialized from index.")
    return histories, targets


def synthetic_arrays(n_series, history_length, seed):
    rng = np.random.default_rng(seed)
    histories, targets = [], []
    for _ in range(n_series):
        history = rng.uniform(0.05, 1.0, size=history_length).astype(np.float32)
        target = np.clip(
            history[-1] + rng.normal(0, 0.02, size=FORECAST_LENGTH), 0.0, 1.2
        ).astype(np.float32)
        histories.append(history)
        targets.append(target)
    return histories, targets


def load_pretrained_pipeline(device: str):
    import torch
    from huggingface_hub import hf_hub_download
    from timesfm import TimesFM_2p5_200M_torch
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable for TimesFM training trial.")
    weights = hf_hub_download(
        MODEL_NAME, "model.safetensors", revision=MODEL_REVISION, local_files_only=True
    )
    pipeline = TimesFM_2p5_200M_torch()
    # Native wrapper uses a class-level model; reload must not reuse LoRA layers.
    pipeline.model = TimesFM_2p5_200M_torch_module()
    pipeline.model.device = torch.device(device)
    pipeline.model.device_count = 1
    pipeline.model.load_checkpoint(weights, torch_compile=False)
    if not np.isclose(pipeline.model.config.quantiles[MEDIAN_INDEX - 1], 0.5):
        raise ValueError("Unexpected TimesFM quantile layout.")
    return pipeline


def prepare_model(model, train_heads_only: bool):
    if train_heads_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        # The continuous quantile head is unused by the median objective.
        for block in (model.output_projection_point,):
            for parameter in block.parameters():
                parameter.requires_grad = True
        return model, "heads"
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA trial requires peft in tf_timesfm.") from exc
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["qkv_proj", "out"],
        bias="none",
    )
    adapted = get_peft_model(model, config)
    return adapted, "lora"


def load_adapter_pipeline(output_dir, device):
    """Reload adapters onto the exact pinned base; merge only in memory for inference."""
    from peft import PeftModel

    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "training_manifest.json").read_text())
    if (manifest["base_model"], manifest["base_revision"]) != (MODEL_NAME, MODEL_REVISION):
        raise ValueError("Adapter base-model identity does not match the pinned loader.")
    if manifest.get("checkpoint_format") != "peft_adapter":
        raise ValueError("Expected an adapter-only checkpoint.")
    pipeline = load_pretrained_pipeline(device)
    adapted = PeftModel.from_pretrained(pipeline.model, output_dir, is_trainable=False)
    pipeline.model = adapted.merge_and_unload()
    pipeline.model.eval()
    return pipeline


def batch_tensors(histories, targets, indices, history_length, device):
    import torch

    batch_hist = np.stack([histories[i] for i in indices], axis=0)
    batch_targ = np.stack([targets[i] for i in indices], axis=0)
    inputs, masks = pad_batch_context(batch_hist, history_length, device)
    target_t = torch.from_numpy(batch_targ).to(device)
    return inputs, masks, target_t


def evaluate_mse(model, histories, targets, history_length, batch_size, device):
    import torch

    model.eval()
    squared_error, n_values = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(histories), batch_size):
            indices = list(range(start, min(start + batch_size, len(histories))))
            inputs, masks, target_t = batch_tensors(histories, targets, indices, history_length, device)
            pred = forecast_with_grad(model, FORECAST_LENGTH, inputs, masks)
            squared_error += torch.sum((pred - target_t) ** 2).item()
            n_values += target_t.numel()
    model.train()
    if not n_values:
        raise ValueError("Validation pool is empty.")
    return squared_error / n_values


def finetune(
    pipeline,
    histories,
    targets,
    val_histories,
    val_targets,
    history_length,
    output_dir,
    batch_size,
    num_steps,
    learning_rate,
    seed,
    train_heads_only,
    smoke_test,
    index_file,
    index_metadata,
    n_train_index_rows,
    train_frame=None,
    cache=None,
    base=None,
    evaluation_steps=100,
    batch_indices=None,
    manifest_extra=None,
    save_adapter_only=False,
):
    import torch
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = str(pipeline.model.device)
    torch.manual_seed(seed)
    model, finetune_mode = prepare_model(pipeline.model, train_heads_only)
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / num_steps)
    )
    rng = np.random.default_rng(seed)
    n_train = len(train_frame) if train_frame is not None else len(histories)
    if not n_train:
        raise ValueError("Training pool is empty.")
    started = perf_counter()
    best_loss, best_step, best_state = float("inf"), None, None
    interval_error, interval_values = 0.0, 0
    history_path = output_dir / "loss_history.jsonl"
    def record(event, step, **metrics):
        row = dict(history_length=history_length, step=step,
                   elapsed_seconds=perf_counter() - started, event=event, **metrics)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(
        f"TimesFM {finetune_mode} fit: train={n_train:,}, val={len(val_histories):,}, "
        f"L={history_length}, steps={num_steps}, batch_size={batch_size}",
        flush=True,
    )
    # The CLI supplies a finite seeded permutation for both limited and full-pass runs.
    batches = (rng.integers(0, n_train, size=batch_size) for _ in range(num_steps))
    if batch_indices is not None:
        batches = iter(batch_indices)
    completed_steps, presented_origins = 0, 0
    for step, indices in enumerate(batches, start=1):
        if step > num_steps:
            raise ValueError("Batch iterator exceeded the requested update count.")
        presented_origins += len(indices)
        if train_frame is not None:
            batch_h, batch_t = index_to_arrays(
                base, train_frame.iloc[indices], cache, history_length, seed, None
            )
            inputs, masks, target_t = batch_tensors(
                batch_h, batch_t, range(len(batch_h)), history_length, device
            )
        else:
            inputs, masks, target_t = batch_tensors(histories, targets, indices, history_length, device)
        pred = forecast_with_grad(model, FORECAST_LENGTH, inputs, masks)
        loss = torch.mean((pred - target_t) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        interval_error += loss.item() * target_t.numel()
        interval_values += target_t.numel()
        if step % 100 == 0 or step == num_steps:
            record("training", step, train_mse=interval_error / interval_values, learning_rate=lr_used)
            interval_error, interval_values = 0.0, 0
        if step % evaluation_steps == 0 or step == num_steps:
            val_mse = evaluate_mse(model, val_histories, val_targets, history_length, batch_size, device)
            if not np.isfinite(val_mse):
                raise FloatingPointError(f"Nonfinite validation MSE at step {step}")
            record("validation", step, val_mse=val_mse)
            if val_mse < best_loss:
                best_loss, best_step = val_mse, step
                best_state = {name: p.detach().cpu().clone()
                              for name, p in model.named_parameters() if p.requires_grad}
            print(
                f"step {step}/{num_steps}: train_mse={loss.item():.6f}, val_mse={val_mse:.6f}",
                flush=True,
            )
        completed_steps = step

    if completed_steps != num_steps:
        raise ValueError(f"Batch iterator ended after {completed_steps}/{num_steps} updates.")

    # Frozen base weights never changed; retaining only trainable best weights
    # avoids repeatedly saving a full ~200M-parameter model at each evaluation.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in best_state:
                parameter.copy_(best_state[name].to(parameter.device))
    adapter_export = save_adapter_only and finetune_mode == "lora"
    if adapter_export:
        config = model.peft_config["default"]
        config.base_model_name_or_path = MODEL_NAME
        config.revision = MODEL_REVISION
        model.save_pretrained(output_dir, safe_serialization=True, save_embedding_layers=False)
        weights_path = output_dir / "adapter_model.safetensors"
    else:
        weights_path = output_dir / "model.safetensors"
        state = model.state_dict()
        if hasattr(model, "merge_and_unload"):
            state = model.merge_and_unload().state_dict()
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}, str(weights_path))
    manifest = {
        "script": Path(__file__).name,
        "base_model": MODEL_NAME,
        "base_revision": MODEL_REVISION,
        "finetune_mode": finetune_mode,
        "history_length": history_length,
        "forecast_length": FORECAST_LENGTH,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_train_series": n_train,
        "n_val_series": len(val_histories),
        "n_train_index_rows": int(n_train_index_rows),
        "n_presented_origins": presented_origins,
        "train_mode": "stream" if train_frame is not None else "materialize",
        "sampling": "random_origins_with_replacement",
        "random_seed": seed,
        "validation_rule": index_metadata.get("validation_rule"),
        "test_station_years": index_metadata.get("test_station_years"),
        "best_step": best_step,
        "best_validation_mse": best_loss,
        "checkpoint_selection": "minimum_validation_mse",
        "evaluation_steps": evaluation_steps,
        "loss_history_file": history_path.name,
        "elapsed_seconds": perf_counter() - started,
        "force_flip_invariance": True,
        "infer_is_positive": True,
        "completed": True,
        "index_file": str(index_file) if index_file else None,
        "dataset_id": index_metadata.get("dataset_id"),
        "smoke_test": smoke_test,
        "weights_file": weights_path.name,
        "checkpoint_format": "peft_adapter" if adapter_export else "merged_full_model",
        "loss": "forecast_median_mse_on_kappa",
    }
    manifest.update(manifest_extra or {})
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {weights_path.resolve()}", flush=True)
    return weights_path


def seeded_batches(n_rows, batch_size, seed):
    """Yield one seeded permutation without replacement, including its partial batch."""
    order = np.random.default_rng(seed).permutation(n_rows)
    for start in range(0, n_rows, batch_size):
        yield order[start:start + batch_size]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--history-length", type=int, default=15, choices=HISTORY_LENGTHS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default=DEVICE)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--num-steps", type=int,
                        help="Seeded no-replacement update budget (default: one full pass).")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-val-series", type=int, default=DEFAULT_MAX_VAL_SERIES)
    parser.add_argument("--evaluation-steps", type=int, default=1000)
    args = parser.parse_args()
    counts = [args.batch_size, args.max_val_series, args.evaluation_steps]
    if args.num_steps is not None:
        counts.append(args.num_steps)
    if min(counts) < 1 or args.seed < 0:
        parser.error("Counts must be positive and seed nonnegative.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if args.train and args.output_dir is None:
        parser.error("Training requires an explicit new --output-dir.")
    if args.train and args.output_dir.exists():
        parser.error("Output directory exists; choose a new path.")
    return args


def main():
    args = parse_args()
    base = chronos_finetune_module()
    if args.preflight:
        import torch
        print(f"torch={torch.__version__}, mps={torch.backends.mps.is_available()}", flush=True)
        return
    print("Reading and thinning the existing index; no new index file...", flush=True)
    frame, metadata = base.load_cadence_index(args.index_file)
    train, val = base.blocked_split(frame, args.seed)
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
    if not args.train:
        print("Inspection only: no training started and no files saved.")
        return

    n_validation_pool_rows = len(val)
    cache = base.KappaCache(base.index_module())
    val = base.sample_index_rows(val, args.max_val_series, args.seed)
    val_h, val_t = index_to_arrays(base, val, cache, args.history_length, args.seed, None)
    pipeline = load_pretrained_pipeline(args.device)
    train_mode = ("one_pass_without_replacement" if steps == full_pass_steps
                  else "fixed_updates_without_replacement")
    finetune(
        pipeline, histories=None, targets=None, val_histories=val_h, val_targets=val_t,
        history_length=args.history_length, output_dir=args.output_dir,
        batch_size=args.batch_size, num_steps=steps, learning_rate=args.learning_rate,
        seed=args.seed, train_heads_only=False, smoke_test=False,
        index_file=args.index_file.resolve(), index_metadata=metadata,
        n_train_index_rows=len(train), train_frame=train, cache=cache, base=base,
        evaluation_steps=args.evaluation_steps,
        batch_indices=islice(seeded_batches(len(train), args.batch_size, args.seed), steps),
        save_adapter_only=True,
        manifest_extra=dict(
            script=Path(__file__).name, train_mode=train_mode,
            sampling="seeded_permutation_without_replacement",
            origin_cadence_minutes=base.CADENCE, cadence_offset_minutes=0,
            validation_rule="station_year_utc_day_fraction",
            validation_fraction=base.VALIDATION_FRACTION, validation_seed=args.seed,
            n_validation_pool_rows=n_validation_pool_rows,
            n_presented_origins=presented_origins,
        ),
    )


if __name__ == "__main__":
    main()
