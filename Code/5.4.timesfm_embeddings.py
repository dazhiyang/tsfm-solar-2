"""Paired layer-wise TimesFM embeddings; never starts training.

Run later, after training has finished (not launched when this script was written):
  python -B Code/5.4.timesfm_embeddings.py \
      --run --checkpoint-dir Data/models/timesfm-2.5-full-L60 \
      --output Data/result/timesfm_embeddings_L60.parquet

Compare the pinned pretrained model with one completed full-pass checkpoint.
Reuse the common 60+15 eligibility cohort, taking the checkpoint's context suffix.
Sample up to 1000 origins per station uniformly without replacement, independent
of forecast performance; also retain the six fixed illustrations from 5.1.
These added illustrations are flagged and excluded from fitting PCA. Ramp groups
use the observed median of the FIRST five targets minus origin kappa (+/-0.25).
Other origins are labelled 'Other', not assumed to be steady.

Capture the final context token after tokenization and selected transformer
blocks. A token represents a 32-minute patch, NOT an individual observation.
Keep positive-input and sign-flipped inference branches separate. For each
stage/branch, fit ONE scaler and PCA to the pooled pretrained/fine-tuned random
sample, then project both models and illustrations into that same coordinate
system. Axes remain stage/branch specific; no cross-layer coordinate comparison.

One Parquet contains projected embeddings, paired high-dimensional displacement,
forecast errors, identifiers, and provenance metadata. Raw 1280-dimensional
embeddings are used in memory, not exported. No plots or folders are created.
Existing outputs require --overwrite; incomplete/mismatched checkpoints are refused.
Without --run it only prints usage; no imports that load models or data.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAYERS = (1, 5, 10, 15, 20)
SEED = 42


def sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_inputs(per_station, seed):
    import pandas as pd
    import pyarrow.parquet as pq

    cases = sibling("embedding_cases", "5.1.persistence_hurdle.py")
    helper = cases.input_module()
    rng = np.random.default_rng(seed)
    frames, histories, targets, clears, provenance = [], [], [], [], []
    for station, year in cases.HOLDOUTS:
        examples, metadata = helper.load_eval_examples(
            stations=[station], years=[year], max_origins=0)
        times = examples["origin_utc_minute"].astype(np.int64)
        random_ids = np.sort(rng.choice(len(times), min(per_station, len(times)), replace=False))
        pinned = {}
        for panel, (s, y, _, timestamp) in enumerate(cases.SELECTED_ORIGINS):
            if (s, y) == (station, year):
                minute = pd.Timestamp(timestamp).value // 60_000_000_000
                hits = np.flatnonzero(times == minute)
                if len(hits) != 1:
                    raise ValueError(f"Missing/duplicate illustration: {station} {timestamp}")
                pinned[int(hits[0])] = chr(ord("a") + panel)
        ids = np.union1d(random_ids, list(pinned)).astype(np.int64)
        rows = examples["origin_row"][ids].astype(np.int64)
        source = pq.read_table(metadata["sources"][0]["path"], columns=["ghi_clear_centi"])
        clear = source["ghi_clear_centi"].to_numpy().astype(np.float64) / 100
        target_clear = clear[rows[:, None] + np.arange(1, 16)]
        h, t = examples["history"][ids], examples["target"][ids]
        # Work in stored integer precision at the inclusive +/-0.25 boundary.
        change = np.median(np.rint(t[:, :5] * 1000), axis=1) - np.rint(h[:, -1] * 1000)
        group = np.where(change >= 250, "Upward ramps",
                         np.where(change <= -250, "Downward ramps", "Other"))
        frames.append(pd.DataFrame(dict(
            station=station, year=year, origin_utc_minute=times[ids], origin_row=rows,
            random_sample=np.isin(ids, random_ids),
            illustration=[pinned.get(int(i), "") for i in ids], group=group,
            observed_kappa_change=change / 1000)))
        histories.append(h)
        targets.append(t)
        clears.append(target_clear)
        provenance.extend(metadata["sources"])
    return (pd.concat(frames, ignore_index=True), np.concatenate(histories),
            np.concatenate(targets), np.concatenate(clears), provenance)


def extract(model, training, histories, length, layers, batch_size, device):
    """Hook both native forecast branches, retaining only the final context token."""
    import torch

    model.eval()
    if any(layer < 1 or layer > len(model.stacked_xf) for layer in layers):
        raise ValueError("Requested layer is outside the transformer stack")
    stages = (0, *layers)
    captured = {stage: [] for stage in stages}
    collected = {(stage, branch): [] for stage in stages for branch in ("positive", "negative")}

    def hook(stage, block):
        def capture(_module, _inputs, output):
            values = output[0] if block else output
            captured[stage].append(values[:, -1].detach().float().cpu().numpy().copy())
        return capture

    handles = [model.tokenizer.register_forward_hook(hook(0, False))]
    handles.extend(model.stacked_xf[layer - 1].register_forward_hook(hook(layer, True))
                   for layer in layers)
    predictions = []
    try:
        with torch.inference_mode():
            for start in range(0, len(histories), batch_size):
                for values in captured.values():
                    values.clear()
                values, masks = training.pad_batch_context(
                    histories[start:start + batch_size, -length:], length, device)
                forecast = training.forecast_with_grad(model, 15, values, masks)
                predictions.append(forecast.detach().float().cpu().numpy())
                for stage in stages:
                    if len(captured[stage]) != 2:
                        raise ValueError("Expected exactly positive and negative prefill passes")
                    for branch, embedding in zip(("positive", "negative"), captured[stage]):
                        if not np.isfinite(embedding).all():
                            raise ValueError("Nonfinite hidden embedding")
                        collected[stage, branch].append(embedding)
                print(f"  {min(start + batch_size, len(histories))}/{len(histories)} origins", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    forecast = np.concatenate(predictions)
    if forecast.shape != (len(histories), 15) or not np.isfinite(forecast).all():
        raise ValueError("Invalid forecast shape/values")
    return {key: np.concatenate(value) for key, value in collected.items()}, forecast


def project(frame, pretrained, finetuned):
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fit_rows = frame.random_sample.to_numpy(dtype=bool)
    if fit_rows.sum() < 2:
        raise ValueError("PCA needs at least two sampled origins")
    result = []
    for (stage, branch), before in pretrained.items():
        after = finetuned[stage, branch]
        if before.shape != after.shape:
            raise ValueError("Paired embedding shapes differ")
        scaler = StandardScaler().fit(np.concatenate((before[fit_rows], after[fit_rows])))
        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(scaler.transform(np.concatenate((before[fit_rows], after[fit_rows]))))
        delta = np.linalg.norm(after - before, axis=1)
        norm = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
        cosine = np.full(len(frame), np.nan)
        np.divide(np.sum(before * after, axis=1), norm, out=cosine, where=norm > 0)
        for method, embeddings in (("TimesFM", before), ("TimesFM-BSRN", after)):
            coordinates = pca.transform(scaler.transform(embeddings))
            part = frame.copy()
            part["model"], part["layer"], part["branch"] = method, stage, branch
            part["pc1"], part["pc2"] = coordinates.T
            part["pc1_variance_ratio"], part["pc2_variance_ratio"] = pca.explained_variance_ratio_
            part["paired_embedding_l2"] = delta
            part["paired_embedding_cosine_similarity"] = cosine
            result.append(part)
    return pd.concat(result, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Explicitly enable model inference and saving")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "Data/models/timesfm-2.5-full-L60")
    parser.add_argument("--output", type=Path, help="New .parquet file in an existing directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace the existing analysis after computation")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--per-station", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    if args.output is None or args.output.suffix != ".parquet":
        parser.error("--run requires an explicit --output ending in .parquet")
    if (args.output.exists() and not args.overwrite) or not args.output.parent.is_dir():
        parser.error("Output parent must exist; replacing an analysis requires --overwrite")
    if args.per_station < 1 or args.batch_size < 1 or len(set(args.layers)) != len(args.layers):
        parser.error("Positive sample/batch sizes and distinct layers are required")

    loader = sibling("embedding_loader", "3.3.forecast_timesfm.py")
    training = sibling("embedding_training", "2.2.finetune_timesfm.py")
    manifest = loader.read_manifest(args.checkpoint_dir, "full")
    if (manifest["base_model"], manifest["base_revision"]) != (training.MODEL_NAME, training.MODEL_REVISION):
        raise ValueError("Checkpoint does not match the pinned pretrained model")
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq

    torch.set_num_threads(1)
    frame, histories, targets, clear, sources = sample_inputs(args.per_station, args.seed)
    length = int(manifest["history_length"])
    frame["history_length"] = length
    frame["context_kappa_std"] = histories[:, -length:].std(axis=1)
    observed_ghi = targets.astype(np.float64) * clear
    persistence = histories[:, -1, None].astype(np.float64) * clear
    frame["rmse_persistence"] = np.sqrt(np.mean((persistence - observed_ghi)**2, axis=1))
    embeddings = []
    for label in ("pretrained", "finetuned"):
        print(f"Extracting {label}: L={length}, {len(frame)} origins", flush=True)
        pipeline = (training.load_pretrained_pipeline(args.device) if label == "pretrained"
                    else loader.load_finetuned_model(args.checkpoint_dir, manifest, args.device))
        hidden, forecast = extract(pipeline.model, training, histories, length,
                                   args.layers, args.batch_size, args.device)
        embeddings.append(hidden)
        frame[f"rmse_{label}"] = np.sqrt(np.mean((forecast.astype(np.float64)*clear - observed_ghi)**2, axis=1))
        del pipeline
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
    frame["rmse_reduction_from_pretrained"] = frame.rmse_pretrained - frame.rmse_finetuned
    frame["rmse_reduction_from_persistence"] = frame.rmse_persistence - frame.rmse_finetuned
    result = project(frame, *embeddings)
    table = pa.Table.from_pandas(result, preserve_index=False)
    provenance = dict(checkpoint=str(args.checkpoint_dir.resolve()), training_manifest=manifest,
                      sources=sources, seed=args.seed, per_station=args.per_station,
                      layers=args.layers, token="last_context_patch", input_patch_length=32,
                      pca_fit="pooled models, random_sample origins only; separate stage/branch",
                      ramp_definition="median first 5 target kappa minus origin kappa >=+0.25 or <=-0.25",
                      note="Retrospective descriptive analysis; overlapping origins; no raw embedding export")
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                          b"embedding_analysis": json.dumps(provenance).encode()})
    pq.write_table(table, args.output, compression="zstd")
    if pq.read_metadata(args.output).num_rows != len(result):
        raise ValueError("Saved row count differs")
    print(f"Saved {len(result)} projected rows for {len(frame)} origins: {args.output}")


if __name__ == "__main__":
    main()
