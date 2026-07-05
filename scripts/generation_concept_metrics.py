#!/usr/bin/env python3
"""Compute reconstruction / sparsity metrics for generation concept explainers.

Interpreto 0.5.0 API. For each generation model and each concept method
wired into ``generation_concepts.py``, this script:

1. Rebuilds the ``SplitterForGeneration`` at the same split point (the
   base LLM is loaded because the splitter's ``get_latent_shape`` runs
   a real trace).
2. Loads the cached activations from ``data/<model_id>/activations.pt``.
   Activations are float16 on disk and cast to float32 in memory.
3. Deterministically splits them 90 % / 10 %; the 10 % eval slice is
   what the metrics score.
4. For each concept method: rebuilds the explainer with the same init
   kwargs used at fit time. Load path:

   * ``NeuronsAsConcepts`` — identity, no load, no fit.
   * ``VanillaSAEConcepts`` / ``MpSAEConcepts`` — restore ``state_dict``
     from ``data/<model_id>/explainers/<method>.pt``.
   * ``BatchTopKSAEConcepts`` — refit on the 90 % train slice because
     the saved ``state_dict`` lacks ``running_threshold``. Fit uses the
     same kwargs as ``generation_concepts.py``.

5. Computes MSE, FID, Sparsity, SparsityRatio on the eval slice and
   dumps the 4 scalars per method into
   ``data/<model_id>/metrics/concept/general.json``.

The manifest builder later broadcasts these to every entry that
references the same (model, method). See ``TODO_METRICS.md``.

Runtime knobs (all optional, mostly for debug runs):

* ``METRIC_MODEL_IDS`` — comma-separated model_id list (default: all).
* ``METRIC_METHODS`` — comma-separated method list (default: all).
* ``METRIC_SPLIT_RATIO`` — override the 0.1 eval ratio.
* ``METRIC_SPLIT_SEED`` — override the split seed.
* ``HF_TOKEN`` — required for gated models (Llama).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

from interpreto import SplitterForGeneration
from interpreto.concepts import BatchTopKSAEConcepts, NeuronsAsConcepts
from interpreto.concepts.metrics import FID, MSE, Sparsity, SparsityRatio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    activations_cache_path,
    explainers_cache_dir,
    load_concept_model,
    metrics_cache_path,
    save_metric_scores,
    split_activations,
)
from generation_concepts import (  # noqa: E402
    ACTIVATIONS_LOAD_DTYPE,
    METHODS,
    MODEL_CONFIGS,
    build_fit_parameters,
    build_init_parameters,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Configuration (edit these)
# ----------------------------
EVAL_RATIO = 0.1
EVAL_SEED = 0
# Cap the eval slice size so FID's Wasserstein 1D sort fits on the GPU.
# 50k activations is more than enough for a statistically meaningful
# score; larger slices bloat GPU memory (gpt2 has 6.8M token-activations
# → a 10% slice of 688k × 768 float32 = 2 GB, plus intermediate sort
# buffers that push it above 24 GB on a single card).
EVAL_MAX_ROWS = 50_000


def _env_list(var: str, default: list[str]) -> list[str]:
    raw = os.environ.get(var)
    if not raw:
        return default
    return [v.strip() for v in raw.split(",") if v.strip()]


def _env_float(var: str, default: float) -> float:
    raw = os.environ.get(var)
    return float(raw) if raw else default


def _env_int(var: str, default: int) -> int:
    raw = os.environ.get(var)
    return int(raw) if raw else default


def evaluate_model(model_id: str, eval_ratio: float, seed: int) -> None:
    config = MODEL_CONFIGS[model_id]
    hf_model_id = config["hf_model_id"]
    split_point = config["split_point"]
    batch_size = config["batch_size"]
    torch_dtype = config.get("torch_dtype")
    print(f"\n== {model_id}  (split_point={split_point}, dtype={torch_dtype})")

    torch.manual_seed(seed)

    # --------------------------------------------------------------
    # 1. Cached activations first (fail fast if missing)
    # --------------------------------------------------------------
    cache_path = activations_cache_path(model_id)
    if not cache_path.exists():
        print(f"  ! no cached activations at {cache_path}, skipping model")
        return
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    activations: torch.Tensor = payload["activations"]
    if activations.dtype != ACTIVATIONS_LOAD_DTYPE:
        activations = activations.to(ACTIVATIONS_LOAD_DTYPE)

    train_slice, eval_slice = split_activations(
        activations, ratio=eval_ratio, seed=seed
    )
    if eval_slice.shape[0] > EVAL_MAX_ROWS:
        # Deterministic sub-sample so the eval slice fits on the GPU.
        gen = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(eval_slice.shape[0], generator=gen)[:EVAL_MAX_ROWS]
        eval_slice = eval_slice[idx]
    print(
        f"  activations: total={activations.shape[0]}  "
        f"train={train_slice.shape[0]}  eval={eval_slice.shape[0]}"
    )

    # --------------------------------------------------------------
    # 2. Splitter (loads the base LLM; needed to construct concept
    #    explainers with the right latent dimension)
    # --------------------------------------------------------------
    splitter_kwargs = dict(
        split_point=split_point,
        batch_size=batch_size,
        device_map=device,
    )
    if torch_dtype is not None:
        splitter_kwargs["torch_dtype"] = torch_dtype
    splitter = SplitterForGeneration(hf_model_id, **splitter_kwargs)

    init_parameters = build_init_parameters(config)
    fit_parameters = build_fit_parameters(config)

    # --------------------------------------------------------------
    # 3. Iterate concept methods
    # --------------------------------------------------------------
    method_scores: dict[str, dict[str, float]] = {}
    method_names = _env_list("METRIC_METHODS", list(METHODS))
    eval_slice_gpu = eval_slice.to(device)

    for method_name in method_names:
        if method_name not in METHODS:
            print(f"  ! unknown method {method_name!r}, skipping")
            continue
        explainer_cls = METHODS[method_name]
        init_params = init_parameters.get(method_name, {})
        fit_params = fit_parameters.get(method_name, {})

        print(f"\n-- {method_name}")
        concept_explainer = explainer_cls(splitter, **init_params)

        explainer_path = explainers_cache_dir(model_id) / f"{method_name}.pt"
        if explainer_cls is NeuronsAsConcepts:
            pass  # identity explainer, nothing to load
        elif explainer_cls is BatchTopKSAEConcepts:
            # No usable saved state; refit on the 90% train slice.
            print(f"   refitting on {train_slice.shape[0]} activations")
            t0 = time.perf_counter()
            concept_explainer.fit(train_slice, **fit_params)
            print(f"   fit {time.perf_counter() - t0:.1f}s")
        elif explainer_path.exists():
            load_concept_model(concept_explainer, explainer_path, device)
        else:
            print(f"   ! no fitted explainer at {explainer_path}, skipping")
            continue

        t0 = time.perf_counter()
        with torch.no_grad():
            mse = MSE(concept_explainer).compute(eval_slice_gpu)
            fid = FID(concept_explainer).compute(eval_slice_gpu)
            sparsity = Sparsity(concept_explainer).compute(eval_slice_gpu)
            sparsity_ratio = SparsityRatio(concept_explainer).compute(eval_slice_gpu)
        dt = time.perf_counter() - t0

        method_scores[f"{method_name}.html"] = {
            "mse": float(mse),
            "fid": float(fid),
            "sparsity": float(sparsity),
            "sparsity_ratio": float(sparsity_ratio),
        }
        print(
            f"   mse={mse:.4f}  fid={fid:.4f}  "
            f"sparsity={sparsity:.4f}  sparsity_ratio={sparsity_ratio:.4f}  "
            f"({dt:.1f}s)"
        )

    if method_scores:
        out_path = metrics_cache_path(model_id, "concept")
        save_metric_scores(out_path, method_scores)
        print(f"\n  wrote {out_path.relative_to(Path.cwd())}")


def main() -> None:
    eval_ratio = _env_float("METRIC_SPLIT_RATIO", EVAL_RATIO)
    seed = _env_int("METRIC_SPLIT_SEED", EVAL_SEED)
    model_ids = _env_list("METRIC_MODEL_IDS", list(MODEL_CONFIGS))

    print(f"config: eval_ratio={eval_ratio}  seed={seed}  models={model_ids}")

    for model_id in model_ids:
        if model_id not in MODEL_CONFIGS:
            print(f"! unknown model_id {model_id!r}, skipping")
            continue
        evaluate_model(model_id, eval_ratio, seed)


if __name__ == "__main__":
    main()
