#!/usr/bin/env python3
"""Compute reconstruction / sparsity metrics for classification concept explainers.

Interpreto 0.5.0 API. For each classification model and each concept
method wired into ``classification_concepts.py``, this script:

1. Rebuilds the ``SplitterForClassification`` (needed to instantiate the
   concept explainer with the correct latent dimension).
2. Loads the cached activations from ``data/<model_id>/activations.pt``.
3. Deterministically splits them 90 % / 10 %; the 10 % eval slice is
   what the metrics score.
4. For each concept method: rebuilds the explainer with the same init
   kwargs used at fit time, restores the fitted ``concept_model`` from
   ``data/<model_id>/explainers/<method>.pt`` (skipping methods that
   have no saved weights, e.g. ``NeuronsAsConcepts``), and computes:

   * :class:`interpreto.concepts.metrics.MSE`
   * :class:`interpreto.concepts.metrics.FID`
   * :class:`interpreto.concepts.metrics.Sparsity`
   * :class:`interpreto.concepts.metrics.SparsityRatio`

5. Dumps the four scalars per method into
   ``data/<model_id>/metrics/concept/general.json``.

The manifest builder later broadcasts these to every entry that
references the same (model, method). See ``TODO_METRICS.md``.

Runtime knobs (all optional, mostly for debug runs):

* ``METRIC_MODEL_IDS`` — comma-separated model_id list (default: all).
* ``METRIC_METHODS`` — comma-separated method list (default: all).
* ``METRIC_SPLIT_RATIO`` — override the 0.1 eval ratio.
* ``METRIC_SPLIT_SEED`` — override the split seed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

from interpreto import SplitterForClassification
from interpreto.concepts import NeuronsAsConcepts
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
from classification_concepts import (  # noqa: E402
    BATCH_SIZE,
    FIT_PARAMETERS,
    INIT_PARAMETERS,
    METHODS,
    MODEL_CONFIGS,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Configuration (edit these)
# ----------------------------
EVAL_RATIO = 0.1
EVAL_SEED = 0


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
    print(f"\n== {model_id}")

    torch.manual_seed(seed)

    # --------------------------------------------------------------
    # 1. Splitter (needed to build concept explainers with the
    #    correct latent dimension).
    # --------------------------------------------------------------
    splitter = SplitterForClassification(
        config["hf_model_id"],
        batch_size=BATCH_SIZE,
        device_map=device,
    )

    # --------------------------------------------------------------
    # 2. Cached activations + 90/10 split
    # --------------------------------------------------------------
    cache_path = activations_cache_path(model_id)
    if not cache_path.exists():
        print(f"  ! no cached activations at {cache_path}, skipping model")
        return
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    activations: torch.Tensor = payload["activations"]
    if activations.dtype != torch.float32:
        activations = activations.to(torch.float32)

    train_slice, eval_slice = split_activations(
        activations, ratio=eval_ratio, seed=seed
    )
    print(
        f"  activations: total={activations.shape[0]}  "
        f"train={train_slice.shape[0]}  eval={eval_slice.shape[0]}"
    )

    eval_slice = eval_slice.to(device)

    # --------------------------------------------------------------
    # 3. Iterate concept methods
    # --------------------------------------------------------------
    method_scores: dict[str, dict[str, float]] = {}
    method_names = _env_list("METRIC_METHODS", list(METHODS))

    for method_name in method_names:
        if method_name not in METHODS:
            print(f"  ! unknown method {method_name!r}, skipping")
            continue
        explainer_cls = METHODS[method_name]

        init_params = INIT_PARAMETERS.get(method_name, {})
        print(f"\n-- {method_name}")

        concept_explainer = explainer_cls(splitter, **init_params)
        explainer_path = explainers_cache_dir(model_id) / f"{method_name}.pt"

        if explainer_cls is NeuronsAsConcepts:
            pass  # identity explainer, nothing to load
        elif explainer_path.exists():
            load_concept_model(concept_explainer, explainer_path, device)
        else:
            print(f"   ! no fitted explainer at {explainer_path}, skipping")
            continue

        t0 = time.perf_counter()
        with torch.no_grad():
            mse = MSE(concept_explainer).compute(eval_slice)
            fid = FID(concept_explainer).compute(eval_slice)
            sparsity = Sparsity(concept_explainer).compute(eval_slice)
            sparsity_ratio = SparsityRatio(concept_explainer).compute(eval_slice)
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
