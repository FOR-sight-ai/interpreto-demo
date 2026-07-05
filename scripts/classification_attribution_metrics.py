#!/usr/bin/env python3
"""Compute Insertion / Deletion metric scores for classification attributions.

Interpreto 0.5.0 API. For each classification model and each attribution
method wired into ``classification_attributions.py``, this script:

1. Loads the model + tokenizer once.
2. Draws a **fresh** deterministic slice of ``NUM_EVAL_SAMPLES`` inputs
   from the model's test dataset (disjoint from the 10 shown samples via
   a different RNG seed).
3. Runs the attribution method on those inputs **without specifying
   targets**, so Insertion/Deletion score the explanation of the
   predicted class only.
4. Evaluates :class:`interpreto.attributions.metrics.Insertion` and
   :class:`interpreto.attributions.metrics.Deletion` on the resulting
   attributions and stores the aggregate AUC scores in
   ``data/<model_id>/metrics/attribution/<scope>.json`` for both
   scopes (``all-classes`` and ``single-class``). Both scopes share the
   same score because the metric is target-agnostic.

The manifest builder later broadcasts each scope's scores to every
sample entry under that (model, method, scope). See
``TODO_METRICS.md`` for the full flow.

Runtime knobs (all optional, mostly for debug runs):

* ``METRIC_MODEL_IDS`` — comma-separated model_id list (default: all).
* ``METRIC_METHODS`` — comma-separated method list (default: all).
* ``METRIC_NUM_SAMPLES`` — override ``NUM_EVAL_SAMPLES`` (int).
* ``METRIC_N_PERTURBATIONS`` — override ``N_PERTURBATIONS`` (int).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from interpreto import Granularity
from interpreto.attributions.metrics import Deletion, Insertion

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import metrics_cache_path, save_metric_scores  # noqa: E402
from classification_attributions import MODEL_CONFIGS, METHODS  # noqa: E402


# ----------------------------
# Configuration (edit these)
# ----------------------------
NUM_EVAL_SAMPLES = 100
EVAL_SEED = 1  # disjoint from classification_attributions.SEED = 0
N_PERTURBATIONS = 50
BATCH_SIZE = 4


def _env_list(var: str, default: list[str]) -> list[str]:
    raw = os.environ.get(var)
    if not raw:
        return default
    return [v.strip() for v in raw.split(",") if v.strip()]


def _env_int(var: str, default: int) -> int:
    raw = os.environ.get(var)
    return int(raw) if raw else default


def filter_by_granularity(
    inputs: list[str],
    tokenizer,
    granularity: Granularity,
    min_elements: int = 2,
) -> list[str]:
    """Drop inputs that yield fewer than ``min_elements`` granular units.

    Insertion / Deletion perturbations refuse to run on a sequence with
    a single granular element (a review that is exactly one sentence at
    ``Granularity.SENTENCE``). We match interpreto's own tokenization to
    decide which inputs are viable.
    """
    kept: list[str] = []
    dropped = 0
    for text in inputs:
        encoded = tokenizer(
            [text],
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
        )
        indices = granularity.get_indices(encoded, tokenizer)
        if len(indices[0]) >= min_elements:
            kept.append(text)
        else:
            dropped += 1
    if dropped:
        print(f"  filtered out {dropped} single-{granularity.name.lower()} inputs")
    return kept


def evaluate_model(model_id: str, num_samples: int, n_perturbations: int) -> None:
    config = MODEL_CONFIGS[model_id]
    granularity: Granularity = config["granularity"]
    hf_model_id: str = config["hf_model_id"]

    torch.manual_seed(EVAL_SEED)

    # Draw a slightly larger slice than requested; some inputs may be
    # filtered out for having only a single granular element (see
    # ``filter_by_granularity``). We over-fetch to try and land at
    # ``num_samples`` after filtering.
    over_fetch = max(num_samples * 2, num_samples + 50)
    dataset = load_dataset(config["hf_dataset_id"])["test"].shuffle(seed=EVAL_SEED)
    raw_inputs = list(
        dataset.select(list(range(min(over_fetch, len(dataset)))))["text"]
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
    model.eval()

    batch_inputs = filter_by_granularity(raw_inputs, tokenizer, granularity)[
        :num_samples
    ]
    if len(batch_inputs) < num_samples:
        print(
            f"  ! only {len(batch_inputs)} usable inputs for {num_samples} requested"
        )
    print(f"  evaluating on {len(batch_inputs)} samples")

    insertion = Insertion(
        model, tokenizer, n_perturbations=n_perturbations, batch_size=BATCH_SIZE
    )
    deletion = Deletion(
        model, tokenizer, n_perturbations=n_perturbations, batch_size=BATCH_SIZE
    )

    # Both scopes share the same score: metrics only evaluate attributions
    # against the predicted class (targets=None), regardless of whether the
    # UI shows the all-classes or single-class HTML.
    method_scores: dict[str, dict[str, float]] = {}

    method_names = _env_list("METRIC_METHODS", list(METHODS))

    for method_name in method_names:
        if method_name not in METHODS:
            print(f"  ! unknown method {method_name!r}, skipping")
            continue
        explainer_cls = METHODS[method_name]
        print(f"\n== {model_id} / {method_name}")

        explainer = explainer_cls(
            model, tokenizer, granularity=granularity, batch_size=BATCH_SIZE
        )

        t0 = time.perf_counter()
        attributions = explainer(model_inputs=batch_inputs)
        t_expl = time.perf_counter() - t0

        t0 = time.perf_counter()
        ins_auc, _ = insertion.evaluate(attributions)
        del_auc, _ = deletion.evaluate(attributions)
        dt = time.perf_counter() - t0
        method_scores[f"{method_name}.html"] = {
            "insertion": float(ins_auc),
            "deletion": float(del_auc),
        }
        print(
            f"  insertion={ins_auc:.4f}  deletion={del_auc:.4f}"
            f"  (explanations {t_expl:.1f}s, metric {dt:.1f}s)"
        )

    if not method_scores:
        return

    # Write the same scores into both scope files (the manifest builder
    # will broadcast to every sample entry in each scope).
    for scope in ("all-classes", "single-class"):
        cache_path = metrics_cache_path(model_id, "attribution", scope=scope)
        save_metric_scores(cache_path, method_scores)
        print(f"\n  wrote {cache_path.relative_to(Path.cwd())}")


def main() -> None:
    num_samples = _env_int("METRIC_NUM_SAMPLES", NUM_EVAL_SAMPLES)
    n_perturbations = _env_int("METRIC_N_PERTURBATIONS", N_PERTURBATIONS)
    model_ids = _env_list("METRIC_MODEL_IDS", list(MODEL_CONFIGS))

    print(
        f"config: num_samples={num_samples}  n_perturbations={n_perturbations}  "
        f"models={model_ids}"
    )

    for model_id in model_ids:
        if model_id not in MODEL_CONFIGS:
            print(f"! unknown model_id {model_id!r}, skipping")
            continue
        evaluate_model(model_id, num_samples, n_perturbations)


if __name__ == "__main__":
    main()
