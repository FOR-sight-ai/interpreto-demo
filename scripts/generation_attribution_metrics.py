#!/usr/bin/env python3
"""Compute Insertion / Deletion metric scores for generation attributions.

Interpreto 0.5.0 API. For each generation model and each attribution
method wired into ``generation_attributions.py``, this script:

1. Loads the model + tokenizer once.
2. Draws a **fresh** deterministic slice of ``NUM_EVAL_SAMPLES`` prompts
   from ``wikimedia/wikipedia`` config ``20231101.en`` (same corpus
   used by ``generation_concepts.py``). Each Wikipedia article is
   split into ``PROMPT_TOKENS`` input tokens + ``TARGET_TOKENS`` target
   tokens for a natural-continuation attribution task.
3. Runs the attribution method to produce ``AttributionOutput`` objects.
4. Evaluates :class:`interpreto.attributions.metrics.Insertion` and
   :class:`interpreto.attributions.metrics.Deletion` on the resulting
   attributions and stores the aggregate AUC scores in
   ``data/<model_id>/metrics/attribution/general.json``.

The manifest builder later broadcasts these scores to every gen
attribution sample entry. See ``TODO_METRICS.md``.

Runtime knobs (all optional, mostly for debug runs):

* ``METRIC_MODEL_IDS`` — comma-separated model_id list (default: all).
* ``METRIC_METHODS`` — comma-separated method list (default: all).
* ``METRIC_NUM_SAMPLES`` — override ``NUM_EVAL_SAMPLES`` (int).
* ``METRIC_N_PERTURBATIONS`` — override the model's ``n_perturbations``.
* ``HF_TOKEN`` — required for gated models (Llama).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from interpreto.attributions.metrics import Deletion, Insertion

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import metrics_cache_path, save_metric_scores  # noqa: E402
from generation_attributions import MODEL_CONFIGS, METHODS  # noqa: E402


# ----------------------------
# Configuration (edit these)
# ----------------------------
NUM_EVAL_SAMPLES = 100
EVAL_SEED = 1  # disjoint from generation_attributions.SEED = 0
BATCH_SIZE = 1
PROMPT_TOKENS = 32
TARGET_TOKENS = 16

DATASET_HF_ID = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
DATASET_SPLIT = "train"

# Per-model n_perturbations override — Llama is by far the most
# expensive per forward pass, so we cut it down.
N_PERTURBATIONS_PER_MODEL: dict[str, int] = {
    "gen:gpt2": 50,
    "gen:qwen3-0.6b": 50,
    "gen:llama3.1-8b": 20,
}


def _env_list(var: str, default: list[str]) -> list[str]:
    raw = os.environ.get(var)
    if not raw:
        return default
    return [v.strip() for v in raw.split(",") if v.strip()]


def _env_int(var: str, default: int) -> int:
    raw = os.environ.get(var)
    return int(raw) if raw else default


def build_eval_samples(
    tokenizer,
    num_samples: int,
    prompt_tokens: int,
    target_tokens: int,
) -> tuple[list[str], list[str]]:
    """Build (inputs, targets) pairs from Wikipedia articles.

    Each article is tokenized, truncated to ``prompt_tokens + target_tokens``
    tokens, split into a prompt slice and a target slice, and both slices
    are decoded back to strings. Articles that produce fewer than
    ``prompt_tokens + target_tokens`` tokens are skipped.
    """
    dataset = load_dataset(DATASET_HF_ID, DATASET_CONFIG)[DATASET_SPLIT]
    dataset = dataset.shuffle(seed=EVAL_SEED)

    inputs: list[str] = []
    targets: list[str] = []
    total = prompt_tokens + target_tokens
    # Walk the shuffled dataset lazily until we have `num_samples` usable
    # articles. We over-read a bit to survive short articles.
    for row in dataset.select(range(min(len(dataset), num_samples * 4))):
        text = row["text"]
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=total,
        )["input_ids"]
        if len(ids) < total:
            continue
        prompt_ids = ids[:prompt_tokens]
        target_ids = ids[prompt_tokens:total]
        inputs.append(tokenizer.decode(prompt_ids, skip_special_tokens=True))
        targets.append(tokenizer.decode(target_ids, skip_special_tokens=True))
        if len(inputs) >= num_samples:
            break

    if len(inputs) < num_samples:
        print(
            f"  ! only found {len(inputs)} usable articles for {num_samples} requested"
        )
    return inputs, targets


def evaluate_model(model_id: str, num_samples: int, n_perturbations: int) -> None:
    config = MODEL_CONFIGS[model_id]
    hf_model_id: str = config["hf_model_id"]
    torch_dtype = config["torch_dtype"]
    skip_methods: set[str] = config.get("skip_methods") or set()

    print(f"\n== {model_id}  (dtype={torch_dtype}, n_perturbations={n_perturbations})")

    torch.manual_seed(EVAL_SEED)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch_dtype,
    )
    model.eval()

    inputs, targets = build_eval_samples(
        tokenizer,
        num_samples=num_samples,
        prompt_tokens=PROMPT_TOKENS,
        target_tokens=TARGET_TOKENS,
    )
    print(f"  built {len(inputs)} (input, target) pairs")

    insertion = Insertion(
        model, tokenizer, n_perturbations=n_perturbations, batch_size=BATCH_SIZE
    )
    deletion = Deletion(
        model, tokenizer, n_perturbations=n_perturbations, batch_size=BATCH_SIZE
    )

    scores: dict[str, dict[str, float]] = {}

    method_names = _env_list("METRIC_METHODS", list(METHODS))

    for method_name in method_names:
        if method_name not in METHODS:
            print(f"  ! unknown method {method_name!r}, skipping")
            continue
        if method_name in skip_methods:
            print(f"-- {method_name} (skipped for {model_id})")
            continue
        explainer_cls = METHODS[method_name]
        print(f"\n-- {method_name}")

        explainer = explainer_cls(model, tokenizer, batch_size=BATCH_SIZE)

        t0 = time.perf_counter()
        attributions = explainer(model_inputs=inputs, targets=targets)
        t_expl = time.perf_counter() - t0
        print(f"  explanations {t_expl:.1f}s")

        t0 = time.perf_counter()
        ins_auc, _ = insertion.evaluate(attributions)
        del_auc, _ = deletion.evaluate(attributions)
        dt = time.perf_counter() - t0
        scores[f"{method_name}.html"] = {
            "insertion": float(ins_auc),
            "deletion": float(del_auc),
        }
        print(
            f"  insertion={ins_auc:.4f}  deletion={del_auc:.4f}  (metric {dt:.1f}s)"
        )

    if scores:
        cache_path = metrics_cache_path(model_id, "attribution", scope="general")
        save_metric_scores(cache_path, scores)
        print(f"\n  wrote {cache_path.relative_to(Path.cwd())}")


def main() -> None:
    num_samples = _env_int("METRIC_NUM_SAMPLES", NUM_EVAL_SAMPLES)
    model_ids = _env_list("METRIC_MODEL_IDS", list(MODEL_CONFIGS))

    print(f"config: num_samples={num_samples}  models={model_ids}")

    for model_id in model_ids:
        if model_id not in MODEL_CONFIGS:
            print(f"! unknown model_id {model_id!r}, skipping")
            continue
        n_perturbations = _env_int(
            "METRIC_N_PERTURBATIONS", N_PERTURBATIONS_PER_MODEL[model_id]
        )
        evaluate_model(model_id, num_samples, n_perturbations)


if __name__ == "__main__":
    main()
