#!/usr/bin/env python3
"""Generate classification concept HTML files and minimal .py snippets.

Interpreto 0.5.0 API. For each classification model, this script:

1. Loads a ``SplitterForClassification`` (auto-detects the head, forces
   the [CLS] granularity).
2. Loads the dataset (capped at ``NUM_SAMPLES``) and computes the [CLS]
   activations once. The tuple ``(activations, predictions)`` is cached
   under ``data/<model_id>/activations.pt``.
3. For every concept method, trains (or loads) the concept explainer,
   interprets it with ``TopKInputs``, ranks concepts with
   ``concept_output_gradient`` and writes ``<method>.html`` +
   a matching minimal ``.py`` snippet.

Set ``DEBUG_SAMPLES`` in the environment (``DEBUG_SAMPLES=1000``) to
override ``NUM_SAMPLES`` for quick iteration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset

from interpreto import SplitterForClassification, plot_concepts
from interpreto.concepts import (
    ICAConcepts,
    MpSAEConcepts,
    NeuronsAsConcepts,
    PCAConcepts,
    SemiNMFConcepts,
    SVDConcepts,
    VanillaSAEConcepts,
)
from interpreto.concepts.interpretations import TopKInputs
from interpreto.concepts.methods.overcomplete import (
    DeadNeuronsReanimationLoss,
    MSELoss,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    activations_cache_path,
    cache_activations,
    explainers_cache_dir,
    format_kwargs_lines,
    load_concept_model,
    save_concept_model,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Configuration (edit these)
# ----------------------------
model_id = "clf:emotion:bert"

MODEL_CONFIGS = {
    "clf:emotion:bert": {
        "hf_model_id": "nateraw/bert-base-uncased-emotion",
        "hf_dataset_id": "dair-ai/emotion",
        "hf_dataset_config": "split",
        "classes_names": [
            "sadness",
            "joy",
            "love",
            "anger",
            "fear",
            "surprise",
        ],
        "forward_kwargs": None,
    },
    "clf:imdb:distilbert": {
        "hf_model_id": "lvwerra/distilbert-imdb",
        "hf_dataset_id": "stanfordnlp/imdb",
        "hf_dataset_config": None,
        "classes_names": [
            "negative",
            "positive",
        ],
        # IMDB reviews are long; DistilBERT accepts up to 512 tokens.
        "forward_kwargs": {"truncation": True},
    },
    "clf:ag-news:roberta": {
        "hf_model_id": "arman1o1/roberta_ag_news_model",
        "hf_dataset_id": "fancyzhx/ag_news",
        "hf_dataset_config": None,
        "classes_names": [
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        ],
        "forward_kwargs": None,
    },
}

DATASET_SPLIT = "train"
NUM_SAMPLES = 200_000  # cap; will be truncated if the dataset is smaller
SEED = 0

NB_CONCEPTS = 30
TOP_K = 10
TOPK_WORDS = 5

BATCH_SIZE = 64
GRADIENT_BATCH_SIZE = 64

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "explanations"

METHODS = {
    "ica": ICAConcepts,
    "mp_sae": MpSAEConcepts,
    "neurons_as_concepts": NeuronsAsConcepts,
    "pca": PCAConcepts,
    "semi_nmf": SemiNMFConcepts,
    "svd": SVDConcepts,
    "vanilla_sae": VanillaSAEConcepts,
}

DEFAULT_INIT_PARAMETERS = {"nb_concepts": NB_CONCEPTS, "device": device}
INIT_PARAMETERS: dict[str, dict] = {k: DEFAULT_INIT_PARAMETERS.copy() for k in METHODS}
INIT_PARAMETERS["neurons_as_concepts"] = {}

SAES_FIT_PARAMETERS = {
    "criterion": DeadNeuronsReanimationLoss,
    "optimizer_class": torch.optim.Adam,
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {"T_max": 20, "eta_min": 1e-6},
    "lr": 1e-3,
    "nb_epochs": 30,
    "batch_size": 32 * BATCH_SIZE,
    "monitoring": 0,
}
FIT_PARAMETERS: dict[str, dict] = {
    k: SAES_FIT_PARAMETERS.copy() for k in METHODS if "sae" in k
}
FIT_PARAMETERS["ica"] = {"max_iter": 5000}
FIT_PARAMETERS["mp_sae"]["criterion"] = MSELoss


# ----------------------------------------------------------------------
# Snippet rendering
# ----------------------------------------------------------------------


def render_code_snippet(
    method_name: str,
    explainer_cls: type,
    hf_model_id: str,
    hf_dataset_id: str,
    hf_dataset_config: str | None,
    classes_names: list[str],
    init_params: dict,
    fit_params: dict,
    count_min_threshold: int,
    forward_kwargs: dict | None,
) -> str:
    """Return a self-contained snippet reproducing one HTML file."""
    init_lines, init_imports = format_kwargs_lines(init_params, indent="    ")
    fit_lines, fit_imports = format_kwargs_lines(fit_params, indent="    ")
    extra_imports = sorted(init_imports | fit_imports)

    lines: list[str] = [
        "import torch",
        "from datasets import load_dataset",
        "from interpreto import SplitterForClassification, plot_concepts",
        f"from interpreto.concepts import {explainer_cls.__name__}",
        "from interpreto.concepts.interpretations import TopKInputs",
    ]
    if extra_imports:
        lines.append(
            "from interpreto.concepts.methods.overcomplete import "
            + ", ".join(extra_imports)
        )
    lines.append("")
    lines.append('device = "cuda" if torch.cuda.is_available() else "cpu"')
    lines.append("")
    lines.append(
        f"splitter = SplitterForClassification({hf_model_id!r}, "
        f"batch_size={BATCH_SIZE}, device_map=device)"
    )

    if hf_dataset_config is None:
        load_call = f"load_dataset({hf_dataset_id!r})"
    else:
        load_call = f"load_dataset({hf_dataset_id!r}, {hf_dataset_config!r})"
    lines.append(f'inputs = {load_call}[{DATASET_SPLIT!r}]["text"]')
    lines.append(f"classes_names = {classes_names!r}")
    lines.append("")

    if forward_kwargs:
        lines.append(
            f"activations, _ = splitter.get_activations(inputs, forward_kwargs={forward_kwargs!r})"
        )
    else:
        lines.append("activations, _ = splitter.get_activations(inputs)")
    lines.append("")

    lines.append(f"concept_explainer = {explainer_cls.__name__}(")
    lines.append("    splitter,")
    lines.extend(init_lines)
    lines.append(")")

    if explainer_cls is not NeuronsAsConcepts:
        lines.append("")
        if fit_lines:
            lines.append("concept_explainer.fit(")
            lines.append("    activations,")
            lines.extend(fit_lines)
            lines.append(")")
        else:
            lines.append("concept_explainer.fit(activations)")

    lines.append("")
    lines.append("topk = TopKInputs(")
    lines.append("    concept_explainer=concept_explainer,")
    lines.append(f"    k={TOPK_WORDS},")
    lines.append("    use_unique_words=3,")
    lines.append(
        f'    unique_words_kwargs={{"count_min_threshold": {count_min_threshold}, "lemmatize": True}},'
    )
    lines.append(")")
    lines.append(
        'labels = {k: list(v.keys()) for k, v in topk.interpret(inputs=inputs, concepts_indices="all").items()}'
    )
    lines.append("")
    lines.append("gradients = concept_explainer.concept_output_gradient(")
    lines.append("    inputs=activations,")
    lines.append("    targets=None,")
    lines.append(f"    batch_size={GRADIENT_BATCH_SIZE},")
    lines.append(")")
    lines.append("mean_gradients = torch.stack(gradients).abs().squeeze().mean(0)")
    lines.append("")
    lines.append("plot_concepts(")
    lines.append("    classes_names=classes_names,")
    lines.append("    concepts_importances=mean_gradients,")
    lines.append("    concepts_labels=labels,")
    lines.append(f"    top_k={TOP_K},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def load_inputs(config: dict) -> list[str]:
    num_samples = int(os.environ.get("DEBUG_SAMPLES", NUM_SAMPLES))
    dataset_kwargs = (
        {"name": config["hf_dataset_config"]}
        if config["hf_dataset_config"] is not None
        else {}
    )
    dataset = load_dataset(config["hf_dataset_id"], **dataset_kwargs)[DATASET_SPLIT]
    dataset = dataset.shuffle(seed=SEED)
    n = min(num_samples, len(dataset))
    print(f"Using {n} inputs (out of {len(dataset)} in {DATASET_SPLIT} split)")
    return list(dataset.select(range(n))["text"])


def main() -> None:
    config = MODEL_CONFIGS[model_id]
    classes_names = config["classes_names"]

    torch.manual_seed(SEED)

    output_root = OUTPUT_ROOT / model_id / "concept" / "general"
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Splitter + inputs + cached activations
    # ------------------------------------------------------------------
    splitter = SplitterForClassification(
        config["hf_model_id"],
        batch_size=BATCH_SIZE,
        device_map=device,
    )
    inputs = load_inputs(config)
    count_min_threshold = max(1, round(len(inputs) * 0.002))

    activations, _predictions = cache_activations(
        splitter,
        inputs,
        activations_cache_path(model_id),
        tqdm_bar=True,
        forward_kwargs=config["forward_kwargs"] or {},
    )

    # ------------------------------------------------------------------
    # 2. Iterate over concept methods
    # ------------------------------------------------------------------
    for method_name, explainer_cls in METHODS.items():
        html_path = output_root / f"{method_name}.html"
        code_path = html_path.with_suffix(".py")
        init_params = INIT_PARAMETERS.get(method_name, {})
        fit_params = FIT_PARAMETERS.get(method_name, {})

        print(f"\n== {method_name}")
        concept_explainer = explainer_cls(splitter, **init_params)

        explainer_path = explainers_cache_dir(model_id) / f"{method_name}.pt"
        if explainer_cls is NeuronsAsConcepts:
            pass  # nothing to fit
        elif explainer_path.exists():
            print(f"   loading fitted explainer from {explainer_path}")
            load_concept_model(concept_explainer, explainer_path, device)
        else:
            print(f"   fitting on {activations.shape[0]} activations")
            concept_explainer.fit(activations, **fit_params)
            try:
                save_concept_model(concept_explainer, explainer_path)
                print(f"   saved to {explainer_path}")
            except NotImplementedError as exc:
                print(f"   (skipped save: {exc})")

        # ---- interpretation ------------------------------------------
        topk = TopKInputs(
            concept_explainer=concept_explainer,
            k=TOPK_WORDS,
            use_unique_words=3,
            unique_words_kwargs={
                "count_min_threshold": count_min_threshold,
                "lemmatize": True,
            },
        )
        topk_words = topk.interpret(inputs=inputs, concepts_indices="all")
        labels = {k: list(v.keys()) for k, v in topk_words.items() if v}

        # ---- importance ---------------------------------------------
        gradients = concept_explainer.concept_output_gradient(
            inputs=activations,
            targets=None,
            batch_size=GRADIENT_BATCH_SIZE,
        )
        mean_gradients = torch.stack(gradients).abs().squeeze().mean(0)

        # ---- HTML ---------------------------------------------------
        print(f"   saving {html_path}")
        plot_concepts(
            classes_names=classes_names,
            concepts_importances=mean_gradients,
            concepts_labels=labels,
            top_k=TOP_K,
            save_path=str(html_path),
        )

        # ---- snippet ------------------------------------------------
        code_path.write_text(
            render_code_snippet(
                method_name=method_name,
                explainer_cls=explainer_cls,
                hf_model_id=config["hf_model_id"],
                hf_dataset_id=config["hf_dataset_id"],
                hf_dataset_config=config["hf_dataset_config"],
                classes_names=classes_names,
                init_params=init_params,
                fit_params=fit_params,
                count_min_threshold=count_min_threshold,
                forward_kwargs=config["forward_kwargs"],
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
