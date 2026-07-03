#!/usr/bin/env python3
"""Generate classification attribution HTML files and minimal .py snippets.

Interpreto 0.5.0 API. For each classification model, this script:

1. Loads the model + tokenizer once.
2. Runs each attribution method on ``NUM_SAMPLES`` fixed test samples.
3. Emits, for every (sample, method) pair, two HTML files (``single-class``
   for the predicted class, ``all-classes`` for all classes) and their
   matching minimal ``.py`` snippet.
"""

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from interpreto import (
    GradientShap,
    Granularity,
    IntegratedGradients,
    KernelShap,
    Lime,
    Occlusion,
    Saliency,
    SmoothGrad,
    Sobol,
    SquareGrad,
    VarGrad,
    plot_attributions,
)


# ----------------------------
# Configuration (edit these)
# ----------------------------
model_id = "clf:emotion:bert"

MODEL_CONFIGS = {
    "clf:emotion:bert": {
        "hf_model_id": "nateraw/bert-base-uncased-emotion",
        "hf_dataset_id": "dair-ai/emotion",
        "classes_names": [
            "sadness",
            "joy",
            "love",
            "anger",
            "fear",
            "surprise",
        ],
        "granularity": Granularity.WORD,
    },
    "clf:imdb:distilbert": {
        "hf_model_id": "lvwerra/distilbert-imdb",
        "hf_dataset_id": "stanfordnlp/imdb",
        "classes_names": [
            "negative",
            "positive",
        ],
        "granularity": Granularity.SENTENCE,
    },
    "clf:ag-news:roberta": {
        "hf_model_id": "arman1o1/roberta_ag_news_model",
        "hf_dataset_id": "fancyzhx/ag_news",
        "classes_names": [
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        ],
        "granularity": Granularity.WORD,
    },
}

NUM_SAMPLES = 10
SEED = 0
BATCH_SIZE = 4

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "explanations"

METHODS = {
    "kernel_shap": KernelShap,
    "lime": Lime,
    "occlusion": Occlusion,
    "sobol": Sobol,
    "gradient_shap": GradientShap,
    "integrated_gradients": IntegratedGradients,
    "saliency": Saliency,
    "smoothgrad": SmoothGrad,
    "squared_grad": SquareGrad,
    "vargrad": VarGrad,
}


def render_code_snippet(
    explainer_cls: type,
    sample_text: str,
    scope: str,
    hf_model_id: str,
    classes_names: list[str],
    granularity: Granularity,
) -> str:
    """Return a minimal, self-contained snippet reproducing one HTML."""
    if scope == "all-classes":
        targets_line = f"    targets=torch.tensor([{list(range(len(classes_names)))!r}]),\n"
    else:
        targets_line = ""

    if granularity is Granularity.WORD:
        # Default in 0.5.0; keep the snippet minimal.
        explainer_line = f"explainer = {explainer_cls.__name__}(model, tokenizer)"
        granularity_import = ""
    else:
        explainer_line = (
            f"explainer = {explainer_cls.__name__}("
            f"model, tokenizer, granularity=Granularity.{granularity.name})"
        )
        granularity_import = ", Granularity"

    return f"""import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import {explainer_cls.__name__}{granularity_import}, plot_attributions

model_id = {hf_model_id!r}
classes_names = {classes_names!r}

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

{explainer_line}
attributions = explainer(
    model_inputs={sample_text!r},
{targets_line})

plot_attributions(attributions[0], classes_names=classes_names)
"""


def save_scope(
    scope: str,
    output_root: Path,
    i: int,
    attribution,
    method_name: str,
    classes_names: list[str],
    explainer_cls: type,
    sample: str,
    hf_model_id: str,
    granularity: Granularity,
) -> None:
    sample_dir = output_root / scope / f"sample-{i:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    html_path = sample_dir / f"{method_name}.html"
    plot_attributions(
        attribution,
        classes_names=classes_names,
        save_path=str(html_path),
    )

    code_path = html_path.with_suffix(".py")
    code_path.write_text(
        render_code_snippet(
            explainer_cls=explainer_cls,
            sample_text=sample,
            scope=scope,
            hf_model_id=hf_model_id,
            classes_names=classes_names,
            granularity=granularity,
        ),
        encoding="utf-8",
    )


def main() -> None:
    config = MODEL_CONFIGS[model_id]
    classes_names = config["classes_names"]
    granularity = config["granularity"]
    hf_model_id = config["hf_model_id"]

    torch.manual_seed(SEED)

    # Fixed sample set so outputs are reproducible.
    dataset = load_dataset(config["hf_dataset_id"])["test"].shuffle(seed=SEED)
    batch_inputs = list(dataset.select(list(range(NUM_SAMPLES)))["text"])
    all_targets = (
        torch.arange(len(classes_names))
        .view(1, -1)
        .repeat((len(batch_inputs), 1))
    )

    # Reuse the classifier across every method.
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
    model.eval()

    output_root = OUTPUT_ROOT / model_id / "attribution"

    for method_name, explainer_cls in METHODS.items():
        print(f"\n== {method_name}")
        explainer = explainer_cls(
            model, tokenizer, granularity=granularity, batch_size=BATCH_SIZE
        )
        all_attributions = explainer(model_inputs=batch_inputs, targets=all_targets)
        single_attributions = explainer(model_inputs=batch_inputs)

        for i, (sample, aa, sa) in enumerate(
            zip(batch_inputs, all_attributions, single_attributions, strict=False)
        ):
            save_scope(
                scope="all-classes",
                output_root=output_root,
                i=i,
                attribution=aa,
                method_name=method_name,
                classes_names=classes_names,
                explainer_cls=explainer_cls,
                sample=sample,
                hf_model_id=hf_model_id,
                granularity=granularity,
            )
            save_scope(
                scope="single-class",
                output_root=output_root,
                i=i,
                attribution=sa,
                method_name=method_name,
                classes_names=classes_names,
                explainer_cls=explainer_cls,
                sample=sample,
                hf_model_id=hf_model_id,
                granularity=granularity,
            )


if __name__ == "__main__":
    main()
