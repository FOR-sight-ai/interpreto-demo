#!/usr/bin/env python3
"""Generate classification attribution HTML files and minimal .py snippets."""

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from interpreto import (
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    Occlusion,
    Saliency,
    SmoothGrad,
    SquareGrad,
    Sobol,
    VarGrad,
    plot_attributions,
)


# ----------------------------
# Configuration (edit these)
# ----------------------------
model_id = "clf:emotion:bert"  # "clf:ag-news:roberta"  # "clf:imdb:distilbert"  # "clf:emotion:bert"

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
    },
    "clf:imdb:distilbert": {
        "hf_model_id": "lvwerra/distilbert-imdb",
        "hf_dataset_id": "stanfordnlp/imdb",
        "classes_names": [
            "negative",
            "positive",
        ],
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
    },
}

NUM_SAMPLES = 10
SEED = 0

OUTPUT_ROOT = Path("explanations")

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
    model_hf_id: str,
    classes_names: list[str],
    scope: str,
) -> str:
    if scope == "all-classes":
        targets = "torch.arange(len(classes_names))"
    else:
        targets = "None"

    return f"""import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import {explainer_cls.__name__}, plot_attributions

model_id = {model_hf_id!r}
classes_names = {classes_names!r}

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = {explainer_cls.__name__}(model, tokenizer)

attributions = explainer(
    model_inputs={sample_text!r},
    targets={targets}
)
plot_attributions(attributions[0], classes_names=classes_names)
"""


def plot_and_snippet_save(
    scope,
    output_root,
    i,
    attribution,
    method_name,
    classes_names,
    explainer_cls,
    sample,
    config,
):
    if scope == "all-classes":
        sample_dir = output_root / "all-classes" / f"sample-{i:03d}"
    else:
        sample_dir = output_root / "single-class" / f"sample-{i:03d}"

    sample_dir.mkdir(parents=True, exist_ok=True)
    html_path = sample_dir / f"{method_name}.html"

    # Plot the attributions to html.
    plot_attributions(
        attribution,
        classes_names=classes_names,
        save_path=str(html_path),
    )

    # Write a code snippet for the attributions.
    code_path = html_path.with_suffix(".py")
    code_path.write_text(
        render_code_snippet(
            explainer_cls=explainer_cls,
            sample_text=sample,
            model_hf_id=config["hf_model_id"],
            classes_names=classes_names,
            scope=scope,
        ),
        encoding="utf-8",
    )


def main() -> None:
    config = MODEL_CONFIGS[model_id]
    classes_names = config["classes_names"]
    torch.manual_seed(0)

    # Load a fixed set of samples so outputs are reproducible.
    dataset = load_dataset(config["hf_dataset_id"])["test"].shuffle(seed=SEED)
    batch_inputs = list(dataset.select(list(range(NUM_SAMPLES)))["text"])
    all_targets = (
        torch.arange(len(classes_names)).view(1, -1).repeat((len(batch_inputs), 1))
    )

    # Load the classifier and reuse it across all methods.
    tokenizer = AutoTokenizer.from_pretrained(config["hf_model_id"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config["hf_model_id"])
    model.eval()

    output_root = OUTPUT_ROOT / model_id / "attribution"

    for method_name, explainer_cls in METHODS.items():
        # Compute attributions for all samples in a batch.
        explainer = explainer_cls(model, tokenizer)
        all_attributions = explainer(model_inputs=batch_inputs, targets=all_targets)
        single_attributions = explainer(model_inputs=batch_inputs)

        for i, (sample, aa, sa) in enumerate(
            zip(batch_inputs, all_attributions, single_attributions)
        ):
            plot_and_snippet_save(
                scope="all-classes",
                output_root=output_root,
                i=i,
                attribution=aa,
                method_name=method_name,
                classes_names=classes_names,
                explainer_cls=explainer_cls,
                sample=sample,
                config=config,
            )
            plot_and_snippet_save(
                scope="single-class",
                output_root=output_root,
                i=i,
                attribution=sa,
                method_name=method_name,
                classes_names=classes_names,
                explainer_cls=explainer_cls,
                sample=sample,
                config=config,
            )


if __name__ == "__main__":
    main()
