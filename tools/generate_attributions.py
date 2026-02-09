#!/usr/bin/env python3
"""Generate classification attribution HTML files and minimal .py snippets."""

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from interpreto import (
    IntegratedGradients,
    KernelShap,
    Lime,
    Occlusion,
    Saliency,
    SmoothGrad,
    Sobol,
    plot_attributions,
)


# ----------------------------
# Configuration (edit these)
# ----------------------------
model_id = "bert-emotion"

MODEL_CONFIGS = {
    "bert-emotion": {
        "hf_model_id": "nateraw/bert-base-uncased-emotion",
        "dataset": {
            "name": "dair-ai/emotion",
            "config": "split",
            "split": "train",
            "text_field": "text",
        },
        "classes_names": [
            # "sadness",
            # "joy",
            # "love",
            # "anger",
            # "fear",
            # "surprise",
        ],
    }
}

NUM_SAMPLES = 5

OUTPUT_ROOT = Path("assets")

METHODS = {  # TODO: add the rest of the methods
    "saliency": Saliency,
    "integrated_gradients": IntegratedGradients,
    "smoothgrad": SmoothGrad,
    "lime": Lime,
    "kernel_shap": KernelShap,
    "occlusion": Occlusion,
    "sobol": Sobol,
}


def main() -> None:
    config = MODEL_CONFIGS[model_id]
    classes_names = config["classes_names"]
    torch.manual_seed(0)

    # Load a fixed set of samples so outputs are reproducible.
    samples = load_samples(config)
    batch_inputs = [sample["input"] for sample in samples]

    # Load the classifier and reuse it across all methods.
    tokenizer = AutoTokenizer.from_pretrained(config["hf_model_id"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(config["hf_model_id"])
    model.eval()

    output_root = OUTPUT_ROOT / model_id
    output_root.mkdir(parents=True, exist_ok=True)

    for method_name, explainer_cls in METHODS.items():

        # Compute attributions for all samples in a batch.
        explainer = explainer_cls(model, tokenizer)
        attributions = explainer(model_inputs=batch_inputs)

        for sample, attribution in zip(samples, attributions):
            sample_dir = output_root / sample["id"]
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
                    sample_text=sample["input"],
                    model_hf_id=config["hf_model_id"],
                    classes_names=classes_names,
                ),
                encoding="utf-8",
            )


def load_samples(config: dict) -> list[dict[str, str]]:
    dataset_cfg = config["dataset"]
    dataset = load_dataset(
        dataset_cfg["name"], dataset_cfg.get("config"), split=dataset_cfg["split"]
    )
    # Shuffle with a fixed seed, then take the first N samples.
    dataset = dataset.shuffle(seed=SEED)
    total = min(NUM_SAMPLES, len(dataset))
    selected = dataset.select(list(range(total)))
    inputs = selected[dataset_cfg["text_field"]]

    samples = []
    for idx, text in enumerate(inputs, start=1):
        samples.append({"id": f"sample-{idx:03d}", "input": text})
    return samples


def render_code_snippet(
    explainer_cls: type,
    sample_text: str,
    model_hf_id: str,
    classes_names: list[str],
) -> str:
    return f"""from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import {explainer_cls.__name__}, plot_attributions

model_id = {model_hf_id!r}
classes_names = {classes_names!r}

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = {explainer_cls.__name__}(model, tokenizer)

attributions = explainer(model_inputs={sample_text!r})
plot_attributions(attributions[0], classes_names=classes_names)
"""


if __name__ == "__main__":
    main()
