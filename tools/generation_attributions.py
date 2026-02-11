#!/usr/bin/env python3
"""Generate generation attribution HTML files and minimal .py snippets."""

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
model_id = "gen:qwen3-0.6b"

HF_MODEL_IDS = {
    "gen:gpt2": "gpt2",
    "gen:qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "gen:llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "gen:mistral7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
}

SAMPLES = [
    {
        "input": "Alice and Bob enter the bar, ",
        "target": "then Alice offers a drink to Bob.",
    },
    {
        "input": "We called our library Interpreto is a good name? ",
        "target": "“Interpreto” is a solid name: short, distinctive, and it strongly cues “interpretability.”",
    },
    {
        "input": "Lorem ipsum dolor sit amet, ",
        "target": "consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    },
]
SEED = 0
BATCH_SIZE = 1

OUTPUT_ROOT = Path("explanations")

METHODS = {
    # "kernel_shap": KernelShap,
    "lime": Lime,
    "occlusion": Occlusion,
    # "sobol": Sobol,
    "gradient_shap": GradientShap,
    "integrated_gradients": IntegratedGradients,
    "saliency": Saliency,
    "smoothgrad": SmoothGrad,
    "squared_grad": SquareGrad,
    "vargrad": VarGrad,
}

ADDITIONAL_PARAMETERS = {
    "kernel_shap": {"n_perturbations": 100},
    "sobol": {"n_token_perturbations": 4},
}


def main() -> None:
    print(f"\n{model_id=}")
    hf_model_id = HF_MODEL_IDS[model_id]
    torch.manual_seed(SEED)

    batch_inputs = [sample["input"] for sample in SAMPLES]
    batch_targets = [sample["target"] for sample in SAMPLES]

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, token=os.environ.get("HF_TOKEN")
    )
    model.eval()

    output_root = OUTPUT_ROOT / model_id / "attribution" / "general"
    output_root.mkdir(parents=True, exist_ok=True)

    for method_name, explainer_cls in METHODS.items():
        print(f"\n{method_name=}")
        explainer = explainer_cls(
            model,
            tokenizer,
            batch_size=BATCH_SIZE,
            **ADDITIONAL_PARAMETERS.get(method_name, {}),
        )
        attributions = explainer(model_inputs=batch_inputs, targets=batch_targets)

        for i, (ipt, tgt, attribution) in enumerate(
            zip(batch_inputs, batch_targets, attributions)
        ):
            sample_dir = output_root / f"sample-{i:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            html_path = sample_dir / f"{method_name}.html"

            plot_attributions(attribution, save_path=str(html_path))

            code_path = html_path.with_suffix(".py")
            code_path.write_text(
                render_code_snippet(
                    explainer_cls=explainer_cls,
                    sample_text=ipt,
                    target_text=tgt,
                    model_hf_id=hf_model_id,
                ),
                encoding="utf-8",
            )


def render_code_snippet(
    explainer_cls: type,
    sample_text: str,
    target_text: str,
    model_hf_id: str,
) -> str:
    return f"""from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import {explainer_cls.__name__}, plot_attributions

tokenizer = AutoTokenizer.from_pretrained({model_hf_id!r}, use_fast=True)
model = AutoModelForCausalLM.from_pretrained({model_hf_id!r})

explainer = {explainer_cls.__name__}(model, tokenizer)
attributions = explainer(
    model_inputs={sample_text!r},
    targets={target_text!r}
)
plot_attributions(attributions[0])
"""


if __name__ == "__main__":
    main()
