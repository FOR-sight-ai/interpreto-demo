#!/usr/bin/env python3
"""Generate generation attribution HTML files and minimal .py snippets.

Interpreto 0.5.0 API. For each generation model, this script:

1. Loads the model + tokenizer once.
2. Runs every attribution method on ``SAMPLES`` (fixed input/target pairs).
3. Emits one HTML file plus a matching minimal ``.py`` snippet per
   (sample, method) pair, under
   ``explanations/<model_id>/attribution/general/sample-XXX/``.
"""

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
    Sobol,
    SquareGrad,
    VarGrad,
    plot_attributions,
)


# ----------------------------
# Configuration (edit these)
# ----------------------------
model_id = "gen:gpt2"

# Gradient-based methods: interpreto 0.5.0 embeds inputs in float32 but
# does not cast them to the model dtype, so these break on models loaded
# with anything other than float32.
GRADIENT_METHODS = {
    "gradient_shap",
    "integrated_gradients",
    "saliency",
    "smoothgrad",
    "squared_grad",
    "vargrad",
}

MODEL_CONFIGS = {
    "gen:gpt2": {
        "hf_model_id": "gpt2",
        "torch_dtype": torch.float32,
    },
    "gen:qwen3-0.6b": {
        "hf_model_id": "Qwen/Qwen3-0.6B",
        "torch_dtype": torch.float32,
    },
    "gen:llama3.1-8b": {
        "hf_model_id": "meta-llama/Llama-3.1-8B",
        # float32 (~32 GB) doesn't fit on a single 24 GB GPU, and
        # ``device_map="auto"`` conflicts with the InferenceWrapper's
        # ``self.model.to(device)`` call. bfloat16 fits but breaks the
        # gradient-based methods (see GRADIENT_METHODS above), so we
        # only emit perturbation methods for this model.
        "torch_dtype": torch.bfloat16,
        "skip_methods": GRADIENT_METHODS,
    },
}

SAMPLES = [
    {
        "input": "Alice and Bob enter the bar, ",
        "target": "then Alice offers a drink to Bob.",
    },
    {
        "input": "We called our library Interpreto is a good name? ",
        "target": "\u201cInterpreto\u201d is a solid name: short, distinctive, and it strongly cues \u201cinterpretability.\u201d",
    },
    {
        "input": "Lorem ipsum dolor sit amet, ",
        "target": "consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    },
]
SEED = 0
BATCH_SIZE = 1

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
    target_text: str,
    hf_model_id: str,
    torch_dtype: torch.dtype,
) -> str:
    """Return a minimal, self-contained snippet reproducing one HTML."""
    dtype_str = str(torch_dtype)  # e.g. "torch.float32"

    return f"""import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import {explainer_cls.__name__}, plot_attributions

tokenizer = AutoTokenizer.from_pretrained({hf_model_id!r}, use_fast=True)
model = AutoModelForCausalLM.from_pretrained({hf_model_id!r}, torch_dtype={dtype_str})

explainer = {explainer_cls.__name__}(model, tokenizer)
attributions = explainer(
    model_inputs={sample_text!r},
    targets={target_text!r},
)

plot_attributions(attributions[0])
"""


def main() -> None:
    print(f"\n== {model_id=}")
    config = MODEL_CONFIGS[model_id]
    hf_model_id = config["hf_model_id"]
    torch_dtype = config["torch_dtype"]
    skip_methods = config.get("skip_methods") or set()
    torch.manual_seed(SEED)

    batch_inputs = [sample["input"] for sample in SAMPLES]
    batch_targets = [sample["target"] for sample in SAMPLES]

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch_dtype,
    )
    model.eval()

    output_root = OUTPUT_ROOT / model_id / "attribution" / "general"
    output_root.mkdir(parents=True, exist_ok=True)

    for method_name, explainer_cls in METHODS.items():
        if method_name in skip_methods:
            print(f"\n-- {method_name} (skipped for {model_id})")
            continue
        print(f"\n-- {method_name}")
        explainer = explainer_cls(model, tokenizer, batch_size=BATCH_SIZE)
        attributions = explainer(model_inputs=batch_inputs, targets=batch_targets)

        for i, (ipt, tgt, attribution) in enumerate(
            zip(batch_inputs, batch_targets, attributions, strict=False)
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
                    hf_model_id=hf_model_id,
                    torch_dtype=torch_dtype,
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
