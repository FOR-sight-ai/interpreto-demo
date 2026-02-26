#!/usr/bin/env python3
"""Generate generation concept HTML files and minimal .py snippets."""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from interpreto import ModelWithSplitPoints, plot_concepts
from interpreto.concepts import (
    BatchTopKSAEConcepts,
    ICAConcepts,
    MpSAEConcepts,
    NeuronsAsConcepts,
    PCAConcepts,
    SemiNMFConcepts,
    SVDConcepts,
    VanillaSAEConcepts,
)
from interpreto.concepts.interpretations import TopKInputs
from interpreto.concepts.methods.overcomplete import DeadNeuronsReanimationLoss, MSELoss

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Configuration (edit these)
# ----------------------------
model_id = "gen:qwen3-0.6b"

MODEL_CONFIGS = {
    "gen:gpt2": {
        "hf_model_id": "gpt2",
        "split_points": 8,
    },
    "gen:qwen3-0.6b": {
        "hf_model_id": "Qwen/Qwen3-0.6B",
        "split_points": 5,
    },
    "gen:llama3.1-8b": {
        "hf_model_id": "meta-llama/Llama-3.1-8B",
        "split_points": 10,
    },
}

DATASET_HF_ID = "wikimedia/wikipedia"
DATASET_SUBSET = "20231101.en"
NUM_SAMPLES = 10000  # TODO: increase for better concepts
LABEL_SAMPLES = 500
SEED = 0

NB_CONCEPTS = 5000
TOP_K = 10
TOPK_WORDS = 5

BATCH_SIZE = 1
CONCEPT_BATCH_SIZE = 2048
GRADIENT_BATCH_SIZE = 1

WRITE_SNIPPETS_ONLY = True

SAMPLES = [
    "Alice and Bob enter the bar, then Alice offers a drink to Bob.",
    "Interpreto ships interpretable concept visualizations for language models.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
]

OUTPUT_ROOT = Path("/home/antonin.poche/interpreto-demo/explanations")

METHODS = {
    "batch_top_k_sae": BatchTopKSAEConcepts,
    "ica": ICAConcepts,
    "mp_sae": MpSAEConcepts,
    "neurons_as_concepts": NeuronsAsConcepts,
    "pca": PCAConcepts,
    "semi_nmf": SemiNMFConcepts,
    "svd": SVDConcepts,
    "vanilla_sae": VanillaSAEConcepts,
}

DEFAULT_INIT_PARAMETERS = {"nb_concepts": NB_CONCEPTS, "device": device}
INIT_PARAMETERS = {k: DEFAULT_INIT_PARAMETERS.copy() for k in METHODS.keys()}
INIT_PARAMETERS["batch_top_k_sae"]["top_k"] = 10 * CONCEPT_BATCH_SIZE
INIT_PARAMETERS["neurons_as_concepts"] = {}

SAES_FIT_PARAMETERS = {
    "criterion": DeadNeuronsReanimationLoss,  # type: ignore
    "optimizer_class": torch.optim.Adam,
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {"T_max": 20, "eta_min": 1e-6},
    "lr": 1e-3,
    "nb_epochs": 20,
    "batch_size": CONCEPT_BATCH_SIZE,
    "monitoring": 0,
}
FIT_PARAMETERS = {k: SAES_FIT_PARAMETERS.copy() for k in METHODS.keys() if "sae" in k}
FIT_PARAMETERS["ica"] = {"max_iter": 5000}
FIT_PARAMETERS["mp_sae"]["criterion"] = MSELoss


def _sample_tokens(model_with_split_points: ModelWithSplitPoints, sample: str) -> list[str]:
    token_ids = model_with_split_points.tokenizer([sample], return_tensors="pt")
    token_granularity = ModelWithSplitPoints.activation_granularities.TOKEN
    return token_granularity.value.get_decomposition(
        token_ids,
        tokenizer=model_with_split_points.tokenizer,
        return_text=True,
    )[0]


def _format_param_value(value: object, param_name: str) -> tuple[str, set[str]]:
    imports: set[str] = set()
    if param_name == "device":
        return "device", imports
    if isinstance(value, type):
        module = value.__module__
        if module.startswith("torch.optim.lr_scheduler"):
            return f"torch.optim.lr_scheduler.{value.__name__}", imports
        if module.startswith("torch.optim"):
            return f"torch.optim.{value.__name__}", imports
        if module.startswith("interpreto."):
            imports.add(value.__name__)
            return value.__name__, imports
        return value.__name__, imports
    return repr(value), imports


def _format_kwargs_lines(params: dict[str, object], indent: str) -> tuple[list[str], set[str]]:
    lines: list[str] = []
    imports: set[str] = set()
    for key, value in params.items():
        value_str, value_imports = _format_param_value(value, key)
        imports.update(value_imports)
        lines.append(f"{indent}{key}={value_str},")
    return lines, imports


def _write_code_snippet(
    code_path: Path,
    explainer_cls: type,
    model_hf_id: str,
    split_points: int,
    sample_text: str,
    init_params: dict[str, object],
    fit_params: dict[str, object],
) -> None:
    code_path.write_text(
        render_code_snippet(
            explainer_cls=explainer_cls,
            model_hf_id=model_hf_id,
            split_points=split_points,
            sample_text=sample_text,
            init_params=init_params,
            fit_params=fit_params,
        ),
        encoding="utf-8",
    )


def render_code_snippet(
    explainer_cls: type,
    model_hf_id: str,
    split_points: int,
    sample_text: str,
    init_params: dict[str, object],
    fit_params: dict[str, object],
) -> str:
    init_lines, init_imports = _format_kwargs_lines(init_params, indent="    ")
    fit_lines, fit_imports = _format_kwargs_lines(fit_params, indent="    ")
    extra_imports = sorted(init_imports | fit_imports)

    concept_imports = [explainer_cls.__name__, "NeuronsAsConcepts"]
    deduped_concept_imports: list[str] = []
    seen: set[str] = set()
    for name in concept_imports:
        if name in seen:
            continue
        seen.add(name)
        deduped_concept_imports.append(name)

    lines = [
        "import torch",
        "from datasets import load_dataset",
        "from transformers import AutoModelForCausalLM",
        "from interpreto import ModelWithSplitPoints, plot_concepts",
        f"from interpreto.concepts import {', '.join(deduped_concept_imports)}",
        "from interpreto.concepts.interpretations import TopKInputs",
    ]
    if extra_imports:
        lines.append("from interpreto.concepts.methods.overcomplete import " + ", ".join(extra_imports))
    lines.append("")
    lines.append('device = "cuda" if torch.cuda.is_available() else "cpu"')
    lines.append("")
    lines.append("mwsp = ModelWithSplitPoints(")
    lines.append(f"    {model_hf_id!r},")
    lines.append("    automodel=AutoModelForCausalLM,")
    lines.append(f"    split_points={split_points!r},")
    lines.append("    device_map=device,")
    lines.append(f"    batch_size={BATCH_SIZE},")
    lines.append(")")
    lines.append("")
    lines.append(f"dataset = load_dataset({DATASET_HF_ID!r}, {DATASET_SUBSET!r}).shuffle(seed={SEED})")
    lines.append(f'inputs = dataset["train"]["text"][:{NUM_SAMPLES}]')
    lines.append("")
    lines.append("TOKEN = ModelWithSplitPoints.activation_granularities.TOKEN")
    lines.append("activations_dict = mwsp.get_activations(")
    lines.append("    inputs=inputs,")
    lines.append("    activation_granularity=TOKEN,")
    lines.append(")")
    lines.append("activations = mwsp.get_split_activations(activations_dict)")
    lines.append("")
    lines.append(f"concept_explainer = {explainer_cls.__name__}(")
    lines.append("    mwsp,")
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
    lines.append("WORD = ModelWithSplitPoints.activation_granularities.WORD")
    lines.append("topk_inputs_method = TopKInputs(")
    lines.append("    concept_explainer=concept_explainer,")
    lines.append("    activation_granularity=WORD,")
    lines.append(f"    k={TOPK_WORDS},")
    lines.append(")")
    lines.append("topk_words = topk_inputs_method.interpret(")
    lines.append(f"    inputs=inputs[:{LABEL_SAMPLES}],")
    lines.append('    concepts_indices="all",')
    lines.append(")")
    lines.append("labels = {k: list(v.keys()) for k, v in topk_words.items() if v}")
    lines.append("")
    lines.append(f"sample = {sample_text!r}")
    lines.append('sample_token_ids = mwsp.tokenizer([sample], return_tensors="pt")')
    lines.append("sample_tokens = TOKEN.value.get_decomposition(")
    lines.append("    sample_token_ids,")
    lines.append("    tokenizer=mwsp.tokenizer,")
    lines.append("    return_text=True,")
    lines.append(")[0]")
    lines.append("")
    lines.append("local_importances = concept_explainer.concept_output_gradient(")
    lines.append("    inputs=[sample],")
    lines.append("    activation_granularity=TOKEN,")
    lines.append("    concepts_x_gradients=False,")
    lines.append("    normalization=False,")
    lines.append(")[0]")
    lines.append("local_importances = local_importances.abs().sum(dim=1)")
    lines.append("")
    lines.append("local_activations = mwsp.get_split_activations(mwsp.get_activations([sample], TOKEN))")
    lines.append("concepts_activations = concept_explainer.encode_activations(local_activations)")
    lines.append("")
    lines.append("plot_concepts(")
    lines.append("    concepts_activations=concepts_activations,")
    lines.append("    concepts_importances=local_importances,")
    lines.append("    concepts_labels=labels,")
    lines.append("    sample=sample_tokens,")
    lines.append(f"    top_k={TOP_K},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    config = MODEL_CONFIGS[model_id]
    split_points = config["split_points"]

    output_root = OUTPUT_ROOT / model_id / "concept" / "local"
    output_root.mkdir(parents=True, exist_ok=True)

    if WRITE_SNIPPETS_ONLY:
        for method_name, explainer_cls in METHODS.items():
            init_params = INIT_PARAMETERS.get(method_name, {})
            fit_params = FIT_PARAMETERS.get(method_name, {})
            for i, sample in enumerate(SAMPLES):
                sample_dir = output_root / f"sample-{i:03d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                code_path = sample_dir / f"{method_name}.py"
                _write_code_snippet(
                    code_path=code_path,
                    explainer_cls=explainer_cls,
                    model_hf_id=config["hf_model_id"],
                    split_points=split_points,
                    sample_text=sample,
                    init_params=init_params,
                    fit_params=fit_params,
                )
        return

    torch.manual_seed(SEED)

    dataset = load_dataset(DATASET_HF_ID, DATASET_SUBSET).shuffle(seed=SEED)
    inputs = dataset["train"]["text"][:NUM_SAMPLES]
    label_inputs = inputs[:LABEL_SAMPLES]

    model_with_split_points = ModelWithSplitPoints(
        config["hf_model_id"],
        automodel=AutoModelForCausalLM,  # type: ignore
        split_points=split_points,
        device_map=device,
        batch_size=BATCH_SIZE,
    )

    token_granularity = ModelWithSplitPoints.activation_granularities.TOKEN
    acti_path = f"./data/{model_id}.pt"
    if os.path.exists(acti_path):
        activations = torch.load(acti_path)
    else:
        activations_dict = model_with_split_points.get_activations(
            inputs=inputs,
            activation_granularity=token_granularity,
            tqdm_bar=True,
        )
        activations = model_with_split_points.get_split_activations(activations_dict)
        torch.save(activations, acti_path)

    for method_name, explainer_cls in METHODS.items():
        sample_paths = [output_root / f"sample-{i:03d}" / f"{method_name}.html" for i in range(len(SAMPLES))]
        if all(path.exists() for path in sample_paths):
            continue

        print("\n", method_name)
        init_params = INIT_PARAMETERS.get(method_name, {})
        fit_params = FIT_PARAMETERS.get(method_name, {})
        concept_explainer = explainer_cls(
            model_with_split_points,
            **init_params,
        )
        if method_name != "neurons_as_concepts":
            try:
                concept_explainer.fit(activations, **fit_params)
            except Exception as e:
                print(f"Error while fitting {method_name}: {e}")
                continue

        topk_inputs_method = TopKInputs(
            concept_explainer=concept_explainer,
            activation_granularity=ModelWithSplitPoints.activation_granularities.WORD,
            concept_encoding_batch_size=CONCEPT_BATCH_SIZE,
            k=TOPK_WORDS,
            concept_model_device=device,
        )

        topk_words = topk_inputs_method.interpret(
            inputs=label_inputs,
            concepts_indices="all",
        )
        labels = {k: list(v.keys()) for k, v in topk_words.items() if v}

        print("\nconcepts learned and interpreted, generating local explanations...")
        for i, sample in enumerate(SAMPLES):
            sample_dir = output_root / f"sample-{i:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            html_path = sample_dir / f"{method_name}.html"
            if os.path.exists(html_path):
                continue

            sample_tokens = _sample_tokens(model_with_split_points, sample)
            local_importances = concept_explainer.concept_output_gradient(
                inputs=[sample],
                targets=None,
                activation_granularity=token_granularity,
                concepts_x_gradients=False,
                normalization=False,
                batch_size=GRADIENT_BATCH_SIZE,
            )[0]
            local_importances = local_importances.abs().sum(dim=1)

            local_activations = model_with_split_points.get_split_activations(
                model_with_split_points.get_activations([sample], token_granularity)
            )
            concepts_activations = concept_explainer.encode_activations(local_activations)

            print(f"Saving {html_path}")
            plot_concepts(
                concepts_activations=concepts_activations,
                concepts_importances=local_importances,
                concepts_labels=labels,
                sample=sample_tokens,
                top_k=TOP_K,
                save_path=str(html_path),
            )

            code_path = html_path.with_suffix(".py")
            _write_code_snippet(
                code_path=code_path,
                explainer_cls=explainer_cls,
                model_hf_id=config["hf_model_id"],
                split_points=split_points,
                sample_text=sample,
                init_params=init_params,
                fit_params=fit_params,
            )

        del (
            concept_explainer,
            topk_inputs_method,
            topk_words,
            labels,
        )


if __name__ == "__main__":
    main()
