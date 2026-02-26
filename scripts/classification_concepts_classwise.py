#!/usr/bin/env python3
"""Generate classification class-wise concept HTML files and minimal .py snippets."""

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

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
model_id = "clf:ag-news:roberta"

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
        "split_points": 11,
    },
    "clf:imdb:distilbert": {
        "hf_model_id": "lvwerra/distilbert-imdb",
        "hf_dataset_id": "stanfordnlp/imdb",
        "classes_names": [
            "negative",
            "positive",
        ],
        "split_points": 5,
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
        "split_points": 11,
    },
}

DATASET_SPLIT = "test"
NUM_SAMPLES = 10000
SEED = 0

NB_CONCEPTS = 20
TOP_K = 10
TOPK_WORDS = 5

BATCH_SIZE = 64
GRADIENT_BATCH_SIZE = 64

WRITE_SNIPPETS_ONLY = True

OUTPUT_ROOT = Path("/home/antonin.poche/interpreto-demo/explanations")

METHODS = {
    # "batch_top_k_sae": BatchTopKSAEConcepts,
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
# INIT_PARAMETERS["batch_top_k_sae"]["top_k"] = 10 * BATCH_SIZE
INIT_PARAMETERS["neurons_as_concepts"] = {}
SAES_FIT_PARAMETERS = {
    "criterion": DeadNeuronsReanimationLoss,  # type: ignore
    "optimizer_class": torch.optim.Adam,
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {"T_max": 20, "eta_min": 1e-6},
    "lr": 1e-3,
    "nb_epochs": 30,
    "batch_size": 32 * BATCH_SIZE,
    "monitoring": 0,
}
FIT_PARAMETERS = {k: SAES_FIT_PARAMETERS.copy() for k in METHODS.keys() if "sae" in k}
FIT_PARAMETERS["ica"] = {"max_iter": 5000}
FIT_PARAMETERS["mp_sae"]["criterion"] = MSELoss


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
    dataset_hf_id: str,
    classes_names: list[str],
    split_points: int,
    init_params: dict[str, object],
    fit_params: dict[str, object],
) -> None:
    code_path.write_text(
        render_code_snippet(
            explainer_cls=explainer_cls,
            model_hf_id=model_hf_id,
            dataset_hf_id=dataset_hf_id,
            classes_names=classes_names,
            split_points=split_points,
            init_params=init_params,
            fit_params=fit_params,
        ),
        encoding="utf-8",
    )


def render_code_snippet(
    explainer_cls: type,
    model_hf_id: str,
    dataset_hf_id: str,
    classes_names: list[str],
    split_points: int,
    init_params: dict[str, object],
    fit_params: dict[str, object],
) -> str:
    init_lines, init_imports = _format_kwargs_lines(init_params, indent="        ")
    fit_lines, fit_imports = _format_kwargs_lines(fit_params, indent="        ")
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
        "from transformers import AutoModelForSequenceClassification",
        "from interpreto import ModelWithSplitPoints, plot_concepts",
        f"from interpreto.concepts import {', '.join(deduped_concept_imports)}",
        "from interpreto.concepts.interpretations import TopKInputs",
    ]
    if extra_imports:
        lines.append("from interpreto.concepts.methods.overcomplete import " + ", ".join(extra_imports))
    lines.append("")
    lines.append('device = "cuda" if torch.cuda.is_available() else "cpu"')
    lines.append("")
    lines.append("model_with_split_points = ModelWithSplitPoints(")
    lines.append(f"    {model_hf_id!r},")
    lines.append("    automodel=AutoModelForSequenceClassification,")
    lines.append(f"    split_points={split_points!r},")
    lines.append("    device_map=device,")
    lines.append(f"    batch_size={BATCH_SIZE},")
    lines.append(")")
    lines.append("")
    lines.append(f"dataset = load_dataset({dataset_hf_id!r})[{DATASET_SPLIT!r}].shuffle(seed={SEED})")
    lines.append(f'inputs = dataset["text"][:{NUM_SAMPLES}]')
    lines.append("")
    lines.append("granularity = ModelWithSplitPoints.activation_granularities.CLS_TOKEN")
    lines.append("activations = model_with_split_points.get_activations(")
    lines.append("    inputs=inputs,")
    lines.append("    activation_granularity=granularity,")
    lines.append("    include_predicted_classes=True,")
    lines.append(")")
    lines.append("")
    lines.append("concepts_importances = {}")
    lines.append("concepts_labels = {}")
    lines.append("")
    lines.append(f"for target, class_name in enumerate({classes_names!r}):")
    lines.append('    indices = (activations["predictions"] == target).nonzero(as_tuple=True)[0].tolist()')
    lines.append("")
    lines.append("    class_inputs = [inputs[i] for i in indices]")
    lines.append("    class_activations = {k: v[indices] for k, v in activations.items()}")
    lines.append("")
    lines.append(f"    concept_explainer = {explainer_cls.__name__}(")
    lines.append("        model_with_split_points,")
    lines.extend(init_lines)
    lines.append("    )")
    if explainer_cls is not NeuronsAsConcepts:
        lines.append("")
        if fit_lines:
            lines.append("    concept_explainer.fit(")
            lines.append("        class_activations,")
            lines.extend(fit_lines)
            lines.append("    )")
        else:
            lines.append("    concept_explainer.fit(class_activations)")
    lines.append("")
    lines.append("    topk_inputs_method = TopKInputs(")
    lines.append("        concept_explainer=concept_explainer,")
    lines.append(f"        k={TOPK_WORDS},")
    lines.append("        activation_granularity=granularity,")
    lines.append("        use_unique_words=True,")
    lines.append("        unique_words_kwargs={")
    lines.append('            "count_min_threshold": max(1, round(len(class_inputs) * 0.002)),')
    lines.append('            "lemmatize": True,')
    lines.append('            "words_to_ignore": [],')
    lines.append("        },")
    lines.append("    )")
    lines.append("")
    lines.append("    topk_words = topk_inputs_method.interpret(")
    lines.append("        inputs=class_inputs,")
    lines.append('        concepts_indices="all",')
    lines.append("    )")
    lines.append("    concepts_labels[target] = {k: list(v.keys()) for k, v in topk_words.items() if v}")
    lines.append("")
    lines.append("    gradients = concept_explainer.concept_output_gradient(")
    lines.append("        inputs=class_inputs,")
    lines.append("        targets=[target],")
    lines.append("        activation_granularity=granularity,")
    lines.append("        concepts_x_gradients=True,")
    lines.append(f"        batch_size={GRADIENT_BATCH_SIZE},")
    lines.append("    )")
    lines.append("")
    lines.append("    concepts_importances[target] = torch.stack(gradients).abs().squeeze().mean(0)")
    lines.append("")
    lines.append("plot_concepts(")
    lines.append(f"    classes_names={classes_names!r},")
    lines.append("    concepts_importances=concepts_importances,")
    lines.append("    concepts_labels=concepts_labels,")
    lines.append(f"    top_k={TOP_K},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    config = MODEL_CONFIGS[model_id]
    classes_names = config["classes_names"]
    split_points = config["split_points"]

    output_root = OUTPUT_ROOT / model_id / "concept" / "class-wise"
    output_root.mkdir(parents=True, exist_ok=True)

    if WRITE_SNIPPETS_ONLY:
        for method_name, explainer_cls in METHODS.items():
            init_params = INIT_PARAMETERS.get(method_name, {})
            fit_params = FIT_PARAMETERS.get(method_name, {})
            code_path = output_root / f"{method_name}.py"
            _write_code_snippet(
                code_path=code_path,
                explainer_cls=explainer_cls,
                model_hf_id=config["hf_model_id"],
                dataset_hf_id=config["hf_dataset_id"],
                classes_names=classes_names,
                split_points=split_points,
                init_params=init_params,
                fit_params=fit_params,
            )
        return

    torch.manual_seed(SEED)

    dataset = load_dataset(config["hf_dataset_id"])[DATASET_SPLIT].shuffle(seed=SEED)
    inputs: list[str] = dataset["text"][:NUM_SAMPLES]  # type: ignore

    model_with_split_points = ModelWithSplitPoints(
        config["hf_model_id"],
        automodel=AutoModelForSequenceClassification,  # type: ignore
        split_points=split_points,
        device_map=device,
        batch_size=BATCH_SIZE,
    )

    granularity = ModelWithSplitPoints.activation_granularities.CLS_TOKEN
    activations = model_with_split_points.get_activations(
        inputs=inputs,  # type: ignore
        activation_granularity=granularity,
        include_predicted_classes=True,
    )

    predictions = activations["predictions"]

    for method_name, explainer_cls in METHODS.items():
        html_path = output_root / f"{method_name}.html"
        if html_path.exists():
            continue
        print("\n", method_name)
        init_params = INIT_PARAMETERS.get(method_name, {})
        fit_params = FIT_PARAMETERS.get(method_name, {})

        concepts_importances = {}
        concepts_labels = {}

        for target in range(len(classes_names)):
            indices = (predictions == target).nonzero(as_tuple=True)[0].tolist()

            class_inputs = [inputs[i] for i in indices]
            class_wise_activations = {k: v[indices] for k, v in activations.items()}

            concept_explainer = explainer_cls(
                model_with_split_points,
                **init_params,
            )
            if method_name != "neurons_as_concepts":
                concept_explainer.fit(activations, **fit_params)

            topk_inputs_method = TopKInputs(
                concept_explainer=concept_explainer,
                k=TOPK_WORDS,
                activation_granularity=granularity,
                use_unique_words=True,
                unique_words_kwargs={
                    "count_min_threshold": max(1, round(len(class_inputs) * 0.002)),
                    "lemmatize": True,
                    "words_to_ignore": [],
                },
            )

            topk_words = topk_inputs_method.interpret(
                inputs=class_inputs,
                concepts_indices="all",
            )

            concepts_labels[target] = {k: list(v.keys()) for k, v in topk_words.items() if v}

            gradients = concept_explainer.concept_output_gradient(
                inputs=class_inputs,
                targets=[target],
                activation_granularity=granularity,
                concepts_x_gradients=True,
                batch_size=GRADIENT_BATCH_SIZE,
            )

            concepts_importances[target] = torch.stack(gradients).abs().squeeze().mean(0)

            del (
                concept_explainer,
                topk_inputs_method,
                topk_words,
                gradients,
            )

        if not concepts_importances:
            print("No classes had enough samples, skipping.")
            continue

        print(f"Saving {html_path}")
        plot_concepts(
            classes_names=classes_names,
            concepts_importances=concepts_importances,
            concepts_labels=concepts_labels,
            top_k=TOP_K,
            save_path=str(html_path),
        )

        code_path = html_path.with_suffix(".py")
        _write_code_snippet(
            code_path=code_path,
            explainer_cls=explainer_cls,
            model_hf_id=config["hf_model_id"],
            dataset_hf_id=config["hf_dataset_id"],
            classes_names=classes_names,
            split_points=split_points,
            init_params=init_params,
            fit_params=fit_params,
        )


if __name__ == "__main__":
    main()
