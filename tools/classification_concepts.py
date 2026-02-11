#!/usr/bin/env python3
"""Generate classification concept HTML files and minimal .py snippets."""

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
    SemiNMFConcepts,
    SparsePCAConcepts,
    SVDConcepts,
    VanillaSAEConcepts,
)
from interpreto.concepts.methods.overcomplete import DeadNeuronsReanimationLoss
from interpreto.concepts.interpretations import TopKInputs


# ----------------------------
# Configuration (edit these)
# ----------------------------
model_id = "clf:imdb:distilbert"

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

DATASET_SPLIT = "train"
NUM_SAMPLES = 1000
SEED = 0

NB_CONCEPTS = 30
TOP_K = 10
TOPK_WORDS = 5

BATCH_SIZE = 64
GRADIENT_BATCH_SIZE = 64

OUTPUT_ROOT = Path("explanations")

METHODS = {
    "batch_top_k_sae": BatchTopKSAEConcepts,
    "ica": ICAConcepts,
    "mp_sae": MpSAEConcepts,
    "neurons_as_concepts": NeuronsAsConcepts,
    "semi_nmf": SemiNMFConcepts,
    "sparse_pca": SparsePCAConcepts,
    "svd": SVDConcepts,
    "vanilla_sae": VanillaSAEConcepts,
}

SAES_TRAIN_PARAMETERS = {
    "criterion": DeadNeuronsReanimationLoss,
    "optimizer_class": torch.optim.Adam,
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {"T_max": 20, "eta_min": 1e-6},
    "lr": 1e-3,
    "nb_epochs": 20,
    "batch_size": 32 * BATCH_SIZE,
    "monitoring": 1,
}

ADDITIONAL_INIT_PARAMETERS = {
    "batch_top_k_sae": {"top_k": 10 * BATCH_SIZE},
}
ADDITIONAL_FIT_PARAMETERS = {
    "batch_top_k_sae": {**SAES_TRAIN_PARAMETERS},
    "mp_sae": {**SAES_TRAIN_PARAMETERS},
    "vanilla_sae": {**SAES_TRAIN_PARAMETERS},
}


def render_code_snippet(
    explainer_cls: type,
    model_hf_id: str,
    dataset_hf_id: str,
    classes_names: list[str],
    split_points: int,
    nb_concepts: int,
    top_k: int,
) -> str:
    return f"""import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from interpreto import ModelWithSplitPoints, plot_concepts
from interpreto.concepts import {explainer_cls.__name__}
from interpreto.concepts.interpretations import TopKInputs

device = "cuda" if torch.cuda.is_available() else "cpu"

model_with_split_points = ModelWithSplitPoints(
    {model_hf_id!r},
    automodel=AutoModelForSequenceClassification,
    split_points={split_points!r},
    device_map=device,
    batch_size={BATCH_SIZE},
)

inputs = load_dataset({dataset_hf_id!r})[{DATASET_SPLIT!r}].shuffle(seed=SEED)["text"][:{NUM_SAMPLES}]

granularity = ModelWithSplitPoints.activation_granularities.CLS_TOKEN
activations = model_with_split_points.get_activations(
    inputs=inputs,
    activation_granularity=granularity,
    include_predicted_classes=True,
)

concept_explainer = {explainer_cls.__name__}(model_with_split_points, nb_concepts={nb_concepts}, device=device)
concept_explainer.fit(activations)

topk_inputs_method = TopKInputs(
    concept_explainer=concept_explainer,
    k={TOPK_WORDS},
    activation_granularity=granularity,
    use_unique_words=True,
    unique_words_kwargs={{
        "count_min_threshold": max(1, round(len(inputs) * 0.002)),
        "lemmatize": True,
        "words_to_ignore": [],
    }},
)

topk_words = topk_inputs_method.interpret(
    inputs=inputs,
    concepts_indices="all",
)

gradients = concept_explainer.concept_output_gradient(
    inputs=inputs,
    targets=None,
    activation_granularity=granularity,
    concepts_x_gradients=True,
    batch_size={GRADIENT_BATCH_SIZE},
)

mean_gradients = torch.stack(gradients).abs().squeeze().mean(0)
labels = {{k: list(v.keys()) for k, v in topk_words.items()}}

plot_concepts(
    classes_names={classes_names!r},
    concepts_importances=mean_gradients,
    concepts_labels=labels,
    top_k={top_k},
)
"""


def main() -> None:
    config = MODEL_CONFIGS[model_id]
    classes_names = config["classes_names"]
    split_points = config["split_points"]

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset(config["hf_dataset_id"])["test"].shuffle(seed=SEED)["text"]
    inputs: list[str] = dataset[:NUM_SAMPLES]  # type: ignore

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

    output_root = OUTPUT_ROOT / model_id / "concept" / "general"
    output_root.mkdir(parents=True, exist_ok=True)

    for method_name, explainer_cls in METHODS.items():
        concept_explainer = explainer_cls(
            model_with_split_points,
            nb_concepts=NB_CONCEPTS,
            device=device,
            **ADDITIONAL_INIT_PARAMETERS.get(method_name, {}),
        )
        concept_explainer.fit(
            activations, **ADDITIONAL_FIT_PARAMETERS.get(method_name, {})
        )

        topk_inputs_method = TopKInputs(
            concept_explainer=concept_explainer,
            k=TOPK_WORDS,
            activation_granularity=granularity,
            use_unique_words=True,
            unique_words_kwargs={
                "count_min_threshold": max(1, round(len(inputs) * 0.002)),
                "lemmatize": True,
                "words_to_ignore": [],
            },
        )

        topk_words = topk_inputs_method.interpret(
            inputs=inputs,
            concepts_indices="all",
        )

        gradients = concept_explainer.concept_output_gradient(
            inputs=inputs,
            targets=None,
            activation_granularity=granularity,
            concepts_x_gradients=True,
            batch_size=GRADIENT_BATCH_SIZE,
        )

        mean_gradients = torch.stack(gradients).abs().squeeze().mean(0)
        labels = {k: list(v.keys()) for k, v in topk_words.items()}

        html_path = output_root / f"{method_name}.html"
        plot_concepts(
            classes_names=classes_names,
            concepts_importances=mean_gradients,
            concepts_labels=labels,
            top_k=TOP_K,
            save_path=str(html_path),
        )

        code_path = html_path.with_suffix(".py")
        code_path.write_text(
            render_code_snippet(
                explainer_cls=explainer_cls,
                model_hf_id=config["hf_model_id"],
                dataset_hf_id=config["hf_dataset_id"],
                classes_names=classes_names,
                split_points=split_points,
                nb_concepts=NB_CONCEPTS,
                top_k=TOP_K,
            ),
            encoding="utf-8",
        )
        del (
            concept_explainer,
            topk_inputs_method,
            topk_words,
            gradients,
            mean_gradients,
            labels,
        )


if __name__ == "__main__":
    main()
