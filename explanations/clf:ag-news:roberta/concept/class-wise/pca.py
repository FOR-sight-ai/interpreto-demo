import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from interpreto import ModelWithSplitPoints, plot_concepts
from interpreto.concepts import PCAConcepts, NeuronsAsConcepts
from interpreto.concepts.interpretations import TopKInputs

device = "cuda" if torch.cuda.is_available() else "cpu"

model_with_split_points = ModelWithSplitPoints(
    'arman1o1/roberta_ag_news_model',
    automodel=AutoModelForSequenceClassification,
    split_points=11,
    device_map=device,
    batch_size=64,
)

dataset = load_dataset('fancyzhx/ag_news')['test'].shuffle(seed=0)
inputs = dataset["text"][:10000]

granularity = ModelWithSplitPoints.activation_granularities.CLS_TOKEN
activations = model_with_split_points.get_activations(
    inputs=inputs,
    activation_granularity=granularity,
    include_predicted_classes=True,
)

concepts_importances = {}
concepts_labels = {}

for target, class_name in enumerate(['World', 'Sports', 'Business', 'Sci/Tech']):
    indices = (activations["predictions"] == target).nonzero(as_tuple=True)[0].tolist()

    class_inputs = [inputs[i] for i in indices]
    class_activations = {k: v[indices] for k, v in activations.items()}

    concept_explainer = PCAConcepts(
        model_with_split_points,
        nb_concepts=20,
        device=device,
    )

    concept_explainer.fit(class_activations)

    topk_inputs_method = TopKInputs(
        concept_explainer=concept_explainer,
        k=5,
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
        batch_size=64,
    )

    concepts_importances[target] = torch.stack(gradients).abs().squeeze().mean(0)

plot_concepts(
    classes_names=['World', 'Sports', 'Business', 'Sci/Tech'],
    concepts_importances=concepts_importances,
    concepts_labels=concepts_labels,
    top_k=10,
)
