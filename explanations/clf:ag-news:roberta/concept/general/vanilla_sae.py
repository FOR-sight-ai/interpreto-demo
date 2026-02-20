import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from interpreto import ModelWithSplitPoints, plot_concepts
from interpreto.concepts import VanillaSAEConcepts, NeuronsAsConcepts
from interpreto.concepts.interpretations import TopKInputs
from interpreto.concepts.methods.overcomplete import DeadNeuronsReanimationLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

model_with_split_points = ModelWithSplitPoints(
    'arman1o1/roberta_ag_news_model',
    automodel=AutoModelForSequenceClassification,
    split_points=11,
    device_map=device,
    batch_size=64,
)

inputs = load_dataset('fancyzhx/ag_news')['train'].shuffle(seed=0)["text"][:1000]

granularity = ModelWithSplitPoints.activation_granularities.CLS_TOKEN
activations = model_with_split_points.get_activations(
    inputs=inputs,
    activation_granularity=granularity,
    include_predicted_classes=True,
)

concept_explainer = VanillaSAEConcepts(
    model_with_split_points,
    nb_concepts=30,
    device=device,
)

concept_explainer.fit(
    activations,
    criterion=DeadNeuronsReanimationLoss,
    optimizer_class=torch.optim.Adam,
    scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs={'T_max': 20, 'eta_min': 1e-06},
    lr=0.001,
    nb_epochs=30,
    batch_size=2048,
    monitoring=0,
)

topk_inputs_method = TopKInputs(
    concept_explainer=concept_explainer,
    k=5,
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
    batch_size=64,
)

mean_gradients = torch.stack(gradients).abs().squeeze().mean(0)
labels = {k: list(v.keys()) for k, v in topk_words.items()}

plot_concepts(
    classes_names=['World', 'Sports', 'Business', 'Sci/Tech'],
    concepts_importances=mean_gradients,
    concepts_labels=labels,
    top_k=10,
)
