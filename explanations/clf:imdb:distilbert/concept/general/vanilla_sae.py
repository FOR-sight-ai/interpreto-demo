import torch
from datasets import load_dataset
from interpreto import SplitterForClassification, plot_concepts
from interpreto.concepts import VanillaSAEConcepts
from interpreto.concepts.interpretations import TopKInputs
from interpreto.concepts.methods.overcomplete import DeadNeuronsReanimationLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

splitter = SplitterForClassification('lvwerra/distilbert-imdb', batch_size=64, device_map=device)
inputs = load_dataset('stanfordnlp/imdb')['train']["text"]
classes_names = ['negative', 'positive']

activations, _ = splitter.get_activations(inputs, forward_kwargs={'truncation': True})

concept_explainer = VanillaSAEConcepts(
    splitter,
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

topk = TopKInputs(
    concept_explainer=concept_explainer,
    k=5,
    use_unique_words=3,
    unique_words_kwargs={"count_min_threshold": 50, "lemmatize": True},
)
labels = {k: list(v.keys()) for k, v in topk.interpret(inputs=inputs, concepts_indices="all").items()}

gradients = concept_explainer.concept_output_gradient(
    inputs=activations,
    targets=None,
    batch_size=64,
)
mean_gradients = torch.stack(gradients).squeeze().mean(0)

plot_concepts(
    classes_names=classes_names,
    concepts_importances=mean_gradients,
    concepts_labels=labels,
    top_k=10,
)
