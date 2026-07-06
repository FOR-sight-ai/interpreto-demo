import torch
from datasets import load_dataset
from interpreto import SplitterForClassification, plot_concepts
from interpreto.concepts import PCAConcepts
from interpreto.concepts.interpretations import TopKInputs

device = "cuda" if torch.cuda.is_available() else "cpu"

splitter = SplitterForClassification('nateraw/bert-base-uncased-emotion', batch_size=64, device_map=device)
inputs = load_dataset('dair-ai/emotion', 'split')['train']["text"]
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

activations, _ = splitter.get_activations(inputs)

concept_explainer = PCAConcepts(
    splitter,
    nb_concepts=30,
    device=device,
)

concept_explainer.fit(activations)

topk = TopKInputs(
    concept_explainer=concept_explainer,
    k=5,
    use_unique_words=3,
    unique_words_kwargs={"count_min_threshold": 32, "lemmatize": True},
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
