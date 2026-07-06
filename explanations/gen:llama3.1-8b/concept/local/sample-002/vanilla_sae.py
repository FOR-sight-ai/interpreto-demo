import torch
from datasets import load_dataset
from interpreto import SplitterForGeneration, plot_concepts
from interpreto.concepts import VanillaSAEConcepts, TopKInputs
from interpreto.concepts.methods.overcomplete import DeadNeuronsReanimationLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

splitter = SplitterForGeneration('meta-llama/Llama-3.1-8B', split_point=10, batch_size=1, device_map=device, torch_dtype=torch.bfloat16)
raw_inputs = load_dataset('wikimedia/wikipedia', '20231101.en')['train']["text"][:500]
# Truncate to the model's context window so `TopKInputs.interpret` and
# `get_activations` agree on the same tokenization.
inputs = [splitter.tokenizer.decode(splitter.tokenizer(t, truncation=True, max_length=4096, add_special_tokens=False)["input_ids"], skip_special_tokens=True) for t in raw_inputs]

activations, _ = splitter.get_activations(inputs, tqdm_bar=True)

concept_explainer = VanillaSAEConcepts(
    splitter,
    nb_concepts=2000,
    device=device,
)

concept_explainer.fit(
    activations,
    criterion=DeadNeuronsReanimationLoss,
    optimizer_class=torch.optim.Adam,
    scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs={'T_max': 5, 'eta_min': 1e-06},
    lr=0.001,
    nb_epochs=5,
    monitoring=0,
    batch_size=256,
)

interpretations = TopKInputs(
    concept_explainer=concept_explainer,
    k=10,
    concept_encoding_batch_size=256,
).interpret(
    inputs=inputs,
    latent_activations=activations,
    concepts_indices="all",
)
labels = {k: [t.lstrip("Ġ") for t in v.keys()] if v else None for k, v in interpretations.items()}

sample = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.'
encoded = splitter.tokenizer(sample, add_special_tokens=True)
sample_tokens = [t.replace("Ġ", " ") for t in splitter.tokenizer.convert_ids_to_tokens(encoded['input_ids'])]

local_importances = concept_explainer.concept_output_gradient(
    inputs=[sample], targets=None,
)[0].sum(dim=1)

local_activations, _ = splitter.get_activations([sample], include_special_tokens=True)
concepts_activations = concept_explainer.activations_to_concepts(local_activations)

plot_concepts(
    concepts_activations=concepts_activations,
    concepts_importances=local_importances,
    concepts_labels=labels,
    sample=sample_tokens,
    top_k=10,
)
