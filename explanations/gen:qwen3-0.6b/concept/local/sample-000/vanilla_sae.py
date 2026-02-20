import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from interpreto import ModelWithSplitPoints, plot_concepts
from interpreto.concepts import VanillaSAEConcepts, NeuronsAsConcepts
from interpreto.concepts.interpretations import TopKInputs
from interpreto.concepts.methods.overcomplete import DeadNeuronsReanimationLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

mwsp = ModelWithSplitPoints(
    'Qwen/Qwen3-0.6B',
    automodel=AutoModelForCausalLM,
    split_points=5,
    device_map=device,
    batch_size=1,
)

dataset = load_dataset('wikimedia/wikipedia', '20231101.en').shuffle(seed=0)
inputs = dataset["train"]["text"][:10000]

TOKEN = ModelWithSplitPoints.activation_granularities.TOKEN
activations_dict = mwsp.get_activations(
    inputs=inputs,
    activation_granularity=TOKEN,
)
activations = mwsp.get_split_activations(activations_dict)

concept_explainer = VanillaSAEConcepts(
    mwsp,
    nb_concepts=5000,
    device=device,
)

concept_explainer.fit(
    activations,
    criterion=DeadNeuronsReanimationLoss,
    optimizer_class=torch.optim.Adam,
    scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs={'T_max': 20, 'eta_min': 1e-06},
    lr=0.001,
    nb_epochs=20,
    batch_size=2048,
    monitoring=0,
)

WORD = ModelWithSplitPoints.activation_granularities.WORD
topk_inputs_method = TopKInputs(
    concept_explainer=concept_explainer,
    activation_granularity=WORD,
    k=5,
)
topk_words = topk_inputs_method.interpret(
    inputs=inputs[:500],
    concepts_indices="all",
)
labels = {k: list(v.keys()) for k, v in topk_words.items() if v}

sample = 'Alice and Bob enter the bar, then Alice offers a drink to Bob.'
sample_token_ids = mwsp.tokenizer([sample], return_tensors="pt")
sample_tokens = TOKEN.value.get_decomposition(
    sample_token_ids,
    tokenizer=mwsp.tokenizer,
    return_text=True,
)[0]

local_importances = concept_explainer.concept_output_gradient(
    inputs=[sample],
    activation_granularity=TOKEN,
    concepts_x_gradients=False,
    normalization=False,
)[0]
local_importances = local_importances.abs().sum(dim=1)

local_activations = mwsp.get_split_activations(mwsp.get_activations([sample], TOKEN))
concepts_activations = concept_explainer.encode_activations(local_activations)

plot_concepts(
    concepts_activations=concepts_activations,
    concepts_importances=local_importances,
    concepts_labels=labels,
    sample=sample_tokens,
    top_k=10,
)
