import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from interpreto import SplitterForGeneration, plot_concepts
from interpreto.concepts import VanillaSAEConcepts, LLMLabels

# ``HuggingFaceLLM`` is not (yet) shipped by interpreto — this import will
# start working once the class lands upstream. In the meantime, see the
# "Using your own LLM interface" section of the generation concept
# tutorial for the reference implementation you can paste in here.
from interpreto.commons import HuggingFaceLLM
from interpreto.concepts.methods.overcomplete import DeadNeuronsReanimationLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

splitter = SplitterForGeneration('Qwen/Qwen3-0.6B', split_point=5, batch_size=8, device_map=device)
raw_inputs = load_dataset('wikimedia/wikipedia', '20231101.en')['train']["text"][:2000]
# Truncate to the model's context window so `LLMLabels.interpret` and
# `get_activations` agree on the same tokenization.
inputs = [splitter.tokenizer.decode(splitter.tokenizer(t, truncation=True, max_length=4096, add_special_tokens=False)["input_ids"], skip_special_tokens=True) for t in raw_inputs]

activations, _ = splitter.get_activations(inputs, tqdm_bar=True)

concept_explainer = VanillaSAEConcepts(
    splitter,
    nb_concepts=1000,
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
    batch_size=512,
)

# Load a small local causal LM as the labeler.
labeler_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
labeler_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-2B', torch_dtype=torch.bfloat16).to(device)
llm_interface = HuggingFaceLLM(labeler_model, labeler_tokenizer)

interpretations = LLMLabels(
    concept_explainer=concept_explainer,
    llm_interface=llm_interface,
    k_examples=20,
    k_context=5,
    concept_encoding_batch_size=512,
).interpret(
    inputs=inputs,
    latent_activations=activations,
    concepts_indices="all",
)
labels = {k: v for k, v in interpretations.items() if v}

sample = 'Alice and Bob enter the bar, then Alice offers a drink to Bob.'
encoded = splitter.tokenizer(sample, add_special_tokens=True)
sample_tokens = [t.replace("Ġ", " ") for t in splitter.tokenizer.convert_ids_to_tokens(encoded['input_ids'])]

local_importances = concept_explainer.concept_output_gradient(
    inputs=[sample], targets=None,
)[0].abs().sum(dim=1)

local_activations, _ = splitter.get_activations([sample], include_special_tokens=True)
concepts_activations = concept_explainer.activations_to_concepts(local_activations)

plot_concepts(
    concepts_activations=concepts_activations,
    concepts_importances=local_importances,
    concepts_labels=labels,
    sample=sample_tokens,
    top_k=10,
)
