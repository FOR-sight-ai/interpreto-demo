import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from interpreto import SplitterForClassification, plot_concepts
from interpreto.concepts import ICAConcepts, LLMLabels

# ``HuggingFaceLLM`` is not (yet) shipped by interpreto — this import will
# start working once the class lands upstream. In the meantime, see the
# "Using your own LLM interface" section of the classification concept
# tutorial for the reference implementation you can paste in here.
from interpreto.commons import HuggingFaceLLM

device = "cuda" if torch.cuda.is_available() else "cpu"

splitter = SplitterForClassification('lvwerra/distilbert-imdb', batch_size=64, device_map=device)
inputs = load_dataset('stanfordnlp/imdb')['train']["text"]
classes_names = ['negative', 'positive']

activations, _ = splitter.get_activations(inputs, forward_kwargs={'truncation': True})

concept_explainer = ICAConcepts(
    splitter,
    nb_concepts=30,
    device=device,
)

concept_explainer.fit(
    activations,
    max_iter=5000,
)

# Load a small local causal LM as the labeler.
labeler_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
labeler_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-2B', torch_dtype=torch.bfloat16).to(device)
llm_interface = HuggingFaceLLM(labeler_model, labeler_tokenizer)

llm_labels = LLMLabels(
    concept_explainer=concept_explainer,
    llm_interface=llm_interface,
    k_examples=20,
)
labels = {k: v for k, v in llm_labels.interpret(inputs=inputs, latent_activations=activations, concepts_indices="all").items() if v}

gradients = concept_explainer.concept_output_gradient(
    inputs=activations,
    targets=None,
    batch_size=64,
)
mean_gradients = torch.stack(gradients).abs().squeeze().mean(0)

plot_concepts(
    classes_names=classes_names,
    concepts_importances=mean_gradients,
    concepts_labels=labels,
    top_k=10,
)
