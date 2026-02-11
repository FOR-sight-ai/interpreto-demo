from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import GradientShap, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B')

explainer = GradientShap(model, tokenizer)
attributions = explainer(
    model_inputs='Alice and Bob enter the bar, ',
    targets='then Alice offers a drink to Bob.'
)
plot_attributions(attributions[0])
