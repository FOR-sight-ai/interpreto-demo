from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import SmoothGrad, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')

explainer = SmoothGrad(model, tokenizer)
attributions = explainer(
    model_inputs='Alice and Bob enter the bar, ',
    targets='then Alice offers a drink to Bob.'
)
plot_attributions(attributions[0])
