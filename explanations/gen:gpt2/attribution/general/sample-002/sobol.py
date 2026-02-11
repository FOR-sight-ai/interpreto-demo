from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import Sobol, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('gpt2')

explainer = Sobol(model, tokenizer)
attributions = explainer(
    model_inputs='Lorem ',
    targets='ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'
)
plot_attributions(attributions[0])
