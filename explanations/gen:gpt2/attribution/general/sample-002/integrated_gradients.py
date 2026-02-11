from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import IntegratedGradients, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('gpt2')

explainer = IntegratedGradients(model, tokenizer)
attributions = explainer(
    model_inputs='Lorem ipsum dolor sit amet, ',
    targets='consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'
)
plot_attributions(attributions[0])
