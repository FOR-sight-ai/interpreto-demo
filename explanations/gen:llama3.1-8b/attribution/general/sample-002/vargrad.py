from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import VarGrad, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B')

explainer = VarGrad(model, tokenizer)
attributions = explainer(
    model_inputs='Lorem ipsum dolor sit amet, ',
    targets='consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'
)
plot_attributions(attributions[0])
