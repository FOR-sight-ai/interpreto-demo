import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import Occlusion, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32)

explainer = Occlusion(model, tokenizer)
attributions = explainer(
    model_inputs='Lorem ipsum dolor sit amet, ',
    targets='consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
)

plot_attributions(attributions[0])
