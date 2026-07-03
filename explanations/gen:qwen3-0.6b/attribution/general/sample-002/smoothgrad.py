import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import SmoothGrad, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float32)

explainer = SmoothGrad(model, tokenizer)
attributions = explainer(
    model_inputs='Lorem ipsum dolor sit amet, ',
    targets='consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
)

plot_attributions(attributions[0])
