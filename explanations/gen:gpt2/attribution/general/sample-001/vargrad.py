import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import VarGrad, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32)

explainer = VarGrad(model, tokenizer)
attributions = explainer(
    model_inputs='We called our library Interpreto is a good name? ',
    targets='“Interpreto” is a solid name: short, distinctive, and it strongly cues “interpretability.”',
)

plot_attributions(attributions[0])
