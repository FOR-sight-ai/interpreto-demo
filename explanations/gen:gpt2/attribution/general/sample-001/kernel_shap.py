from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import KernelShap, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('gpt2')

explainer = KernelShap(model, tokenizer)
attributions = explainer(
    model_inputs='We called our library Interpreto is a good name? ',
    targets='“Interpreto” is a solid name: short, distinctive, and it strongly cues “interpretability.”'
)
plot_attributions(attributions[0])
