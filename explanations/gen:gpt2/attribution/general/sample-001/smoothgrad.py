from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import SmoothGrad, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('gpt2')

explainer = SmoothGrad(model, tokenizer)
attributions = explainer(
    model_inputs='We called our library Interpreto is a good name? ',
    targets='“Interpreto” is a solid name: short, distinctive, and it strongly cues “interpretability.”'
)
plot_attributions(attributions[0])
