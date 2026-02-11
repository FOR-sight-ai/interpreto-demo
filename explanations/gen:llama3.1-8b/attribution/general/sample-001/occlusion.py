from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import Occlusion, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B')

explainer = Occlusion(model, tokenizer)
attributions = explainer(
    model_inputs='We called our library Interpreto is a good name? ',
    targets='“Interpreto” is a solid name: short, distinctive, and it strongly cues “interpretability.”'
)
plot_attributions(attributions[0])
