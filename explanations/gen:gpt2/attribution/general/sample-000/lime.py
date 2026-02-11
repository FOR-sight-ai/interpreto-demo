from transformers import AutoTokenizer, AutoModelForCausalLM
from interpreto import Lime, plot_attributions

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('gpt2')

explainer = Lime(model, tokenizer)
attributions = explainer(
    model_inputs='Alice and Bob enter the bar, ',
    targets='then Alice offers a drink to Bob.'
)
plot_attributions(attributions[0])
