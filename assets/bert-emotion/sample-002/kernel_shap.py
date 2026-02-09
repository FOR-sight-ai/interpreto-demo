from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import KernelShap, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = []

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = KernelShap(model, tokenizer)

attributions = explainer(model_inputs='i feel disappointed by myself')
plot_attributions(attributions[0], classes_names=classes_names)
