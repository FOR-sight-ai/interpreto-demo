import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import KernelShap, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = KernelShap(model, tokenizer)

attributions = explainer(
    model_inputs='i feel bitchy saying it but i think that next saturday i just want to be alone',
    targets=None
)
plot_attributions(attributions[0], classes_names=classes_names)
