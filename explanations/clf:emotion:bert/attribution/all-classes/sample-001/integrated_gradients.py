import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import IntegratedGradients, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = IntegratedGradients(model, tokenizer)

attributions = explainer(
    model_inputs='im far ahead than the released tankouban that are sold here it just wont be the same anymore and the wait wont be as thrilling but damn me if i even feel slightly remorseful for that',
    targets=torch.arange(len(classes_names))
)
plot_attributions(attributions[0], classes_names=classes_names)
