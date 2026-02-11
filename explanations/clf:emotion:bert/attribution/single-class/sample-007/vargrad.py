import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import VarGrad, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = VarGrad(model, tokenizer)

attributions = explainer(
    model_inputs='i feel and talk like a disadvantaged child and am waiting for half my face to come back to me',
    targets=None
)
plot_attributions(attributions[0], classes_names=classes_names)
