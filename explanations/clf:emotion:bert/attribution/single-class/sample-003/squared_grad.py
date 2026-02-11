import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import SquareGrad, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = SquareGrad(model, tokenizer)

attributions = explainer(
    model_inputs='im feeling good these days and my only complaints are that its getting harder and harder to move around and chase after stone and its getting harder and harder to find clothes that fit',
    targets=None
)
plot_attributions(attributions[0], classes_names=classes_names)
