import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import SquareGrad, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = SquareGrad(model, tokenizer)

attributions = explainer(
    model_inputs='i feel his gracious presence even now',
    targets=torch.arange(len(classes_names))
)
plot_attributions(attributions[0], classes_names=classes_names)
