import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import IntegratedGradients, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

explainer = IntegratedGradients(model, tokenizer)
attributions = explainer(
    model_inputs='i feel really selfish and feel guilty when i think about hurting myself',
    targets=torch.tensor([[0, 1, 2, 3, 4, 5]]),
)

plot_attributions(attributions[0], classes_names=classes_names)
