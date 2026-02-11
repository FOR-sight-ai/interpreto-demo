import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Occlusion, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Occlusion(model, tokenizer)

attributions = explainer(
    model_inputs='i have learned how to present in front of a class without feeling nervous',
    targets=torch.arange(len(classes_names))
)
plot_attributions(attributions[0], classes_names=classes_names)
