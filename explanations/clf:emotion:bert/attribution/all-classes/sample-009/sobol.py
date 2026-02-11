import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Sobol, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Sobol(model, tokenizer)

attributions = explainer(
    model_inputs='im feeling really really sarcastic caustic or theres been an influx of idiots into my flists daily lives',
    targets=torch.arange(len(classes_names))
)
plot_attributions(attributions[0], classes_names=classes_names)
