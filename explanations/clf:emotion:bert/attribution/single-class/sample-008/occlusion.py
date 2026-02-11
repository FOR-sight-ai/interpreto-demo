import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Occlusion, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Occlusion(model, tokenizer)

attributions = explainer(
    model_inputs='i feel that i am afraid of whatever ad anything that will happen and idc is it good or bad i am just afraid and i hope god you will help me in whatever i do',
    targets=None
)
plot_attributions(attributions[0], classes_names=classes_names)
