import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Lime, plot_attributions

model_id = 'lvwerra/distilbert-imdb'
classes_names = ['negative', 'positive']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Lime(model, tokenizer)

attributions = explainer(model_inputs='Oh boy ! It was just a dream ! What a great idea ! Mr Lynch is very lucky most people try to tell classical stories. This way he can play with his little plantings and his even more little payoffs. Check out Polanski\'s "The lodger" for far more intelligent mix of fantasy and reality.', targets=torch.arange(len(classes_names)))
plot_attributions(attributions[0], classes_names=classes_names)
