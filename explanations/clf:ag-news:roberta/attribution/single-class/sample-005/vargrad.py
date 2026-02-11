import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import VarGrad, plot_attributions

model_id = 'arman1o1/roberta_ag_news_model'
classes_names = ['World', 'Sports', 'Business', 'Sci/Tech']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = VarGrad(model, tokenizer)

attributions = explainer(model_inputs="Myskina, Kuznetsov to Play in Fed Cup (AP) AP - Anastasia Myskina and Svetlana Kuznetsova will lead Russia's Fed Cup team when it plays Austria in this month's semifinals. Defending champion France will feature Amelie Mauresmo and Mary Pierce in the other semifinal against Spain, which has won this event five times.", targets=None)
plot_attributions(attributions[0], classes_names=classes_names)
