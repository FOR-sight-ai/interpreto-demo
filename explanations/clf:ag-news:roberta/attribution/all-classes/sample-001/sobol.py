import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Sobol, plot_attributions

model_id = 'arman1o1/roberta_ag_news_model'
classes_names = ['World', 'Sports', 'Business', 'Sci/Tech']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Sobol(model, tokenizer)

attributions = explainer(model_inputs="Peru Gov't: Police Killed in Self-Defense Peru's interior minister said Wednesday that police acted in self-defense when they killed three coca farmers who were part of a group that hurled rocks and tried to burn a police lieutenant alive to protest U.S.-backed eradication of their cocaine producing crop.", targets=torch.arange(len(classes_names)))
plot_attributions(attributions[0], classes_names=classes_names)
