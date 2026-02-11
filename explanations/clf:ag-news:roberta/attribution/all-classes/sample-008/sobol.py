import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Sobol, plot_attributions

model_id = 'arman1o1/roberta_ag_news_model'
classes_names = ['World', 'Sports', 'Business', 'Sci/Tech']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Sobol(model, tokenizer)

attributions = explainer(model_inputs="Cardinals to Play Broncos Boise State accepts a bid Tuesday to play Louisville in the Liberty Bowl on Dec. 31, in a matchup of the nation's top two offenses.", targets=torch.arange(len(classes_names)))
plot_attributions(attributions[0], classes_names=classes_names)
