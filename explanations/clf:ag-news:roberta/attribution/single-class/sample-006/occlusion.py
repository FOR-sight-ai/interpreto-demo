import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Occlusion, plot_attributions

model_id = 'arman1o1/roberta_ag_news_model'
classes_names = ['World', 'Sports', 'Business', 'Sci/Tech']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Occlusion(model, tokenizer)

attributions = explainer(model_inputs="Prosecutor seeks 8 years in jail for Berlusconi  MILAN -- An Italian prosecutor asked a court yesterday to sentence Silvio Berlusconi to eight years in jail for bribing judges as the prime minister's four-year corruption trial reached its closing stages.", targets=None)
plot_attributions(attributions[0], classes_names=classes_names)
