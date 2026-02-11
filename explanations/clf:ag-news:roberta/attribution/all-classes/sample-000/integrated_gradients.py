import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import IntegratedGradients, plot_attributions

model_id = 'arman1o1/roberta_ag_news_model'
classes_names = ['World', 'Sports', 'Business', 'Sci/Tech']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = IntegratedGradients(model, tokenizer)

attributions = explainer(model_inputs='McTeer: Lonesome Dove to be an Aggie NEW YORK (CNN/Money) - A New Economy champion, a lover of the Texas picker poets who write lovesick country songs...and, oh, by the way, a member of the Federal Reserve system for 36 years.', targets=torch.arange(len(classes_names)))
plot_attributions(attributions[0], classes_names=classes_names)
