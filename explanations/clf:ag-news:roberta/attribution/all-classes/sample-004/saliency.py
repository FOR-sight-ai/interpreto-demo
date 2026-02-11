import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Saliency, plot_attributions

model_id = 'arman1o1/roberta_ag_news_model'
classes_names = ['World', 'Sports', 'Business', 'Sci/Tech']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Saliency(model, tokenizer)

attributions = explainer(model_inputs='Spawn of X Prize on Horizon Innovators take note: The folks behind the X Prize vow there will soon be more competitions in several disciplines. Also: The da Vinci team presses ahead in Canada.... Rubicon team plans another launch attempt. By Dan Brekke.', targets=torch.arange(len(classes_names)))
plot_attributions(attributions[0], classes_names=classes_names)
