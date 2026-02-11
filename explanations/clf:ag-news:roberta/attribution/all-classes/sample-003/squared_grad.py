import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import SquareGrad, plot_attributions

model_id = 'arman1o1/roberta_ag_news_model'
classes_names = ['World', 'Sports', 'Business', 'Sci/Tech']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = SquareGrad(model, tokenizer)

attributions = explainer(model_inputs='Cards unfazed by Series deficit Monday #39;s workout at Busch Stadium contained a few more St. Louis Cardinals than you #39;d expect considering it was optional, but you could understand why they #39;d want to ', targets=torch.arange(len(classes_names)))
plot_attributions(attributions[0], classes_names=classes_names)
