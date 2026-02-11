import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import SquareGrad, plot_attributions

model_id = 'arman1o1/roberta_ag_news_model'
classes_names = ['World', 'Sports', 'Business', 'Sci/Tech']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = SquareGrad(model, tokenizer)

attributions = explainer(model_inputs='SpaceShipOne Rolls Toward Victory MOJAVE, California -- A Southern California aerospace team took a big step toward capturing the \\$10 million Ansari X Prize Wednesday, but not without surviving a scary moment when the pilot found himself in a rapid spin as he roared across the threshold ', targets=torch.arange(len(classes_names)))
plot_attributions(attributions[0], classes_names=classes_names)
