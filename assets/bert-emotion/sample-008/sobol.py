from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Sobol, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = []

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Sobol(model, tokenizer)

attributions = explainer(model_inputs='i feel like were in this together and im glad for that')
plot_attributions(attributions[0], classes_names=classes_names)
