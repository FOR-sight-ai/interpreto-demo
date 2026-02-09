from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Saliency, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = []

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Saliency(model, tokenizer)

attributions = explainer(model_inputs='i feel sorry for rafael bosch')
plot_attributions(attributions[0], classes_names=classes_names)
