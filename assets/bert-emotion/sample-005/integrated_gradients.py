from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import IntegratedGradients, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = []

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = IntegratedGradients(model, tokenizer)

attributions = explainer(model_inputs='im feeling generous lately spirit of after christmas maybe')
plot_attributions(attributions[0], classes_names=classes_names)
