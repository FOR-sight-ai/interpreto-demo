from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Lime, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = []

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Lime(model, tokenizer)

attributions = explainer(model_inputs='i feel these unwelcome guests beginning to take hold of me i will retreat to pray if but only for a moment')
plot_attributions(attributions[0], classes_names=classes_names)
