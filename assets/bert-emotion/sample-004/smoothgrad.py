from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import SmoothGrad, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = []

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = SmoothGrad(model, tokenizer)

attributions = explainer(model_inputs='i get bored i get scared i feel ignored i feel happy i get silly i choke on my own words i make wishes i have dreams and i still want to believe anything can happen in this world for an ordinary girl a class profile link href http www')
plot_attributions(attributions[0], classes_names=classes_names)
