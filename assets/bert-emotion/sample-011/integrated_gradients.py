from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import IntegratedGradients, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = []

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = IntegratedGradients(model, tokenizer)

attributions = explainer(model_inputs='i have good camwhore skill thanks to instagram and pudding which is anotehr super popular social apps to post all your vain picture without feeling vain because others will do the same so ftw')
plot_attributions(attributions[0], classes_names=classes_names)
