import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import VarGrad, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = VarGrad(model, tokenizer)

attributions = explainer(
    model_inputs='i love a hearty chat where i mean everything that i say and laugh from the heart gut w e not because i wanna let the person feel im entertained',
    targets=torch.arange(len(classes_names))
)
plot_attributions(attributions[0], classes_names=classes_names)
