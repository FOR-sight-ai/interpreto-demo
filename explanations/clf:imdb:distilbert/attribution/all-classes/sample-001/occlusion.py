import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Occlusion, Granularity, plot_attributions

model_id = 'lvwerra/distilbert-imdb'
classes_names = ['negative', 'positive']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Occlusion(model, tokenizer, granularity=Granularity.SENTENCE)

attributions = explainer(
    model_inputs="When the Italians and Miles O'keeffe work together nothing can go wrong! As ever, Miles is great as the almost as great Ator; the most lovable barbarian of all times. Totally lives up to the first movie.",
    targets=torch.arange(len(classes_names))
)
plot_attributions(attributions[0], classes_names=classes_names)
