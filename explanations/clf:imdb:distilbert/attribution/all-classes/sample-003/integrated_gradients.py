import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import IntegratedGradients, Granularity, plot_attributions

model_id = 'lvwerra/distilbert-imdb'
classes_names = ['negative', 'positive']

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = IntegratedGradients(model, tokenizer, granularity=Granularity.SENTENCE)

attributions = explainer(
    model_inputs='This is one of my 3 favorite movies. I\'ve been out on the water since I was 13, so I got a lot of the humor as well as recognizing a lot of the near-land scenery (the movie, although taking place in and around Virginia, was filmed around the San Francisco Bay), most notably the mothball fleet just east of the Benicia Bridge where Kelsey Grammar\'s character was first introduced to the USS Stingray, and the piers of San Francisco at the very end of the movie (including a boat that I\'ve worked on). As other people have said, the actors appeared to have fun making this movie as well as making it entertaining. The line "We\'re approaching the bottom, sir! I can hear a couple of lobsters duking it out" is, at least to me, priceless.<br /><br />I am one of numerous people who is anxiously awaiting a letterboxed DVD of Down Periscope to be introduced.',
    targets=torch.arange(len(classes_names))
)
plot_attributions(attributions[0], classes_names=classes_names)
