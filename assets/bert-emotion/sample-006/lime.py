from transformers import AutoTokenizer, AutoModelForSequenceClassification
from interpreto import Lime, plot_attributions

model_id = 'nateraw/bert-base-uncased-emotion'
classes_names = []

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
explainer = Lime(model, tokenizer)

attributions = explainer(model_inputs='i feel glad to be teaching nursery children who have special needs and know that the study of art has better helped me to use art in the curriculum to make lessons more enjoyable and interesting for the pupils')
plot_attributions(attributions[0], classes_names=classes_names)
