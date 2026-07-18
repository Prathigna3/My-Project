from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification
import torch

MODEL_NAME = "anwarvic/distilbert-base-uncased-for-fakenews"

# Use the specific tokenizer class
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs).item()
    return prediction, probs
