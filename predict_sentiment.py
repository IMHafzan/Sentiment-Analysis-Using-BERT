import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'bert_model.pth')


print(f"Base directory: {base_dir}")
print(f"Model path: {model_path}")


if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")


model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


def predict_sentiment(review_text):
    inputs = tokenizer.encode_plus(
        review_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']


    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).item()


    label_map = {0: 'negative', 1: 'positive'}
    sentiment = label_map[predictions]

    return sentiment


if __name__ == "__main__":
    review_text = "The movie was trash"
    sentiment = predict_sentiment(review_text)
    print(f"Sentiment: {sentiment}")
