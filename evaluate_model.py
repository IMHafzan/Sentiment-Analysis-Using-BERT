import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    val_inputs = torch.load('../data/val_inputs.pt')
    val_masks = torch.load('../data/val_masks.pt')
    val_labels = torch.load('../data/val_labels.pt')

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    return val_data


def evaluate_model(val_data, batch_size=16):
    val_dataloader = DataLoader(
        val_data, sampler=SequentialSampler(val_data), batch_size=batch_size
    )

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )

    model.load_state_dict(torch.load('../models/bert_model.pth'))
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    predictions, true_labels = [], []

    for batch in val_dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
