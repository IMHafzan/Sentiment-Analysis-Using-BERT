import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np


def load_data():
    input_ids = torch.load('../data/input_ids.pt')
    attention_masks = torch.load('../data/attention_masks.pt')
    labels = torch.load('../data/labels.pt')

    return train_test_split(input_ids, attention_masks, labels, test_size=0.1, random_state=42)


def evaluate_model():
    _, val_inputs, _, val_masks, _, val_labels = load_data()

    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=16
    )

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )

    model.load_state_dict(torch.load('../models/bert_model.pth'))
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in val_dataloader:
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        loss = outputs.loss
        logits = outputs.logits

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += np.sum(np.argmax(logits, axis=1) == label_ids) / len(label_ids)

    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print(f"Validation Accuracy: {avg_val_accuracy:.2f}")
    print(f"Validation Loss: {total_eval_loss / len(val_dataloader):.2f}")


if __name__ == "__main__":
    evaluate_model()
