import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np


def load_data():
    input_ids = torch.load('../data/input_ids.pt')
    attention_masks = torch.load('../data/attention_masks.pt')
    labels = torch.load('../data/labels.pt')

    return train_test_split(input_ids, attention_masks, labels, test_size=0.1, random_state=42)


def train_model(train_data, val_data, epochs=4):
    train_inputs, train_masks, train_labels = train_data
    val_inputs, val_masks, val_labels = val_data

    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=16
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16
    )

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = batch

            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()

            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

    torch.save(model.state_dict(), '../models/bert_model.pth')


if __name__ == "__main__":
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = load_data()

    train_data = (train_inputs, train_masks, train_labels)
    val_data = (val_inputs, val_masks, val_labels)

    train_model(train_data, val_data, epochs=4)
