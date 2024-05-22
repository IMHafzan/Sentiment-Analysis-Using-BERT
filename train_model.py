import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


def load_data():
    train_inputs = torch.load('../data/train_inputs.pt')
    val_inputs = torch.load('../data/val_inputs.pt')
    train_masks = torch.load('../data/train_masks.pt')
    val_masks = torch.load('../data/val_masks.pt')
    train_labels = torch.load('../data/train_labels.pt')
    val_labels = torch.load('../data/val_labels.pt')

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    val_data = TensorDataset(val_inputs, val_masks, val_labels)

    return train_data, val_data


def train_model(train_data, val_data, batch_size=16, epochs=4):
    train_dataloader = DataLoader(
        train_data, sampler=RandomSampler(train_data), batch_size=batch_size
    )
    val_dataloader = DataLoader(
        val_data, sampler=SequentialSampler(val_data), batch_size=batch_size
    )

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_train_loss}")

    torch.save(model.state_dict(), '../models/bert_model.pth')
