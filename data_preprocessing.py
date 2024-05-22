import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    le = LabelEncoder()
    df['sentiment'] = le.fit_transform(df['sentiment'])
    return df, le


def tokenize_data(df, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for review in df['review']:
        encoded = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.tensor(df['sentiment'].values)


def split_data(input_ids, attention_masks, labels, test_size=0.2):
    return train_test_split(input_ids, attention_masks, labels, test_size=test_size)
