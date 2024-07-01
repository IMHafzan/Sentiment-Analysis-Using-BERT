import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    df['label'] = df['Rating'].apply(lambda x: 1 if x >= 6 else 0)  #  Rating >= 6 is positive

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    return df, le


def tokenize_data(df, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for review in df['Review']:
        encoded_dict = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['label'].values)

    return input_ids, attention_masks, labels
