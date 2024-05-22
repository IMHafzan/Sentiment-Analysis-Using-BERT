from src.data_preprocessing import load_data, preprocess_data, tokenize_data, split_data
from transformers import BertTokenizer
import torch


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = load_data('../data/imdb-movies-dataset.csv')
    df, le = preprocess_data(df)
    input_ids, attention_masks, labels = tokenize_data(df, tokenizer, max_length=128)
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = split_data(input_ids, attention_masks,
                                                                                            labels)

    torch.save(train_inputs, '../data/train_inputs.pt')
    torch.save(val_inputs, '../data/val_inputs.pt')
    torch.save(train_masks, '../data/train_masks.pt')
    torch.save(val_masks, '../data/val_masks.pt')
    torch.save(train_labels, '../data/train_labels.pt')
    torch.save(val_labels, '../data/val_labels.pt')


if __name__ == "__main__":
    main()
