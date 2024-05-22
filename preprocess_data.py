from transformers import BertTokenizer
from src.data_preprocessing import load_data, preprocess_data, tokenize_data
import torch


def main():
    # Load the dataset
    file_path = '../data/imdb-movies-dataset.csv'
    df = load_data(file_path)

    # Preprocess the data
    df, le = preprocess_data(df)

    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the data
    input_ids, attention_masks, labels = tokenize_data(df, tokenizer, max_length=128)

    # Save the tokenized data
    torch.save(input_ids, '../data/input_ids.pt')
    torch.save(attention_masks, '../data/attention_masks.pt')
    torch.save(labels, '../data/labels.pt')


if __name__ == "__main__":
    main()
