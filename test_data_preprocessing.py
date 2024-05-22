import unittest
import pandas as pd
from transformers import BertTokenizer
from src.data_preprocessing import load_data, preprocess_data, tokenize_data


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.file_path = '../data/imdb-movies-dataset.csv'
        self.df = load_data(self.file_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def test_load_data(self):
        df = load_data(self.file_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_preprocess_data(self):
        df, le = preprocess_data(self.df)
        self.assertIn('sentiment', df.columns)
        self.assertEqual(df['sentiment'].dtype, 'int32')
        self.assertEqual(len(le.classes_), 2)

    def test_tokenize_data(self):
        df, _ = preprocess_data(self.df)
        input_ids, attention_masks, labels = tokenize_data(df, self.tokenizer, max_length=128)
        self.assertEqual(input_ids.shape[0], df.shape[0])
        self.assertEqual(attention_masks.shape[0], df.shape[0])
        self.assertEqual(labels.shape[0], df.shape[0])
        self.assertEqual(input_ids.shape[1], 128)  # max_length
        self.assertEqual(attention_masks.shape[1], 128)


if __name__ == "__main__":
    unittest.main()
