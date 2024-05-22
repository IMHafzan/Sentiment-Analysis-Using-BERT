import unittest
import torch
from transformers import BertForSequenceClassification
from src.train_model import train_model, load_data


class TestTrainModel(unittest.TestCase):

    def setUp(self):
        self.train_data, self.val_data = load_data()

    def test_load_data(self):
        train_data, val_data = load_data()
        self.assertIsInstance(train_data, torch.utils.data.TensorDataset)
        self.assertIsInstance(val_data, torch.utils.data.TensorDataset)

    def test_train_model(self):
        # Load the data
        train_data, val_data = load_data()

        # Train the model
        train_model(train_data, val_data, epochs=1)

        # Load the trained model
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        model.load_state_dict(torch.load('../models/bert_model.pth'))

        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
