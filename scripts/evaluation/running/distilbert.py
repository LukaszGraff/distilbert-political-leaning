import os
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
import glob

class DistilbertPredictor:
    """
    A class used to run the DistilBERT model on a dataset.

    Attributes:
        model_path (str): The path to the pre-trained DistilBERT model.
        data_path (str): The path to the directory containing the test data.
        max_length (int): The maximum length of the input text for the tokenizer.
        batch_size (int): The batch size for the DataLoader.
    """

    def __init__(self, model_path, data_path, max_length=500, batch_size=16):
        """
        Initializes the DistilbertPredictor with the specified parameters.

        Args:
            model_path (str): The path to the pre-trained DistilBERT model.
            data_path (str): The path to the directory containing the test data.
            max_length (int, optional): The maximum length of the input text for the tokenizer. Defaults to 500.
            batch_size (int, optional): The batch size for the DataLoader. Defaults to 16.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=3)
        self.model.to(self.device)
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size

    def load_data(self):
        """
        Loads and tokenizes the test data.

        Returns:
            DataLoader: A DataLoader for the tokenized test dataset.
        """
        df_test = pd.read_csv(os.path.join(self.data_path, 'test.csv'))
        dataset = Dataset.from_dict({"post": df_test['post']})
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["post"])
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        return DataLoader(tokenized_dataset, batch_size=self.batch_size)

    def tokenize_function(self, examples):
        """
        Tokenizes the input examples.

        Args:
            examples (dict): A dictionary containing the input examples.

        Returns:
            dict: A dictionary containing the tokenized input examples.
        """
        return self.tokenizer(examples['post'], truncation=True, padding='max_length', max_length=self.max_length)

    def predict(self, dataloader):
        """
        Runs the model on the dataset and returns the predictions.

        Args:
            dataloader (DataLoader): A DataLoader for the tokenized dataset.

        Returns:
            np.ndarray: An array of predicted probabilities.
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(probs)
        return np.array(predictions)

# Example usage
if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'distilbert_500')
    model_path = glob.glob(os.path.join(model_dir, 'checkpoint-*'))[0]
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '500')
    
    predictor = DistilbertPredictor(model_path, data_path)
    test_dataloader = predictor.load_data()
    predictions = predictor.predict(test_dataloader)
    print(predictions)