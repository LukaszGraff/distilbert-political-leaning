import pandas as pd
import numpy as np
import torch
import os
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

class DistilBertTrainer:
    """
    A class used to train a DistilBERT model for sequence classification.

    Attributes:
        model_name (str): The name of the pre-trained DistilBERT model.
        data_path (str): The path to the directory containing the training, validation, and test data.
        text_column (str): The name of the column containing the text data.
        label_column (str): The name of the column containing the label data.
        label_mapping (dict): A dictionary mapping label names to numerical values.
    """

    def __init__(self, model_name, data_path, text_column, label_column, label_mapping):
        """
        Initializes the DistilBertTrainer with the specified parameters.

        Args:
            model_name (str): The name of the pre-trained DistilBERT model.
            data_path (str): The path to the directory containing the training, validation, and test data.
            text_column (str): The name of the column containing the text data.
            label_column (str): The name of the column containing the label data.
            label_mapping (dict): A dictionary mapping label names to numerical values.
        """
        self.model_name = model_name
        self.data_path = data_path
        self.text_column = text_column
        self.label_column = label_column
        self.label_mapping = label_mapping
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))
        
        self.df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
        self.df_val = pd.read_csv(os.path.join(data_path, 'val.csv'))
        self.df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))
        
        self._map_labels()
        self.train_dataset = self._convert_to_dataset(self.df_train)
        self.val_dataset = self._convert_to_dataset(self.df_val)
        self.test_dataset = self._convert_to_dataset(self.df_test)

    def compute_metrics(self, pred):
        """
        Computes evaluation metrics for the model predictions.

        Args:
            pred (tuple): A tuple containing the logits and labels.

        Returns:
            dict: A dictionary containing the accuracy, balanced accuracy, and F1 score.
        """
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        balanced_accuracy = balanced_accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "f1": f1}

    def train(self, output_dir, epochs=3, batch_size=8):
        """
        Trains the DistilBERT model.

        Args:
            output_dir (str): The directory where the model checkpoints and logs will be saved.
            epochs (int, optional): The number of training epochs. Defaults to 3.
            batch_size (int, optional): The batch size for training and evaluation. Defaults to 8.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

    def _map_labels(self):
        """
        Maps the labels in the training, validation, and test data to numerical values.
        """
        self.df_train[self.label_column] = self.df_train["political_leaning"].map(self.label_mapping)
        self.df_val[self.label_column] = self.df_val["political_leaning"].map(self.label_mapping)
        self.df_test[self.label_column] = self.df_test["political_leaning"].map(self.label_mapping)

    def _convert_to_dataset(self, df):
        """
        Converts a pandas DataFrame to a Hugging Face Dataset and tokenizes the text data.

        Args:
            df (pd.DataFrame): The DataFrame to convert.

        Returns:
            Dataset: The tokenized Dataset.
        """
        def tokenize_function(examples):
            return self.tokenizer(examples[self.text_column], padding="max_length", truncation=True)
        dataset = Dataset.from_pandas(df)
        return dataset.map(tokenize_function, batched=True)

if __name__ == "__main__":
    model_name = "distilbert-base-uncased"
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '500')
    text_column = "post"
    label_column = "label"
    label_mapping = {"left": 0, "center": 1, "right": 2}
    
    distilbert_trainer = DistilBertTrainer(model_name, data_path, text_column, label_column, label_mapping)
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'distilbert_500')
    distilbert_trainer.train(output_dir)
