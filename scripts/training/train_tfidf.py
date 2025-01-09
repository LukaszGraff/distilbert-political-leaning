from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
import pandas as pd


class TfIDFTrainer:
    """
    A baseline text classification model using TF-IDF for feature extraction and Logistic Regression for prediction.

    This model serves as a simple machine learning baseline for text classification tasks.
    """

    def __init__(self):
        """
        Initializes the model with a TF-IDF vectorizer and Logistic Regression classifier.
        """
        self.pipeline = make_pipeline(
            TfidfVectorizer(),
            StandardScaler(with_mean=False),  # Scale the data
            LogisticRegression(max_iter=500)
        )

    def train(self, X_train, y_train):
        """
        Fits the model to the training data.

        Parameters:
        - X_train (list): A list of text samples for training.
        - y_train (list): A list of corresponding labels for training.

        The method first vectorizes the text data using TF-IDF and then trains the Logistic Regression model.
        """
        self.pipeline.fit(X_train, y_train)

    def save_model(self, model_path):
        """
        Saves the model to the specified file path.

        Parameters:
        - model_path (str): The path to save the model to.
        """
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, model_path):
        """
        Loads the model from the specified file path.

        Parameters:
        - model_path (str): The path to load the model from.
        """
        with open(model_path, 'rb') as f:
            self.pipeline = pickle.load(f)



if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', '500'))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

    train_path = os.path.join(data_path, 'train.csv')
    df_train = pd.read_csv(train_path)

    # Map political leanings to numerical values
    mapping = {'left': 0, 'center': 1, 'right': 2}
    df_train['political_leaning'] = df_train['political_leaning'].map(mapping)

    # Extract posts and labels
    X_train = df_train["post"].tolist()
    y_train = df_train["political_leaning"].tolist()

    # Train the model
    trainer = TfIDFTrainer()
    trainer.train(X_train, y_train)
    trainer.save_model(output_path)
