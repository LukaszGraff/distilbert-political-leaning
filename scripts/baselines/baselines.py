import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

class RandomGuessing:
    """
    A baseline model that makes random predictions based on the distribution of labels in the training data.

    This model uses the class distribution from the training set to randomly predict labels for a given test set.
    It serves as a simple benchmark for classification tasks to compare against more sophisticated models.
    """

    def __init__(self):
        """
        Initializes the RandomGuessing model.

        Attributes:
        - class_labels (list): The unique class labels identified from the training data.
        - class_probabilities (list): The corresponding probabilities of each class based on the distribution in the training data.
        """
        self.class_labels = None
        self.class_probabilities = None

    def train(self, y_train):
        """
        Trains the RandomGuessing model by calculating the class distribution from the training labels.

        Parameters:
        - y_train (list): A list of training labels used to determine the class distribution.

        This method sets the `class_labels` and `class_probabilities` attributes based on the frequency of each label in `y_train`.
        """
        label_counts = Counter(y_train)
        total_count = sum(label_counts.values())
        self.class_labels = list(label_counts.keys())
        self.class_probabilities = [count / total_count for count in label_counts.values()]

    def predict(self, X_test):
        """
        Makes random predictions for the given test set based on the learned class distribution.

        Parameters:
        - X_test (list): A list of test samples. The content of the samples is not used; only the length of the list matters.

        Returns:
        - np.ndarray: A NumPy array containing the randomly predicted labels for each sample in `X_test`.
        """
        return np.random.choice(self.class_labels, size=len(X_test), p=self.class_probabilities)

class MostFrequent:
    """
    A baseline model that always predicts the most frequent label from the training data.

    This model is useful as a simple benchmark for classification tasks, particularly for imbalanced datasets.
    It predicts the same label for all test samples, which is the label that appears most frequently in the training set.
    """

    def __init__(self):
        """
        Initializes the MostFrequent model.

        Attributes:
        - most_common_label (str or int): The most frequent label in the training data.
        """
        self.most_common_label = None

    def train(self, y_train):
        """
        Trains the MostFrequent model by identifying the most frequent label in the training data.

        Parameters:
        - y_train (list): A list of training labels used to determine the most common label.

        This method sets the `most_common_label` attribute to the label that appears most frequently in `y_train`.
        """
        self.most_common_label = Counter(y_train).most_common(1)[0][0]

from collections import Counter
import numpy as np

class MostFrequent:
    """
    A baseline model that always predicts the most frequent label from the training data.

    This model is useful as a simple benchmark for classification tasks, particularly for imbalanced datasets.
    It predicts the same label for all test samples, which is the label that appears most frequently in the training set.
    """

    def __init__(self):
        """
        Initializes the MostFrequent model.

        Attributes:
        - most_common_label (str or int): The most frequent label in the training data.
        """
        self.most_common_label = None

    def train(self, y_train):
        """
        Trains the MostFrequent model by identifying the most frequent label in the training data.

        Parameters:
        - y_train (list): A list of training labels used to determine the most common label.

        This method sets the `most_common_label` attribute to the label that appears most frequently in `y_train`.
        """
        self.most_common_label = Counter(y_train).most_common(1)[0][0]

    def predict(self, X_test):
        """
        Makes predictions for the given test set by returning the most frequent label for each test sample.

        Parameters:
        - X_test (list): A list of test samples. The content of the samples is not used; only the length of the list matters.

        Returns:
        - np.ndarray: A NumPy array containing the most frequent label predicted for each sample in `X_test`.
        """
        return np.full(len(X_test), self.most_common_label)

class TfIDF:
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

    def predict(self, X_test):
        """
        Makes predictions for the given test set.

        Parameters:
        - X_test (list): A list of text samples for testing.

        Returns:
        - np.ndarray: Predicted labels for the test samples.
        """
        return self.pipeline.predict(X_test)


df_train = pd.read_csv("data/processed/train.csv")
df_test = pd.read_csv("data/processed/test.csv")

# Extract posts and labels
X_train = df_train["post"].tolist()
y_train = df_train["political_leaning"].tolist()
X_test = df_test["post"].tolist()
y_test = df_test["political_leaning"].tolist()

# # Train and test RandomGuessing
# model = RandomGuessing()
# model.train(y_train)
# predictions = model.predict(X_test)
# # Evaluate the predictions
# accuracy = accuracy_score(y_test, predictions)
# print("RandomGuessing accuracy:", accuracy)

# # Train and test MostFrequent 
# model = MostFrequent()
# model.train(y_train)
# predictions = model.predict(X_test)
# # Evaluate the predictions
# accuracy = accuracy_score(y_test, predictions)
# print("MostFrequent accuracy:", accuracy)

# Train and test TfIDF
model = TfIDF()
model.train(X_train, y_train)
predictions = model.predict(X_test)
# Evaluate the predictions
accuracy = accuracy_score(y_test, predictions)
print("TfIDF accuracy:", accuracy)

    
