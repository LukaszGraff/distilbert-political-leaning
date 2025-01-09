import numpy as np
from collections import Counter
from sklearn.pipeline import make_pipeline
from transformers import pipeline
import torch
from datasets import Dataset
import tqdm
import pickle

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
        Makes predictions for the given test set.

        Parameters:
        - X_test (list): A list of text samples for testing.

        Returns:
        - np.ndarray: Predicted probability distributions for the test samples.
        """
        num_samples = len(X_test)
        return np.array([[self.class_probabilities[self.class_labels.index(label)] for label in self.class_labels] for _ in range(num_samples)])

class MostFrequent:
    """
    A baseline model that always predicts the most frequent class from the training data.

    This model serves as a simple benchmark for classification tasks to compare against more sophisticated models.
    """

    def __init__(self):
        """
        Initializes the MostFrequent model.

        Attributes:
        - most_frequent_class (str): The most frequent class label identified from the training data.
        - class_labels (list): The unique class labels identified from the training data.
        """
        self.most_frequent_class = None
        self.class_labels = None

    def train(self, y_train):
        """
        Trains the MostFrequent model by identifying the most frequent class from the training labels.

        Parameters:
        - y_train (list): A list of training labels used to determine the most frequent class.

        This method sets the `most_frequent_class` attribute based on the frequency of each label in `y_train`.
        """
        label_counts = Counter(y_train)
        self.most_frequent_class = label_counts.most_common(1)[0][0]
        self.class_labels = list(label_counts.keys())

    def predict(self, X_test):
        """
        Makes predictions for the given test set.

        Parameters:
        - X_test (list): A list of text samples for testing.

        Returns:
        - np.ndarray: Predicted probability distributions for the test samples.
        """
        num_samples = len(X_test)
        return np.array([[1.0 if label == self.most_frequent_class else 0.0 for label in self.class_labels] for _ in range(num_samples)])

class TfIDF:
    """
    A baseline text classification model using TF-IDF for feature extraction and Logistic Regression for prediction.

    This model serves as a simple machine learning baseline for text classification tasks.
    """
    def __init__(self, model_path=None):
        """
        Initializes the model from
        """
        self.pipeline = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Loads the model from the specified file path.

        Parameters:
        - model_path (str): The path to load the model from.
        """
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def predict(self, X_test):
        """
        Makes predictions for the given test set.

        Parameters:
        - X_test (list): A list of text samples for testing.

        Returns:
        - np.ndarray: Predicted probability distributions for the test samples.
        """
        return self.pipeline.predict_proba(X_test)

class ZeroShotClassifier:
    """
    A class used to perform zero-shot classification using a pre-trained model.

    Attributes:
        labels (list): A list of candidate labels for classification.
        hypothesis_template (str): The hypothesis template for zero-shot classification.
    """

    def __init__(self, labels=["left-wing/socialism", "centrism", "right-wing/capitalism"], hypothesis_template="The author of this posts leans towards {}."):
        """
        Initializes the ZeroShotClassifier with the specified parameters.

        Args:
            labels (list, optional): A list of candidate labels for classification. Defaults to ["left-wing/socialism", "centrism", "right-wing/capitalism"].
            hypothesis_template (str, optional): The hypothesis template for zero-shot classification. Defaults to "The author of this posts leans towards {}.".
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
        self.labels = labels
        self.hypothesis_template = hypothesis_template

    def __call__(self, posts):
        """
        Classifies the given posts using zero-shot classification.

        Args:
            posts (list): A list of text posts to classify.

        Returns:
            np.ndarray: Predicted probability distributions for the test samples.
        """
        dataset = Dataset.from_dict({"post": posts})
        results = self.classifier(dataset["post"], candidate_labels=self.labels, hypothesis_template=self.hypothesis_template, batch_size=8)

        scores = [[result['scores'][result['labels'].index(label)] for label in self.labels] for result in results]

        return np.array(scores)
    
class LLamaBatchPredictor:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def classify_texts_batch(self, texts, batch_size=8):
        predictions = []
        with tqdm.tqdm(total=len(texts), desc="Predicting", unit="sample") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                for text in batch:
                    outputs = self.pipe(
                        [
                            {
                                "role": "system",
                                "content": "Classify the political leaning of the following text as 'left', 'right', or 'center'. Only respond with one of these three words.",
                            },
                            {"role": "user", "content": f"Text: {text}"},
                        ],
                        max_new_tokens=5,
                        pad_token_id=128001
                    )
                    prediction = self._postprocess_output(outputs)
                    predictions.append(prediction)
                    pbar.update(1)
        return self._postprocess_output(predictions)


    def _postprocess_output(self, output):
        if isinstance(output, list):
            # Extract the assistant's message
            for message in output[0]['generated_text']:
                if message['role'] == 'assistant':
                    return message['content'].lower().strip()
        return "unknown"

    def _map_to_labels(self, text):
        """
        Maps the generated text to the corresponding label probabilities.

        Args:
            text (str): The generated text from the LLaMA model.

        Returns:
            list: The probability distribution for the labels [left, center, right].
        """
        if "left" in text:
            return [1.0, 0.0, 0.0]
        elif "center" in text:
            return [0.0, 1.0, 0.0]
        elif "right" in text:
            return [0.0, 0.0, 1.0]
        else:
            return [0.0, 0.0, 0.0]