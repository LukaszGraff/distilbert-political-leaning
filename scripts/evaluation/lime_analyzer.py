import os
import torch
from lime.lime_text import LimeTextExplainer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class LimeAnalyzer:
    """
    A class used to analyze text using LIME (Local Interpretable Model-agnostic Explanations) with a DistilBERT model.

    Attributes:
        model_path (str): The path to the pre-trained DistilBERT model.
        class_names (list): A list of class names for the classification task.
        max_length (int): The maximum length of the input text for the tokenizer.
        random_state (int, optional): The random state for reproducibility.
    """

    def __init__(self, model_path, class_names, max_length=500, random_state=None):
        """
        Initializes the LimeAnalyzer with the specified parameters.

        Args:
            model_path (str): The path to the pre-trained DistilBERT model.
            class_names (list): A list of class names for the classification task.
            max_length (int, optional): The maximum length of the input text for the tokenizer. Defaults to 500.
            random_state (int, optional): The random state for reproducibility. Defaults to None.
        """
        torch.cuda.empty_cache()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=len(class_names))
        self.model.to(self.device)
        self.explainer = LimeTextExplainer(class_names=class_names, random_state=random_state)
        self.max_length = max_length

    def predict_proba(self, texts):
        """
        Predicts the probabilities for the given texts using the DistilBERT model.

        Args:
            texts (list): A list of texts to predict.

        Returns:
            np.ndarray: An array of predicted probabilities.
        """
        inputs = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()
        return probs

    def explain_text(self, text, num_features=10, labels=[0, 1, 2]):
        """
        Generates an explanation for the given text using LIME.

        Args:
            text (str): The text to explain.
            num_features (int, optional): The number of features to include in the explanation. Defaults to 10.
            labels (list, optional): The list of labels to explain. Defaults to [0, 1, 2].

        Returns:
            lime.explanation.Explanation: The explanation object generated by LIME.
        """
        explanation = self.explainer.explain_instance(
            text,
            self.predict_proba,
            num_features=num_features,
            labels=labels
        )
        return explanation

    def save_explanation(self, explanation, file_path):
        """
        Saves the explanation to a file.

        Args:
            explanation (lime.explanation.Explanation): The explanation object to save.
            file_path (str): The path to the file where the explanation will be saved.
        """
        explanation.save_to_file(file_path)

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'distilbert_500', 'checkpoint-6607')
    class_names = ["left", "center", "right"]
    random_state = 42
    lime_analyzer = LimeAnalyzer(model_path, class_names, random_state=random_state)

    text = "I believe in free markets and individual liberty but I also believe that tolerance and equality are important."
    explanation = lime_analyzer.explain_text(text)
    lime_analyzer.save_explanation(explanation, 'explanations/explanation.html')