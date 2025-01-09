import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
    classification_report
)

class MultiClassEvaluator:
    """
    A class to evaluate multi-class classification models using various metrics.

    Attributes:
        df (pd.DataFrame): DataFrame containing prediction scores for each class.
        label_col (str): Column name for the true class labels.
    """

    def __init__(self, df: pd.DataFrame, label_col: str):
        """
        Initialize the MultiClassEvaluator.

        Args:
            df (pd.DataFrame): DataFrame with prediction scores for each class.
            label_col (str): Column name containing the true class labels (indices).
        """
        self.df = df.copy()
        self.label_col = label_col
        self.true_labels = self.df[label_col].values
        self.pred_scores = self.df.drop(columns=[label_col]).values
        self.pred_classes = np.argmax(self.pred_scores, axis=1)

    def get_accuracy(self) -> float:
        """
        Compute the accuracy of the predictions.

        Returns:
            float: Accuracy score.
        """
        return accuracy_score(self.true_labels, self.pred_classes)
    
    def get_balanced_accuracy(self) -> float:
        """
        Compute the balanced accuracy of the predictions.

        Returns:
            float: Balanced accuracy score.
        """
        return balanced_accuracy_score(self.true_labels, self.pred_classes)

    def get_precision(self, average='macro') -> float:
        """
        Compute the precision of the predictions.

        Args:
            average (str): Type of averaging performed on the data. Defaults to 'macro'.

        Returns:
            float: Precision score.
        """
        return precision_score(self.true_labels, self.pred_classes, average=average)

    def get_recall(self, average='macro') -> float:
        """
        Compute the recall of the predictions.

        Args:
            average (str): Type of averaging performed on the data. Defaults to 'macro'.

        Returns:
            float: Recall score.
        """
        return recall_score(self.true_labels, self.pred_classes, average=average)

    def get_f1_score(self, average='macro') -> float:
        """
        Compute the F1 score of the predictions.

        Args:
            average (str): Type of averaging performed on the data. Defaults to 'macro'.

        Returns:
            float: F1 score.
        """
        return f1_score(self.true_labels, self.pred_classes, average=average)
    
    def get_f1_score_per_class(self) -> dict:
        """
        Compute the F1 score for each class using micro averaging.

        Returns:
            dict: F1 score for each class.
        """
        unique_classes = np.unique(self.true_labels)
        f1_scores = {}
        for cls in unique_classes:
            cls_true_labels = (self.true_labels == cls).astype(int)
            cls_pred_labels = (self.pred_classes == cls).astype(int)
            f1_scores[cls] = f1_score(cls_true_labels, cls_pred_labels, average='micro')
        return f1_scores

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Compute the confusion matrix for the predictions.

        Returns:
            np.ndarray: Confusion matrix.
        """
        return confusion_matrix(self.true_labels, self.pred_classes)

    def get_log_loss(self) -> float:
        """
        Compute the log loss of the predictions.

        Returns:
            float: Log loss.
        """
        return log_loss(self.true_labels, self.pred_scores)

    def get_classification_report(self) -> str:
        """
        Generate a classification report with precision, recall, F1-score, and support for each class.

        Returns:
            str: Classification report.
        """
        return classification_report(self.true_labels, self.pred_classes)

    def evaluate_all(self) -> dict:
        """
        Compute all available metrics and return as a dictionary.

        Returns:
            dict: Dictionary of metrics.
        """
        return {
            'accuracy': self.get_accuracy(),
            'balanced_accuracy': self.get_balanced_accuracy(),
            'f1_score_per_class': self.get_f1_score_per_class(),
            'f1_score_macro': self.get_f1_score(average='macro'),
            'confusion_matrix': self.get_confusion_matrix()
        }

# Example usage:
# df = pd.DataFrame({
#     'class_0': [0.1, 0.8, 0.1],
#     'class_1': [0.7, 0.1, 0.1],
#     'class_2': [0.2, 0.1, 0.8],
#     'true_label': [1, 0, 2]
# })
# evaluator = MultiClassEvaluator(df, label_col='true_label')
# print(evaluator.evaluate_all())