import pandas as pd
import os

from multi_class_evaluator import MultiClassEvaluator
from sklearn.metrics import balanced_accuracy_score

def main():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results'))
    file_names = ['MostFrequent_predictions.csv', 'TfIDF_predictions.csv', 'ZeroShotClassifier_predictions.csv', 'DistilbertPredictor_predictions.csv']
    result_paths = [os.path.join(data_path, file_name) for file_name in file_names]

    df = evaluate_models(result_paths)

    df.to_csv('evaluation_results.csv')

def evaluate_models(result_paths):
    """
    Evaluates the models based on the prediction results.

    Args:
        result_paths (list): List of file paths containing the prediction results.

    Returns:
        pd.DataFrame: DataFrame indexed by model names with columns being the metric values.
    """
    results = {}
    for path in result_paths:
        df = pd.read_csv(path)
        model_name = os.path.splitext(os.path.basename(path))[0].replace('_predictions', '')
        evaluator = MultiClassEvaluator(df, 'political_leaning')
        metrics = evaluator.evaluate_all()  # dictionary of metrics
        results[model_name] = metrics

    return pd.DataFrame.from_dict(results, orient='index')

if __name__ == "__main__":
    main()