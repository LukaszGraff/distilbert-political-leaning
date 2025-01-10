from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import os

from multi_class_evaluator import MultiClassEvaluator


def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'metrics'))
    file_names = ['MostFrequent_predictions.csv', 'TfIDF_predictions.csv', 'ZeroShotClassifier_predictions.csv', 'DistilbertPredictor_predictions.csv']
    result_paths = [os.path.join(results_dir, file_name) for file_name in file_names]
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', '500'))
    model_names = [os.path.splitext(os.path.basename(path))[0].replace('_predictions', '') for path in result_paths]

    # 500-word fragments
    result_dfs_500 = [pd.read_csv(path) for path in result_paths]
    df_500 = evaluate_models(result_dfs_500, model_names)
    df_500.to_csv(os.path.join(results_dir, 'evaluation_results_500.csv'))
    
    # average on 1500-word fragments
    result_dfs_1500 = [pd.read_csv(path) for path in result_paths]

    # Group into 5 consecutive rows (of one author) and collapse them
    test_path = os.path.join(data_path, 'test.csv')
    df_test = pd.read_csv(test_path)
    for i in range(len(result_dfs_1500)):
        result_dfs_1500[i]['author_ID'] = df_test['author_ID']
        result_dfs_1500[i] = result_dfs_1500[i].groupby('author_ID').apply(collapse_rows).reset_index(drop=True)
        result_dfs_1500[i].drop(columns=['author_ID'], inplace=True)

    df_1500 = evaluate_models(result_dfs_1500, model_names)
    df_1500.to_csv(os.path.join(results_dir, 'evaluation_results_1500.csv'))


def evaluate_models(dataframes, model_names):
    """
    Evaluates the models based on the prediction results.

    Args:
        dataframes (list): List of DataFrames containing the prediction results.

    Returns:
        pd.DataFrame: DataFrame indexed by model names with columns being the metric values.
    """
    results = {}
    for i, df in enumerate(dataframes):
        evaluator = MultiClassEvaluator(df, 'political_leaning')
        metrics = evaluator.evaluate_all()
        results[model_names[i]] = metrics

    return pd.DataFrame.from_dict(results, orient='index')

def collapse_rows(group):
    # Collapse every 5 consecutive rows
    collapsed = group.groupby(np.arange(len(group)) // 5).agg({
        'left_score': 'mean',
        'center_score': 'mean',
        'right_score': 'mean',
        'political_leaning': 'first',
        'author_ID': 'first'
    }).reset_index(drop=True)
    return collapsed

if __name__ == "__main__":
    main()