import os
import pandas as pd
import json
import glob

from baselines import RandomGuessing, MostFrequent, TfIDF, ZeroShotClassifier, LLamaBatchPredictor
from distilbert import DistilbertPredictor
from sklearn.metrics import accuracy_score

def main():
    # We use the processed data (500-word fragments)
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', '500'))

    test_path = os.path.join(data_path, 'test.csv')
    df_test = pd.read_csv(test_path)

    # For the majority baseline
    train_path = os.path.join(data_path, 'train.csv')
    df_train = pd.read_csv(train_path)

    # Map political leanings to numerical values
    mapping = {'left': 0, 'center': 1, 'right': 2}
    df_train['political_leaning'] = df_train['political_leaning'].map(mapping)
    df_test['political_leaning'] = df_test['political_leaning'].map(mapping)

    # Extract posts and labels
    X_train = df_train["post"].tolist()
    y_train = df_train["political_leaning"].tolist()

    X_test = df_test["post"].tolist()
    y_test = df_test["political_leaning"].tolist()

    #models = ['RandomGuessing', 'MostFrequent', 'TfIDF', 'ZeroShotClassifier', 'LLamaBatchPredictor', 'DistilBERT']
    models = ['ZeroShotClassifier']

    run_models(models=models, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, data_path=data_path)

def run_models(models, X_train, y_train, X_test, y_test, data_path):
    """
    Runs the specified models on the given dataset and saves the results.

    Args:
        models (list): A list of model names to run.
        X_train (list): The training data (posts).
        y_train (list): The training labels (political leanings).
        X_test (list): The test data (posts).
        y_test (list): The test labels (political leanings).
        data_path (str): The path to the directory containing the data files.

    Returns:
        None
    """
    for model_name in models:
        if model_name == 'RandomGuessing':
            model = RandomGuessing()
            model.train(y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions.argmax(axis=1))
            print("RandomGuessing accuracy:", accuracy)
            save_predictions(predictions, y_test, model_name)
        elif model_name == 'MostFrequent':
            model = MostFrequent()
            model.train(y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions.argmax(axis=1))
            print("MostFrequent accuracy:", accuracy)
            save_predictions(predictions, y_test, model_name)
        elif model_name == 'TfIDF':
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'tfidf_model.pkl'))
            model = TfIDF(model_path)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions.argmax(axis=1))
            print("TfIDF accuracy:", accuracy)
            save_predictions(predictions, y_test, model_name)
        elif model_name == 'ZeroShotClassifier':
            model = ZeroShotClassifier()
            predictions = model(X_test)
            accuracy = accuracy_score(y_test, predictions.argmax(axis=1))
            print("ZeroShotClassifier accuracy:", accuracy)
            save_predictions(predictions, y_test, model_name)
        elif model_name == 'LLamaBatchPredictor':
            model = LLamaBatchPredictor()
            predictions = model.classify_texts_batch(X_test)
            accuracy = accuracy_score(y_test, predictions.argmax(axis=1))
            print("LLamaBatchPredictor accuracy:", accuracy)
            save_predictions(predictions, y_test, model_name)
        elif model_name == 'DistilBERT':
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'distilbert_500'))
            model_path = glob.glob(os.path.join(model_dir, 'checkpoint-*'))[0]
            predictor = DistilbertPredictor(model_path, data_path)
            test_dataloader = predictor.load_data()
            predictions = predictor.predict(test_dataloader)
            accuracy = accuracy_score(y_test, predictions.argmax(axis=1))
            print("DistilBERT accuracy:", accuracy)
            save_predictions(predictions, y_test, model_name)

def save_predictions(predictions, y_test, model_name):
    """
    Saves the predictions to a CSV file with columns left_score, right_score, center_score, and political_leaning.

    Args:
        predictions (np.ndarray): The predicted probabilities.
        y_test (list): The true labels.
        model_name (str): The name of the model.
    """
    results_dir =  os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'{model_name}_predictions.csv')

    df = pd.DataFrame(predictions, columns=['left_score', 'center_score', 'right_score'])
    df['political_leaning'] = y_test
    df.to_csv(results_file, index=False)

# Example usage
if __name__ == "__main__":
    main()