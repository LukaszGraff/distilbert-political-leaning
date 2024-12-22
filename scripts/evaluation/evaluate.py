from zero_shot_classifier import ZeroShotClassifier

import os
import pandas as pd

def run_zero_shot_classifier(data):
    classifier = ZeroShotClassifier()
    return classifier(data)

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    file_path = os.path.join(data_path, 'test.csv')
    data = pd.read_csv(file_path)

    scores = run_zero_shot_classifier(data['post'])

    results = pd.DataFrame(scores)
    results.rename(columns={'left-wing/socialism': 'left_score', 'right-wing/capitalism': 'right_score', 'centrism': 'center_score'}, inplace=True)
    results['political_leaning'] = data['political_leaning']

    results_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

    results.to_csv(os.path.join(results_path, 'zero_shot_results.csv'), index=False)
    

if __name__ == '__main__':
    main()