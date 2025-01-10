import os
import glob
import pandas as pd
import torch

from lime_analyzer import LimeAnalyzer

def main():
    result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'DistilbertPredictor_predictions.csv'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'explanations'))
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', '500', 'test.csv'))
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'distilbert_500')
    model_path = glob.glob(os.path.join(model_dir, 'checkpoint-*'))[0]

    class_names = ["left", "center", "right"]
    
    column_mapping = {'left_score': 0, 'center_score': 1, 'right_score': 2}
    label_mapping = {0: 'left', 1: 'center', 2: 'right'}

    random_state = 42

    analyzer = LimeAnalyzer(model_path, class_names, random_state=random_state)

    df = pd.read_csv(result_path)
    df['post'] = pd.read_csv(data_path)['post']
    df['pred'] = df[['left_score', 'center_score', 'right_score']].idxmax(axis=1).map(column_mapping)


    # # 5 random instances
    k_rand = 5
    random_posts = list(df.sample(n=k_rand, random_state=random_state)['post'])
    names = [f'random_post_{i}.html' for i in range(k_rand)]
    explain_texts(random_posts, analyzer, output_dir, names)

    k_top = 3
    # # Top 3 confidence and correct for each class
    df_correct = df[df['pred'] == df['political_leaning']]
    for pred in ['left_score', 'center_score', 'right_score']:
        top_correct_posts = list(df_correct.nlargest(k_top, pred)['post'])
        names = [f'{pred}_correct_{i}.html' for i in range(k_top)]
        explain_texts(top_correct_posts, analyzer, output_dir, names)

    # Top 3 confidence and incorrect for each class
    df_incorrect = df[df['pred'] != df['political_leaning']]
    for pred in ['left_score', 'center_score', 'right_score']:
        top_incorrect_posts = list(df_incorrect.nlargest(k_top, pred)['post'])
        labels = [label_mapping[x] for x in df_incorrect.nlargest(k_top, pred)['political_leaning']]
        pred_name = pred.split('_')[0]
        names = [f'{pred_name}_but_was_{labels[i]}_{i}.html' for i in range(k_top)]
        explain_texts(top_incorrect_posts, analyzer, output_dir, names)

def explain_texts(texts, analyzer, output_dir, names):
    for i, text in enumerate(texts):
        torch.cuda.empty_cache()

        explanation = analyzer.explain_text(text)
        analyzer.save_explanation(explanation, os.path.join(output_dir, names[i]))


if __name__ == "__main__":
    main()