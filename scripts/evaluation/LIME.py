
#%% [Cell 1] - Imports and DataHandler
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Simple DataHandler for loading CSV and splitting data
class DataHandler:

    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Loading data from {file_path}...")
        self.data = pd.read_csv(file_path)
        print("Data loaded successfully.")
    
    def basic_preprocess(self, text_column="post"):
        """
        (Optional) Minimal text cleaning or transformations.
        """
        self.data[text_column] = self.data[text_column].fillna("").astype(str)
        
    def split_data(self, text_column="post", label_column="label",
                   test_size=0.2, random_state=42):
        """
        Simple train/test split. Returns X_train, X_test, y_train, y_test as lists.
        """
        X = self.data[text_column].tolist()
        y = self.data[label_column].tolist()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

#%% [Cell 2] - Model Loading & Prediction Function
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file

model_name_or_path = "distilbert-base-uncased"  # Public model name
safetensors_path="C:/Users/tomek/Downloads/qlora-political-leaning-master/distilbert_politics/model.safetensors",

def load_distilbert_model(model_name_or_path: str,
                          safetensors_path: str,
                          num_labels: int = 3):
    """
    Load DistilBERT (config + tokenizer) from a local dir or huggingface model name.
    Then load the .safetensors weights.
    """
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels
    )
    
    # Load .safetensors state_dict
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
    model.eval()
    return tokenizer, model

def predict_proba(texts, tokenizer, model):
    """
    Takes a list of text strings -> returns NxC prob array (C=number of classes).
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

#%% [Cell 3] - LIME Explanation
from lime.lime_text import LimeTextExplainer

def run_lime_explanations(
    csv_path="C:/Users/tomek/Downloads/qlora-political-leaning-master/political_leaning.csv",
    safetensors_path="C:/Users/tomek/Downloads/qlora-political-leaning-master/distilbert_politics/model.safetensors",
    model_name_or_path="distilbert-base-uncased",
    num_labels=3,
    text_column="post",
    label_column="label",
    class_names=None,
    test_size=0.2,
    random_state=42,
    n_samples_to_explain=3
):
    """
    Loads data, splits it, loads DistilBERT model + weights,
    and runs LIME on a few test samples.
    """
    # 1. Load/Preprocess data
    data_handler = DataHandler(csv_path)
    data_handler.basic_preprocess(text_column=text_column)
    
    X_train, X_test, y_train, y_test = data_handler.split_data(
        text_column=text_column,
        label_column=label_column,
        test_size=test_size,
        random_state=random_state
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # 2. Load model + tokenizer
    tokenizer, model = load_distilbert_model(
        model_name_or_path=model_name_or_path,
        safetensors_path=safetensors_path,
        num_labels=num_labels
    )
    
    # 3. Setup LIME
    if class_names is None:
        # If not provided, just guess class names
        class_names = [f"class_{i}" for i in range(num_labels)]
    
    explainer = LimeTextExplainer(class_names=class_names)
    
    # 4. Pick samples to explain
    X_samples = X_test[-n_samples_to_explain:]
    y_samples = y_test[-n_samples_to_explain:]
    
    # 5. Generate explanations
    for i, sample_text in enumerate(X_samples):
        explanation = explainer.explain_instance(
            sample_text,
            lambda txts: predict_proba(txts, tokenizer, model),
            labels=list(range(num_labels)),
            num_features=10
        )
        
        print(f"\n{'='*60}")
        print(f"Sample {i+1} of {n_samples_to_explain}")
        print(f"Text: {sample_text}")
        print(f"True Label Index: {y_samples[i]} -> {class_names[y_samples[i]]}")
        
        # Explanation output
        explanation.save_to_file(f"lime_explanation_{i}.html")
        print(f"Explanation saved to 'lime_explanation_{i}.html'")

#%% [Cell 4] - Main Entry Point
if __name__ == "__main__":
    run_lime_explanations(
        csv_path="C:/Users/tomek/Downloads/qlora-political-leaning-master/political_leaning.csv",
        safetensors_path="C:/Users/tomek/Downloads/qlora-political-leaning-master/distilbert_politics/model.safetensors",
        model_name_or_path="distilbert-base-uncased",
        num_labels=3,  # e.g., for left/center/right
        text_column="post",
        label_column="political_leaning",
        class_names=["left", "center", "right"],
        test_size=0.2,
        random_state=42,
        n_samples_to_explain=3
    )

# %%
