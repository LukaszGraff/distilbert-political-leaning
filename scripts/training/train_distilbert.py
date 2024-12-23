import pandas as pd
import numpy as np
import torch
import os
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score

# Read the processed data
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
df_val = pd.read_csv(os.path.join(data_path, 'val.csv'))
df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))

# Map political_leaning labels to numerical values
label_mapping = {"left": 0, "center": 1, "right": 2}
df_train["label"] = df_train["political_leaning"].map(label_mapping)
df_val["label"] = df_val["political_leaning"].map(label_mapping)
df_test["label"] = df_test["political_leaning"].map(label_mapping)

# Define evaluation metrics
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

# Convert DataFrames to Hugging Face Dataset format
def convert_to_dataset(df, tokenizer, text_column, label_column):
    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding="max_length", truncation=True)

    dataset = Dataset.from_pandas(df)
    return dataset.map(tokenize_function, batched=True)

# Assuming df_train, df_val, df_test are available
text_column = "post"  # Change as per your DataFrame structure
label_column = "label"

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Convert datasets
datasets = DatasetDict({
    "train": convert_to_dataset(df_train, tokenizer, text_column, label_column),
    "validation": convert_to_dataset(df_val, tokenizer, text_column, label_column),
    "test": convert_to_dataset(df_test, tokenizer, text_column, label_column),
})

# Load the DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    evaluation_strategy="epoch",    # Evaluate every epoch
    save_strategy="epoch",
    logging_dir="./logs",           # directory for storing logs
    logging_steps=10,
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    save_total_limit=1,              # Only keep the most recent model checkpoint
    load_best_model_at_end=True      # Load best model after training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Test the model
test_results = trainer.predict(datasets["test"])
print("Test Results:", test_results.metrics)
