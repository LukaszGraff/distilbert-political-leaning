from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score
import tqdm
import torch

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
        return predictions


    def _postprocess_output(self, output):
        if isinstance(output, list):
            # Extract the assistant's message
            for message in output[0]['generated_text']:
                if message['role'] == 'assistant':
                    return message['content'].lower().strip()
        return "unknown"

# Load the CSV file
print("Loading data")
df = pd.read_csv("data/processed/test.csv")
# df = df.sample(n=64, random_state=42)

# Extract posts and labels
print("Extracting posts")
X_test = df["post"].tolist()
print("Extracting labels")
y_test = df["political_leaning"].tolist()

# Initialize the batch predictor
predictor = LLamaBatchPredictor()

# Make predictions
predictions = predictor.classify_texts_batch(X_test, batch_size=8)

# Evaluate the predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
