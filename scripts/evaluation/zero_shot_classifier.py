from transformers import pipeline
from datasets import Dataset
import torch

class ZeroShotClassifier:
    def __init__(self, labels=["left-wing/socialism", "right-wing/capitalism", "centrism"], hypothesis_template="The author of this posts leans towards {}."):
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
        self.labels = labels
        self.hypothesis_template = hypothesis_template

    def __call__(self, posts):
        dataset = Dataset.from_dict({"post": posts})
        results = self.classifier(dataset["post"], candidate_labels=self.labels, hypothesis_template=self.hypothesis_template, batch_size=8)

        scores = {label: [result['scores'][result['labels'].index(label)] for result in results] for label in self.labels}

        return scores