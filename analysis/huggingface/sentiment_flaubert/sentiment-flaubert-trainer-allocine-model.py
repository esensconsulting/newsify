import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer, FlaubertTokenizer, FlaubertForSequenceClassification, \
    IntervalStrategy
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import datasets

# Read data
raw_train_ds, raw_test_ds, raw_validation_ds = datasets.load_dataset('allocine', split=['train', 'test', 'validation'])

# Define pretrained tokenizer and model
model_name = "flaubert/flaubert_base_cased"
tokenizer = FlaubertTokenizer.from_pretrained(model_name)
model = FlaubertForSequenceClassification.from_pretrained(model_name, num_labels=2)


# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters


model_path = "output/trainer-flaubert"
# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def preprocess_data():
    logging.info("Preprocessing data")
    # ----- 1. Preprocess data -----#
    # Preprocess data
    X_train = list(raw_train_ds["review"])
    y_train = list(raw_train_ds["label"])
    X_val = list(raw_validation_ds["review"])
    y_val = list(raw_validation_ds["label"])
    X_test = list(raw_test_ds["review"])
    y_test = list(raw_test_ds["label"])
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)



    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)
    test_dataset = Dataset(X_test_tokenized, y_test)
    logging.info("Data preprocessed")
    return train_dataset, val_dataset, test_dataset

def create_model(train_dataset, val_dataset):
    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Define Trainer
    args = TrainingArguments(
        output_dir="output",
        learning_rate=5e-6,
        # adam_epsilon=1e-8,
        evaluation_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=3,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer

def train_model(trainer: Trainer):
    trainer.train()
    return trainer

def evaluate_model(trainer: Trainer):
    trainer.evaluate()

def save_model(trainer: Trainer):
    trainer.save_model(model_path)

def load_model():
    # Load trained model

    model = FlaubertForSequenceClassification.from_pretrained(model_path, num_labels=2)

    # Define test trainer
    return Trainer(model)

def main():
    train_dataset, val_dataset, test_dataset = preprocess_data()
    trainer = create_model(train_dataset, val_dataset)
    train_model(trainer)
    evaluate_model(trainer)
    save_model(trainer)
    saved_model = load_model()

    # # Make prediction
    # raw_pred, _, _ = model.predict(test_dataset)
    #
    # # Preprocess raw predictions
    # y_pred = np.argmax(raw_pred, axis=1)
    # print(y_pred)

    examples = [
            "The movie was great!",
            "The movie was okay.",
            "The movie was terrible...",
            "The movie was terrific!",
            "Le film était pas terrible",
            "Le film était terrible !",
            "Le film était terriblement bien"
        ]

    x_examples = tokenizer(examples, padding=True, truncation=True, max_length=512)
    raw_pred, _, _ = saved_model.predict(Dataset(x_examples))
    y_pred = np.argmax(raw_pred, axis=1)
    print(y_pred)

main()