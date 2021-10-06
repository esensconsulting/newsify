import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, FlaubertTokenizer, FlaubertForSequenceClassification, \
    IntervalStrategy
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import datasets
from sklearn import metrics
from utils import print_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
# Read data


# Define pretrained tokenizer and model





# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters


model_path = "output/allocine-flaubert-model"
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

# On charge le dataset allociné depuis Huggingface et le séparons en 3
raw_train_ds, raw_test_ds, raw_validation_ds = datasets.load_dataset('allocine', split=['train', 'test', 'validation'])

# On définit le modèle à affiner
model_name = "flaubert/flaubert_base_cased"

# On charge l'utilitaire de vectorisation du jeu de données associé à notre modèle
tokenizer = FlaubertTokenizer.from_pretrained(model_name)
model = FlaubertForSequenceClassification.from_pretrained(model_name, num_labels=2)
def preprocess_data():

    # On extrait les vecteurs d'entrainement et de validation
    X_train = list(raw_train_ds["review"])  # X=le texte a traité
    y_train = list(raw_train_ds["label"])   # Y=le label associé au texte (positif ou negatif)
    X_val = list(raw_validation_ds["review"])
    y_val = list(raw_validation_ds["label"])

    # On vectorise les textes à traiter
    X_train_tokenized = tokenizer(  # On utilise l'utilitaire pour vectoriser
        X_train,    # On passe le jeu d'entrainement
        truncation=True,    # On demande à tronquer les exemples trop longs
        max_length=512,     # On tronque à partir de 512 tokens
        padding=True,   # On rembourre les  vecteurs pour qu'ils aient tous une taille de 512 tokens
    )
    X_val_tokenized = tokenizer(
        X_val,
        padding=True,
        truncation=True,
        max_length=512
    )

    # On créer des Dataset PyTorch à partir de nos données vectorisées et du label associé à chaque exemple
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    return train_dataset, val_dataset

# On charge notre modèle à partir de l'utilitaire utilisé pour la classification de phrases FlauBERT.
model = FlaubertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def create_model(train_dataset, val_dataset):
    # On précise comment évaluer les métriques permettant de déterminer la performance du modèle pendant l'entraînement
    def compute_metrics(p):
        prediction, labels = p
        prediction = np.argmax(prediction, axis=1)

        # On réutilise des fonctions fournies par la librairie sklearn.
        accuracy = accuracy_score(y_true=labels, y_pred=prediction)
        recall = recall_score(y_true=labels, y_pred=prediction)
        precision = precision_score(y_true=labels, y_pred=prediction)
        f1 = f1_score(y_true=labels, y_pred=prediction)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


    # Configuration de l'entraînement
    args = TrainingArguments(
        output_dir="output",    # Dossier de travail
        evaluation_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=3,  # On limite le nombre de traitements parallèles pour l'entrainement
        per_device_eval_batch_size=3,   # Et pour l'évaluation
        learning_rate=1e-5,
        adam_epsilon=1e-8,
        weight_decay=0.07,
        num_train_epochs=3,     # On sépare l'entraînement en 3.
        metric_for_best_model="accuracy",   # Pour trouver le meilleur modèle, il faut comparer la précision
        load_best_model_at_end=True,
    )

    # On prépare l'utilitaire d'entrainement
    trainer = Trainer(
        args=args,  # Configuration définie ci-dessus
        model=model,    # Le modèle chargé
        train_dataset=train_dataset,    # Le jeu de données tokenizé d'entraînement
        eval_dataset=val_dataset,   # Le jeu de données tokenizé de validation
        compute_metrics=compute_metrics,    # Les méthodes à utiliser pour évaluer les performances du modèle
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],   # La configuration d'arrêt prématuré
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
    # On charge notre modèle sauvegardé
    model = FlaubertForSequenceClassification.from_pretrained(model_path, num_labels=2)

    # Et on le wrappe dans un objet Trainer
    return Trainer(model)

def model_analysis(model):
    X_test = raw_test_ds["review"]
    y_test = list(raw_test_ds["label"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
    test_dataset = Dataset(X_test_tokenized)

    raw_pred, _, _ = model.predict(test_dataset)
    y_pred = np.argmax(raw_pred, axis=1)

    print("Val Accuracy: {:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred)))
    print("Val F1-Score: {:.2f}".format(100 * metrics.f1_score(y_test, y_pred)))

    conf_mx = confusion_matrix(y_test, y_pred)

    fig = print_confusion_matrix(
        conf_mx,
        ['NEGATIVE', 'POSITIVE'],
        figsize=(7, 5)
    )

    # Finalize the plot
    sns.despine(bottom=True)
    plt.tight_layout(h_pad=2)

    # Saving plot
    fig.savefig('output/val_confusion_mx.png', dpi=200)

    inference_times = []

    for i in range(len(X_test_tokenized['input_ids'])):
        x = Dataset({
            'input_ids': np.array([X_test_tokenized['input_ids'][i], ]),
            'attention_mask': np.array([X_test_tokenized['attention_mask'][i], ]),
        })
        start_time = time.time()
        y_pred = model.predict(x)
        stop_time = time.time()

        inference_times.append(stop_time - start_time)

    OUTPUT_PATH = 'output/flaubert_times.pickle'

    with open(OUTPUT_PATH, 'wb') as writer:
        pickle.dump(inference_times, writer)

def add_times_to_df(path, df, model_name):
    with open(path, 'rb') as reader:
        times = pickle.load(reader)
        df = df.append(
            pd.DataFrame([[model_name, 1000*time] for time in times], columns=["model", "times"]),
            ignore_index=True
        )
    return df

def main():
    train_dataset, val_dataset = preprocess_data()
    trainer = create_model(train_dataset, val_dataset)
    train_model(trainer)
    evaluate_model(trainer)
    save_model(trainer)
    saved_model = load_model()
    model_analysis(saved_model)
    # better display of review text in dataframes
    pd.set_option('display.max_colwidth', None)

    # Seaborn options
    sns.set(style="whitegrid", font_scale=1.4)
    time_data = pd.DataFrame()
    time_data = add_times_to_df(
        'output/flaubert_times.pickle',
        time_data, 'FlauBert')

    fig = plt.figure(figsize=(10, 7))

    plt.figtext(.5, -0.05, 'TF-IDF model runs on CPU - AMD Ryzen 5 3600 (6-core)', ha='center', color='b')
    plt.figtext(.5, -0.1, 'Flaubert models run on GPU', ha='center',
                color='b')

    sns.barplot(x='model', y='times', data=time_data, ci=None)  # capsize=.2)
    plt.xlabel('')
    plt.ylabel('Inference time (ms)')

    # Finalize the plot
    sns.despine(bottom=True)
    plt.tight_layout(h_pad=10)

    # Saving plot
    fig.savefig('output/inference_time.png', bbox_inches="tight", dpi=200)
    # Finalize the plot
    sns.despine(bottom=True)
    plt.tight_layout(h_pad=10)

    # Saving plot
    fig.savefig('img/inference_time.png', bbox_inches="tight", dpi=200)

    examples = [
            "The movie was great!",
            "The movie was okay.",
            "The movie was terrible...",
            "The movie was terrific!",
            "Le film était pas terrible",
            "Le film était terrible !",
            "Le film était terriblement bien"
        ]

    # On doit tokenizer nos exemples avant de les passer à notre modèle
    x_examples = tokenizer(examples, padding=True, truncation=True, max_length=512)
    # On demande au modèle de nous sortir les prédictions de nos exemples. On ignore ici les labels retournés et les métriques
    raw_pred, _, _ = saved_model.predict(Dataset(x_examples))
    # Pour chaque exemple, on récupère la prédiction avec la plus forte probabilité.
    y_pred = np.argmax(raw_pred, axis=1)

    # On affiche les résultats de chaque exemple
    for i in range(len(examples)):
        prediction = "POSITIVE" if y_pred[i] == 1 else "NEGATIVE"
        print(examples[i] + " : " + prediction)

main()