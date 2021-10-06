import logging

from ray import tune
from sklearn import metrics
import datasets
import numpy as np
import torch
from datasets import load_dataset, DatasetDict, ClassLabel
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer, FlaubertTokenizer, FlaubertForSequenceClassification, \
    IntervalStrategy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read data
from transformers.trainer_utils import HPSearchBackend

raw_train_ds, raw_test_ds, raw_validation_ds = None, None, None



# Graph styling
# better display of review text in dataframes
pd.set_option('display.max_colwidth', None)

sns.set(style="whitegrid")

categories_list = [
#'alpes-maritimes',
#'hautes-alpes',
#'festival-avignon',
#'bouches-du-rhone',
# 'en-image',
#'bac-2014',
#'brevet',
# 'home',
# 'vrai-ou-fake',
# 'choix',
# 'partenariats',
#'festival-de-cannes',
#'bac',
# 'animaux',
# 'internet',
# 'sciences',
# 'replay-magazine',
#'elections',
# 'decouverte',
'sports',
# 'replay-jt',
'france',
#'meteo',
'societe',
'faits-divers',
'sciences-technologie',
'politique',
'sante',
'culture',
# 'replay-radio',
'economie',
'monde'
]

# Define pretrained tokenizer and model
model_name = "flaubert/flaubert_base_cased"
tokenizer = FlaubertTokenizer.from_pretrained(model_name)

# Define Trainer parameters

dataset_path = "articles-without-html.json"
model_path = "trainer-flaubert-small-3"

# Initialisation des solutions possible
labels = ClassLabel(names=categories_list)

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

def CountFrequency(list):

    # Creating an empty dictionary
    freq = {}
    for item in list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    return freq

def merge_categories(article):
    if article['categorie'] in ['alpes-maritimes', 'hautes-alpes', 'bouches-du-rhone', 'bac-2014', 'bac', 'brevet']:
        article['categorie'] = 'france'
    elif article['categorie'] in ['festival-avignon', 'festival-de-cannes']:
        article['categorie'] = 'culture'
    elif article['categorie'] in ['elections']:
        article['categorie'] = 'politique'
    elif article['categorie'] in ['sciences', 'animaux', 'internet']:
        article['categorie'] = 'sciences-technologie'

    return article

def analyze_data():
    dataset = load_dataset('json', data_files=dataset_path)['train']
    before_filter_length = len(dataset)
    dataset_filtered = dataset.filter(lambda article: article['article'] is not None)

    filtered_length = before_filter_length - len(dataset_filtered)
    print("{} articles with no data ({:.2f} % of total data removed)".format(
        filtered_length,
        100 * filtered_length / before_filter_length
    ))

    dataset_filtered = dataset_filtered.map(merge_categories)
    print("Merged articles categories")

    before_filter_length = len(dataset_filtered)
    dataset_filtered = dataset_filtered.filter(lambda article: article['categorie'] in categories_list )
    filtered_length = before_filter_length - len(dataset_filtered)
    print("{} articles with an invalid category ({:.2f} % of total data removed)".format(
        filtered_length,
        100 * filtered_length / before_filter_length
    ))

    # Filter coronavirus
    before_filter_length = len(dataset_filtered)
    dataset_filtered = dataset_filtered.filter(lambda article: article['categorie'] != "sante" or "coronavirus" not in article['url'])
    filtered_length = before_filter_length - len(dataset_filtered)
    print("{} articles with coronavirus subject ({:.2f} % of total data removed)".format(
        filtered_length,
        100 * filtered_length / before_filter_length
    ))


    WORDS_MIN_THRESHOLD = 20
    WORDS_MAX_THRESHOLD = 600
    before_filter_length = len(dataset_filtered)
    dataset_filtered = dataset_filtered.filter(lambda article: article['mots'] < WORDS_MAX_THRESHOLD and article['mots'] > WORDS_MIN_THRESHOLD )
    filtered_length = before_filter_length - len(dataset_filtered)
    print("{} articles with words < {} or words > {} ({:.2f} % of total data removed)".format(
        filtered_length,
        WORDS_MIN_THRESHOLD,
        WORDS_MAX_THRESHOLD,
        100 * filtered_length / before_filter_length
    ))

    LENGTH_MAX_THRESHOLD = 3000
    before_filter_length = len(dataset_filtered)
    dataset_filtered = dataset_filtered.filter(lambda article: len(article['article']) <= LENGTH_MAX_THRESHOLD )
    filtered_length = before_filter_length - len(dataset_filtered)
    print("{} articles with length > {} ({:.2f} % of total data removed)".format(
        filtered_length,
        LENGTH_MAX_THRESHOLD,
        100 * filtered_length / before_filter_length
    ))

    print("{} articles in dataset".format(len(dataset_filtered)))

    def plot_length_repartition(dataset):
        articles_length = []
        for article in dataset:
            articles_length.append(len(article['article']))

        plt.figure(figsize=(10, 5))
        ax = sns.distplot(articles_length, bins=150, kde=False, hist_kws=dict(alpha=0.8))
        ax.set(xlabel='Article Length')

        # Finalize the plot
        sns.despine(bottom=True)
        plt.tight_layout(h_pad=2)

        # Saving plot
        fig = ax.get_figure()
        fig.savefig('output/articles_length.png', dpi=200)
        return

    def plot_word_count_repartition(dataset):
        words_count = []
        for article in dataset:
            if article['article'] is not None:
                words_count.append(article['mots'])

        plt.figure(figsize=(10, 5))
        ax = sns.distplot(words_count, bins=150, kde=False, hist_kws=dict(alpha=0.8))
        ax.set(xlabel='Words Count')

        # Finalize the plot
        sns.despine(bottom=True)
        plt.tight_layout(h_pad=2)

        # Saving plot
        fig = ax.get_figure()
        fig.savefig('output/articles_words.png', dpi=200)
        return

    def plot_count_by_categories(dataset):

        articles_per_categorie = CountFrequency(dataset['categorie'])
        plt.figure(figsize=(8, 5))
        articles_per_categorie = dict(sorted(articles_per_categorie.items(), key=lambda item: item[1]))
        for key, value in articles_per_categorie.items():
            print("{} : {}".format(key, value))

        ax = sns.barplot(y=list(articles_per_categorie.keys()), x=list(articles_per_categorie.values()))
        ax.set(xlabel='', ylabel='Categories')

        # Finalize the plot
        sns.despine(bottom=True)
        plt.tight_layout(h_pad=2)

        # Saving plot
        fig = ax.get_figure()
        fig.savefig('output/articles_per_categories.png', dpi=200)
        return

    def plot_count_by_year(dataset):

        years = map(lambda dateString : pd.to_datetime(dateString).year, dataset['date'])
        articles_per_year = CountFrequency(years)
        for key, value in articles_per_year.items():
            print("{} : {}".format(key, value))
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(list(articles_per_year.keys()), list(articles_per_year.values()))
        ax.set(xlabel='Year', ylabel='')

        # Finalize the plot
        sns.despine(bottom=True)
        plt.tight_layout(h_pad=2)

        # Saving plot
        fig = ax.get_figure()
        fig.savefig('output/articles_per_year.png', dpi=200)
        return

    plot_length_repartition(dataset_filtered)
    plot_word_count_repartition(dataset_filtered)
    plot_count_by_year(dataset_filtered)
    plot_count_by_categories(dataset_filtered)

    replays = dataset_filtered.filter(lambda article: article['categorie'] == 'replay-radio')
    replays = replays.shuffle(seed=42).select([0, 10, 20, 30, 40, 50])
    replays.map(lambda example: print(example))

    # dataset is already `map`'d and already has `set_format`
    # 90% train, 10% test + validation
    train_testvalid = dataset_filtered.train_test_split(test_size=0.1)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    datasets = DatasetDict({
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"]})

    datasets.save_to_disk('output/processed-dataset')


def preprocess_data():
    global raw_test_ds, raw_validation_ds, raw_train_ds

    # Chargement des datasets d'entrainement et de validation
    raw_train_ds = datasets.load_from_disk('./output/processed-dataset/train')
    raw_validation_ds = datasets.load_from_disk('./output/processed-dataset/valid')

    # Transformation des datasets d'entrainement et de validation
    X_train = list(raw_train_ds["article"])
    y_train = labels.str2int(raw_train_ds["categorie"])
    X_val = list(raw_validation_ds["article"])
    y_val = labels.str2int(raw_validation_ds["categorie"])

    # Tokenization des datasets d'entrainement et de validation
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    return train_dataset, val_dataset

def model_init():
    return FlaubertForSequenceClassification.from_pretrained(model_name, num_labels=len(categories_list))

def create_model(train_dataset, val_dataset):

    # Initialisation du modèle
    model = FlaubertForSequenceClassification.from_pretrained("flaubert/flaubert_base_cased", num_labels=len(categories_list))
    tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")

    # Définition des métrics à utiliser
    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average="micro")
        precision = precision_score(y_true=labels, y_pred=pred, average="micro")
        f1 = f1_score(y_true=labels, y_pred=pred, average="micro")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Configuration du trainer
    args = TrainingArguments(
        output_dir="output",
        learning_rate=1e-5,
        evaluation_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=3,
        weight_decay=0.07,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True
    )

    # Initialisation du trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer

def hyperparameter_space(trial):

    return {
        "per_device_train_batch_size": tune.choice([1, 2, 3, 4]),
        'learning_rate': tune.uniform(1e-5, 5e-5),
        'seed': tune.choice(range(1, 50)),
        'weight_decay': tune.uniform(0.0, 0.3),
        'num_train_epochs': tune.choice([2, 3, 4, 5]),
    }

def train_model(trainer: Trainer):

    trainer = trainer.train()
    return trainer

def evaluate_model(trainer: Trainer):
    trainer.evaluate()

def save_model(trainer: Trainer):
    trainer.save_model(model_path)

def load_model():
    # Load trained model
    model = FlaubertForSequenceClassification.from_pretrained(model_path, num_labels=len(categories_list))

    # Define test trainer
    return Trainer(model)

def main():
    analyze_data()
    train_dataset, val_dataset = preprocess_data()
    trainer = create_model(train_dataset, val_dataset)
    train_model(trainer)
    evaluate_model(trainer)
    save_model(trainer)
    saved_model = load_model()

    # On initialise nos exemples
    examples = [
        "\nRafael Nadal se disait \u00ab\u00a0pr\u00eat et en confiance\u00a0\u00bb avant d\u2019attaquer la saison sur terre battue...",
        "\nLe\u00a0cancer l\u2019a emport\u00e9e.\u00a0L\u2019actrice britannique Helen McCrory, qui a jou\u00e9 au cin\u00e9ma dans Skyfall et Harry Potter...",
        "\nL\u2019Europe doit \u00eatre \u00ab\u00a0aux avant-postes\u00a0\u00bb de la cr\u00e9ation d\u2019une monnaie num\u00e9rique commune..."
    ]
    examples_tokenized = tokenizer(examples, padding=True, truncation=True, max_length=512)

    # On exécute notre modèle sur nos exemples
    raw_pred, _, _ = saved_model.predict(Dataset(examples_tokenized))
    y_pred = np.argmax(raw_pred, axis=1)

    # Et on affiche les résultats
    for pred in labels.int2str(y_pred):
         print("Categorie : {}".format(pred))
main()
