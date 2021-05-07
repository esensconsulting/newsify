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

dataset_path = "all-articles.json"
model_path = "output/trainer-flaubert-small-3"
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
    labels = ClassLabel(names = categories_list)
    logging.info("Preprocessing data")
    # ----- 1. Preprocess data -----#
    raw_train_ds = datasets.load_from_disk('./output/processed-dataset/train')
    raw_validation_ds = datasets.load_from_disk('./output/processed-dataset/valid')

    # Preprocess data
    X_train = list(raw_train_ds["article"])
    y_train = labels.str2int(raw_train_ds["categorie"])
    X_val = list(raw_validation_ds["article"])
    y_val = labels.str2int(raw_validation_ds["categorie"])
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)




    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    logging.info("Data preprocessed")
    return train_dataset, val_dataset

def model_init():
    return FlaubertForSequenceClassification.from_pretrained(model_name, num_labels=len(categories_list))

def create_model(train_dataset, val_dataset):
    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average="micro")
        precision = precision_score(y_true=labels, y_pred=pred, average="micro")
        f1 = f1_score(y_true=labels, y_pred=pred, average="micro")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Define Trainer
    args = TrainingArguments(
        output_dir="output",
        learning_rate=1e-5,
        # tpu_num_cores=8,
        # adam_epsilon=1e-8,
        evaluation_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=3,
        weight_decay=0.07,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model_init=model_init,
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

    # best_run = trainer.hyperparameter_search(n_trials=10,
    #                                          hp_space=hyperparameter_space,
    #                                          direction="maximize", backend=HPSearchBackend.RAY,
    #                                          resources_per_trial={"cpu": 1, "gpu": 1},)
    # for n, v in best_run.hyperparameters.items():
    #     setattr(trainer.args, n, v)

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
    # analyze_data()
    train_dataset, val_dataset = preprocess_data()
    trainer = create_model(train_dataset, val_dataset)
    train_model(trainer)
    evaluate_model(trainer)
    save_model(trainer)
    saved_model = load_model()
    # model_analysis(saved_model)


    raw_test_ds = datasets.load_from_disk('./output/processed-dataset/test')


    labels = ClassLabel(names=categories_list)
    X_val = list(raw_test_ds["article"])
    y_val = labels.str2int(raw_test_ds["categorie"])
    X_test_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
    # Make prediction
    raw_pred, _, _ = saved_model.predict(Dataset(X_test_tokenized))
    #
    # # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    test_acc  = metrics.accuracy_score(y_val, y_pred)
    print("Test accuracy : {}".format(test_acc))

    sport = "\nRafael Nadal se disait \u00ab\u00a0pr\u00eat et en confiance\u00a0\u00bb avant d\u2019attaquer la saison sur terre battue.\u00a0Et nous, bah, on plaignait d\u2019avance ses adversaires. Et pourtant\u00a0:\u00a0ce vendredi Andrey Rublev (8e mondial) a assomm\u00e9 l\u2019Espagnol (3e) 6-2, 4-6, 6-2 en 2h32, en quarts de finale du Masters 1000 de \r\nMonte-Carlo. Rublev retrouvera Casper Ruud (27e) dans le dernier carr\u00e9.\r\n\r\nNadal\u00a0ne s\u2019est pas montr\u00e9 au sommet de son art, commettant de nombreuses fautes, mais il a souvent \u00e9t\u00e9 pris par les coups violents et pr\u00e9cis du Russe. Ce dernier jouera sa deuxi\u00e8me demi-finale de Masters 1000 apr\u00e8s celle de Miami au d\u00e9but du mois.\r\n\r\nMedvedev, Djokovic et Nadal out\r\n\r\n\u00ab\u00a0Pour lui, \u00e7a doit \u00eatre incroyablement difficile de jouer avec cette pression de devoir toujours gagner. Je suis sous le choc de voir le niveau auquel il peut \u00e9voluer malgr\u00e9 cette pression. C\u2019est beaucoup plus facile de jouer quand on n\u2019a rien \u00e0 perdre\u00a0\u00bb, a comment\u00e9 Rublev.\r\n\r\nApr\u00e8s l\u2019exclusion du N.2 mondial Daniil Medvedev pour un test positif au Covid-19\u00a0avant le d\u00e9but du tournoi, puis l\u2019\u00e9limination du N.1 Novak Djokovic la veille en 8es de finale, Nadal est la troisi\u00e8me t\u00eate d\u2019affiche \u00e0 quitter le tournoi mon\u00e9gasque.\r\n\r\nL\u2019Espagnol n\u2019avait plus jou\u00e9 depuis l\u2019Open d\u2019Australie en f\u00e9vrier avant de s\u2019aligner sur le tournoi mon\u00e9gasque qu\u2019il a remport\u00e9 \u00e0 onze reprises.\r\n\r\nSportMonte-Carlo\u00a0: Novak Djokovic balay\u00e9 en deux manches par Daniel EvansSportMonte-Carlo\u00a0: \u00ab\u00a0J\u2019en ai rien \u00e0 branler\u00a0\u00bb, Beno\u00eet Paire en roue libre apr\u00e8s sa d\u00e9faite au premier tour\n"
    divers = "\nLe\u00a0cancer l\u2019a emport\u00e9e.\u00a0L\u2019actrice britannique Helen McCrory, qui a jou\u00e9 au cin\u00e9ma dans Skyfall et Harry Potter, et \u00e0 la t\u00e9l\u00e9vision dans la s\u00e9rie Peaky Blinders, est morte \u00e0 l\u2019\u00e2ge de 52 ans d\u2019un cancer, a annonc\u00e9 ce vendredi son \u00e9poux Damian Lewis sur Twitter.\r\n\r\n\u00ab\u00a0J\u2019ai le coeur bris\u00e9 d\u2019annoncer que, apr\u00e8s une bataille h\u00e9ro\u00efque contre le cancer, la femme magnifique qu\u2019est Helen McCrory est morte paisiblement chez elle, au milieu d\u2019une vague d\u2019amour de ses amis et de sa famille\u00a0\u00bb, a d\u00e9clar\u00e9 le com\u00e9dien dans un court texte sur le r\u00e9seau social.\u00a0\u00ab\u00a0Elle est morte comme elle a v\u00e9cu. Sans peur. Dieu que nous l\u2019aimons et savons la chance que nous avons eue de l\u2019avoir dans nos vies\u00a0\u00bb, a-t-il ajout\u00e9.\r\n\r\npic.twitter.com\/gSx8ib9PY9\u2014 Damian Lewis (@lewis_damian) April 16, 2021\n\nStar de\u00a0Peaky Blinders\n\r\n\r\nApparue pour la premi\u00e8re fois au cin\u00e9ma dans un petit r\u00f4le dans Entretien avec un Vampire\u00a0apr\u00e8s avoir commenc\u00e9 sa carri\u00e8re \u00e0 la t\u00e9l\u00e9vision, Helen McCrory a notamment incarn\u00e9 Narcissa Malfoy dans les derniers films de la saga Harry Potter.\u00a0\r\n\r\nL\u2019actrice, qui a \u00e9galement jou\u00e9 dans Skyfall\u00a0de la saga James Bond, incarnait\u00a0\u00e0 la perfection le personnage de tante Polly, matriarche du clan Shelby, dans la s\u00e9rie britannique \u00e0 succ\u00e8s Peaky Blinders, qui retrace les aventures d\u2019une famille de malfrats de Birmingham au d\u00e9but du 20e\u00a0si\u00e8cle.\u00a0Elle avait \u00e9pous\u00e9 \r\nDamian Lewis en 2007, avec qui elle a eu deux enfants.\r\n\r\nPeopleCoronavirus : Devant l\u2019insistance du casting, le tournage de \u00ab\u00a0Peaky Blinders\u00a0\u00bb s\u2019est arr\u00eat\u00e9 d\u00e8s le d\u00e9but de la pand\u00e9mie\n"
    economie = "\nL'Europe doit \u00eatre \u00ab\u00a0aux avant-postes\u00a0\u00bb de la cr\u00e9ation d\u2019une monnaie num\u00e9rique commune et \u00ab\u00a0activement\u00a0\u00bb oeuvrer pour que ce nouvel outil de paiement voie le jour, a plaid\u00e9 vendredi le ministre \r\nallemand des Finances, Olaf Scholz.\r\n\r\n\u00ab\u00a0Une Europe souveraine a besoin de solutions de paiement innovantes et comp\u00e9titives\u00a0\u00bb, a d\u00e9clar\u00e9 Olaf\u00a0Scholz en amont d\u2019une visioconf\u00e9rence des ministres des Finances de la zone euro (Eurogroupe), qui doit aborder cette question. Pour le ministre social-d\u00e9mocrate, \u00ab\u00a0l\u2019Europe doit \u00eatre aux avant-postes sur la question des monnaies digitales de banque centrale et doit activement le faire progresser\u00a0\u00bb.\r\n\r\nLa BCE d\u00e9cidera cet \u00e9t\u00e9\r\n\r\nAinsi, la premi\u00e8re \u00e9conomie de la zone euro  \u00ab\u00a0soutiendra de fa\u00e7on constructive\u00a0\u00bb les travaux engag\u00e9s par la Banque centrale europ\u00e9enne (BCE) en vue de la possible cr\u00e9ation d\u2019un euro digital. \u00ab\u00a0Nous ne devons pas \u00eatre spectateurs\u00a0\u00bb de cette \u00e9volution, a estim\u00e9 le ministre allemand qui a \u00e9galement appel\u00e9 \u00e0 \u00ab\u00a0ne pas se rendre d\u00e9pendant l\u00e0 o\u00f9 la souverainet\u00e9 des Etats est en jeu\u00a0\u00bb.\r\n\r\nLa BCE d\u00e9cidera cet \u00e9t\u00e9 si elle se lance ou non dans la cr\u00e9ation d\u2019un euro num\u00e9rique, \u00e0 l\u2019issue d\u2019une vaste consultation et d\u2019\u00e9tudes engag\u00e9es ces derniers mois, a indiqu\u00e9 cette semaine l\u2019un de ses responsables.\r\n\r\nPas besoin de compte en banque\r\n\r\nSelon une enqu\u00eate publique de l\u2019institution de Francfort, \u00e9galement d\u00e9voil\u00e9e cette semaine, les particuliers et les professionnels interrog\u00e9s attendent en premier lieu de la monnaie num\u00e9rique la confidentialit\u00e9 (43\u00a0%), suivie de la s\u00e9curit\u00e9 (18\u00a0%), la capacit\u00e9 de payer dans la zone euro (11\u00a0%), l\u2019absence de frais suppl\u00e9mentaires (9\u00a0%) et la possibilit\u00e9 de payer en dehors de l\u2019internet (8\u00a0%).\r\n\r\nLa question des cryptomonnaies est \u00e9tudi\u00e9e de pr\u00e8s par de nombreux pays, face notamment au projet de monnaie num\u00e9rique initi\u00e9 par Facebook, la Libra.\u00a0Plusieurs banques centrales planchent sur le sujet, la Chine et son projet de crypto-yuan comptant parmi les plus avanc\u00e9s.\r\n\r\nLes monnaies num\u00e9riques sont stock\u00e9es sur des supports \u00e9lectroniques, sans avoir besoin de compte en banque, et sont accept\u00e9es comme moyen de paiement par des entreprises.\r\n\r\nMondeCoronavirus\u00a0: Contrairement \u00e0 une grande partie de l\u2019Europe, la Suisse all\u00e8ge ses restrictionsSant\u00e9Coronavirus : Peut-on s'inspirer des pays en d\u00e9confinement pour r\u00e9ussir le n\u00f4tre ?\n"

    examples = [sport, divers, economie]
    examples_tokenized = tokenizer(examples, padding=True, truncation=True, max_length=512)

    raw_pred, _, _ = saved_model.predict(Dataset(examples_tokenized))
    y_pred = np.argmax(raw_pred, axis=1)

    for pred in labels.int2str(y_pred):
        print("Categorie : {}".format(pred))
main()
