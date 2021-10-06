import tarfile

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.callbacks import History

train_ds = None
validation_ds = None
test_ds = None
raw_test_ds = None
MAX_FEATURES = 10000

vectorize_layer = None

def download_dataset():
    """
    Permet le téléchargement d'un jeu de données de type tar.gz et son extraction
    :return:
    """
    DATASET_DIR = "dataset/"
    DATASET_URL = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
    DATASET_NAME = "stack_overflow"

    # On vide le dossier dataset systématiquement pour s'assurer d'une copie propre
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)

    # Utilisation de l'utilitaire get_file pour récupérer le dataset et l'extraire
    tf.keras.utils.get_file(DATASET_DIR + DATASET_NAME + ".tar.gz", DATASET_URL,
                            extract=False,
                            cache_dir='./',
                            cache_subdir='./')

    # Extraction
    tarFile = tarfile.open(DATASET_DIR + DATASET_NAME + ".tar.gz")
    tarFile.extractall(DATASET_DIR)


def prepare_dataset():
    global train_ds
    global val_ds
    global test_ds
    global vectorize_layer
    global raw_test_ds

    # Répertoires du dataset
    DATASET_DIR = "dataset/"
    dataset_dir = os.path.join(DATASET_DIR)
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    seed = 32
    batch_size =32
    # Import du jeu de données d'entrainement
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        validation_split=0.2,   # 20% des données doivent être utilisées pour le jeu de validation
        subset='training',   # On extrait ici le jeu d'entrainement
        seed=seed,   # Paramétrage de randomisation
        batch_size=batch_size
    )

    # Import du jeu de données de validation
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,  # On utilise le même dossier que pour l'entrainement
        validation_split=0.2,
        subset='validation',    # Mais on extrait le jeu de validation
        seed=seed,
        batch_size=batch_size
    )

    # Import du jeu de données de test
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(test_dir,
                                                                     seed=seed,
                                                                     batch_size=batch_size)


    # Standardisation de nos données
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)    # On passe tout en minuscules
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')  # On supprime les balises HTML
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation), # Et on enlève toute ponctuation
                                        '')

    # On utilise une couche de vectorisation du texte pour normaliser (a partir de notre fonction définie au-dessus)
    # et transformer en vecteurs d'entiers nos données en limitant la taille de ces derniers
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=250
    )

    # On appelle "adapt" pour créer un dictionnaire des mots utilisés et leur assigner un entier
    vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

    # Méthode vectorisant un texte. Prends en entrée le texte, et le sentiment associé afin de garder
    def vectorize_text(text, sentiment):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), sentiment

    # On vectorise tous nos jeux de données et les mettons en cache pour accélérer l'entrainement
    train_ds = raw_train_ds.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = raw_val_ds.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = raw_test_ds.map(vectorize_text).cache().prefetch(buffer_size=tf.data.AUTOTUNE)





def create_model():
    model = tf.keras.Sequential([
        layers.Embedding(10000 + 1, 16),    # On veut une matrice a 16 dimensions a partir de notre dictionnaire de 10000 mots
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),    # Calcul d'une moyenne
        layers.Dropout(0.2),
        layers.Dense(4)])   # On ne veut qu'un seul résultat, la probabilité de positivité

    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.SparseCategoricalAccuracy())
    return model

def train_model(model) -> History:
    # On définit une méthode permettant l'arrêt de l'apprentissage si nécessaire
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # On arrête l'apprentissage si la perte ne diminue plus suffisament
            monitor="val_loss",
            # Delta pour déterminer si la perte diminue suffisament ou pas
            min_delta=0.02,
            # On attend au moins 2 phases d'entrainement sans perte suffisante pour arrêter l'apprentissage
            patience=2,
            verbose=1,
        )
    ]

    # Lancement de l'apprentissage
    training_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10, # On sépare l'apprentissage en 10 étapes
        callbacks=callbacks
    )

    return training_history

def evaluate_model(model, training_history: History):
    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = training_history.history
    history_dict.keys()

    acc = history_dict['sparse_categorical_accuracy']
    val_acc = history_dict['val_sparse_categorical_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

def export_model(model):
    export_model = tf.keras.Sequential([
      vectorize_layer,
      model,
      layers.Activation('sigmoid')
    ])

    # Application de la fonction de loss
    export_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    # Test it with `raw_test_ds`, which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)
    return export_model

def main():
    download_dataset()
    prepare_dataset()
    model = create_model()
    training_history = train_model(model)
    evaluate_model(model, training_history)
    result_model = export_model(model)

    examples = [
        "The movie was great!",
        "The movie was okay.",
        "The movie was terrible...",
        "The movie was terrific!",
        "Le film était pas terrible",
        "Le film était terrible !",
        "Le film était terriblement bien"
    ]
    result_model.predict(examples)





main()