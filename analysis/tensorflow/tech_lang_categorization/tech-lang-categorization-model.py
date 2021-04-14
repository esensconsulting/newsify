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
MAX_FEATURES = 5000

vectorize_layer = None
DATASET_DIR = "dataset/"
DATASET_URL = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
DATASET_NAME = "stack_overflow"

def prepare_dataset():
    global train_ds
    global val_ds
    global test_ds
    global vectorize_layer
    global raw_test_ds

    dataset_dir = os.path.join(DATASET_DIR)
    if not os.path.isdir(dataset_dir):
        download_dataset()


    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    remove_dir = os.path.join(train_dir, 'unsup')
    if os.path.isdir(remove_dir):
        shutil.rmtree(remove_dir)

    batch_size = 32
    seed = 42
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size)

    sequence_length = 500

    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')

    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Note that the layer uses the custom standardization defined above.
    # Set maximum_sequence length as all samples are not of the same length.
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=MAX_FEATURES,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = raw_test_ds.map(vectorize_text).cache().prefetch(buffer_size=AUTOTUNE)


def download_dataset():
    url = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    tf.keras.utils.get_file(DATASET_DIR + "aclImdb.tar.gz", url,
                            extract=False, cache_dir='',
                            cache_subdir='')
    tarFile = tarfile.open(DATASET_DIR + "aclImdb.tar.gz")
    tarFile.extractall(DATASET_DIR)
    if (os.path.isdir(DATASET_DIR + DATASET_NAME)):
        files_list = os.listdir(DATASET_DIR + DATASET_NAME)
        for files in files_list:
            shutil.move(DATASET_DIR + DATASET_NAME + "/" + files, DATASET_DIR)
        os.removedirs(DATASET_DIR + DATASET_NAME)

def create_model():
    ## Le modèle est composé d'une couche de vectorisation des mots (embedding) en 16 dimension qui est ensuite averagée et densifiée en 1 dimension
    embedding_dim = 128
    model = tf.keras.Sequential([
        layers.Embedding(MAX_FEATURES + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(4)])
    model.summary()

    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.SparseCategoricalAccuracy())
    return model

def train_model(model) -> History:
    epochs = 10

    # On applique un early stopping a l'entrainement, pour éviter de continuer de travailler sur des échantillons déjà vlaidés
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=0.02,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
        )
    ]

    training_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks)
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
    prepare_dataset()
    model = create_model()
    training_history = train_model(model)
    evaluate_model(model, training_history)
    result_model = export_model(model)

    examples = [
        "Cannot seem to extend my interface. Should I use implements instead ?",
        "I Keep getting ValueError when using Tensorflow",
        "Which version of the JDK should I use ?",
        "Which version or python is correct ?  2.7 or 3.6 ?",

    ]
    results = result_model.predict(examples)

    print('Results from the saved model:')
    print_my_examples(examples, results)
    # print('Results from the model in memory:')
    # print_my_examples(examples, original_results)

def print_my_examples(inputs, results):
  for i in range(len(inputs)):
    result_for_printing = f'input: {inputs[i]:<30} : score: \nC#: {results[i][0]:.6f}\nJava: {results[i][1]:.6f}\nJavascript: {results[i][2]:.6f}\nPython: {results[i][3]:.6f}\n'
    print(result_for_printing)




main()