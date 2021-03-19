import tarfile
import logging

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp import optimization  # to create AdamW optmizer

import tensorflow_text as text  # Registers the ops.
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.callbacks import History

train_ds = None
validation_ds = None
test_ds = None
raw_test_ds = None
MAX_FEATURES = 10000

vectorize_layer = None
DATASET_DIR = "./dataset/"
DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATASET_NAME = "aclImdb"

PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
BERT_MODEL = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"

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

    sequence_length = 250

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
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    tf.keras.utils.get_file(DATASET_DIR + DATASET_NAME + ".tar.gz", DATASET_URL,
                                      extract=False, cache_dir='.',
                                      cache_subdir='')
    tarFile = tarfile.open(DATASET_DIR + DATASET_NAME + ".tar.gz")
    tarFile.extractall(DATASET_DIR)
    if(os.path.isdir(DATASET_DIR + DATASET_NAME)):
        files_list = os.listdir(DATASET_DIR + DATASET_NAME)
        for files in files_list:
            shutil.move(DATASET_DIR + DATASET_NAME + "/" +  files, DATASET_DIR)
        os.removedirs(DATASET_DIR + DATASET_NAME)

def create_model():

    logging.info('creating model')
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(PREPROCESS_MODEL, name="preprocessing")
    encoder_input = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(BERT_MODEL, trainable=True, name="BERT_encoder")
    outputs = encoder(encoder_input)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    model = tf.keras.Model(text_input, net)
    logging.info('model created')

    text_test = ['this is such an amazing movie!']
    bert_raw_result = model(tf.constant(text_test))
    print(tf.sigmoid(bert_raw_result))

    # Optimizer
    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model

def train_model(model) -> History:
    epochs = 5

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
        x=train_ds,
        validation_data=val_ds)
    return training_history

def evaluate_model(model, training_history: History):
    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = training_history.history
    history_dict.keys()

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
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
        loss=losses.BinaryCrossentropy(from_logits=True), optimizer="adam", metrics=['accuracy']
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
    #
    # examples = [
    #     "The movie was great!",
    #     "The movie was okay.",
    #     "The movie was terrible...",
    #     "The movie was terrific!",
    #     "Le film était pas terrible",
    #     "Le film était terrible !",
    #     "Le film était terriblement bien"
    # ]
    # result_model.predict(examples)
    text_test = ['this is such an amazing movie!']
    result_model.predict(text_test)

main()