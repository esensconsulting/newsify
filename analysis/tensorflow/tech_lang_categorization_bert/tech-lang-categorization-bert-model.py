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
DATASET_DIR = "dataset/"
DATASET_URL = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
DATASET_NAME = "stack_overflow"

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


    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = raw_test_ds.cache().prefetch(buffer_size=AUTOTUNE)


def download_dataset():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    tf.keras.utils.get_file(DATASET_DIR + DATASET_NAME + ".tar.gz", DATASET_URL,
                            extract=False, cache_dir='',
                            cache_subdir='')
    tarFile = tarfile.open(DATASET_DIR + DATASET_NAME + ".tar.gz")
    tarFile.extractall(DATASET_DIR)
    if(os.path.isdir(DATASET_DIR + DATASET_NAME)):
        files_list = os.listdir(DATASET_DIR + DATASET_NAME)
        for files in files_list:
            shutil.move(DATASET_DIR + DATASET_NAME + "/" +  files, DATASET_DIR)
        os.removedirs(DATASET_DIR + DATASET_NAME)

def create_model():
    logging.info('model informations')
    bert_preprocess_model = hub.KerasLayer(PREPROCESS_MODEL, name="preprocessing")
    text_test = ['this is such an amazing movie!']
    text_preprocessed = bert_preprocess_model(text_test)

    print(f'Keys       : {list(text_preprocessed.keys())}')
    print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
    print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

    bert_model = hub.KerasLayer(BERT_MODEL, trainable=True, name="BERT_encoder")
    bert_results = bert_model(text_preprocessed)

    print(f'Loaded BERT: {BERT_MODEL}')
    print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
    print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
    print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
    print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

    logging.info('creating model')
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    encoder_inputs = bert_preprocess_model(text_input)
    outputs = bert_model(encoder_inputs)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(4, activation=None, name='classifier')(net)
    model = tf.keras.Model(text_input, net)
    logging.info('model created')

    logging.info('model verification')
    text_test = ['Value Error ']
    bert_raw_result = model(tf.constant(text_test))
    print(tf.sigmoid(bert_raw_result))

    logging.info('model compilation')
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

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = tf.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model

def train_model(model) -> History:
    epochs = 5


    training_history = model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=epochs)
    return training_history

def evaluate_model(model, training_history: History):
    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = training_history.history
    print(history_dict.keys())

    acc = history_dict['sparse_categorical_accuracy']
    val_acc = history_dict['val_sparse_categorical_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

def export_model(model):
    saved_model_path = './{}_bert'.format(DATASET_NAME.replace('/', '_'))

    model.save(saved_model_path, include_optimizer=False)

    return saved_model_path

def load_model(path):
    return  tf.saved_model.load(path)

def main():
    # prepare_dataset()
    # model = create_model()
    # training_history = train_model(model)
    # evaluate_model(model, training_history)
    #saved_model_path = export_model(model)
    result_model = load_model('./{}_bert'.format(DATASET_NAME.replace('/', '_')))
    examples = [
        "Cannot seem to extend my interface. Should I use implements instead ?",
        "I Keep getting ValueError when using Tensorflow",
        "Which version of the JDK should I use ?",
        "Which version or python is correct ?  2.7 or 3.6 ?",

    ]

    reloaded_results = tf.sigmoid(result_model(tf.constant(examples)))
    # original_results = tf.sigmoid(model(tf.constant(examples)))

    print('Results from the saved model:')
    print_my_examples(examples, reloaded_results)
    # print('Results from the model in memory:')
    # print_my_examples(examples, original_results)

def print_my_examples(inputs, results):
  for i in range(len(inputs)):
    result_for_printing = f'input: {inputs[i]:<30} : score: \nC#: {results[i][0]:.6f}\nJava: {results[i][1]:.6f}\nJavascript: {results[i][2]:.6f}\nPython: {results[i][3]:.6f}\n'
    print(result_for_printing)

main()