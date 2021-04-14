import logging
import random

import datasets
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from transformers import FlaubertTokenizer, TFFlaubertForSequenceClassification, AutoConfig
from flaubertpreprocessor import FlaubertPreprocessor

PICKLE_PATH = "dataset/allocine_dataset.pickle"
DATASET_NAME = "allocine_dataset"
MODEL_PATH = './{}_flaubert'.format(DATASET_NAME.replace('/', '_'))
MAX_SEQ_LEN = 400

train_ds = None
validation_ds = None
test_ds = None
tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
encoded_dataset = None
raw_train_ds = None
raw_validation_ds = None
train_labels_ds = None
test_labels_ds = None
validation_labels_ds = None



def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)
        value = dataset[pick]["review"]
        print(value)
        print(tokenizer.tokenize(value)[:15])
        print(tokenizer.encode(value)[:15])
        print(tokenizer.decode(tokenizer.encode(value)))

    print(dataset[picks])


def prepare_dataset():
    global train_ds,test_ds, tokenizer, encoded_dataset, raw_train_ds,validation_ds, raw_validation_ds, train_labels_ds, test_labels_ds, validation_labels_ds
    logging.info('preparing dataset')

    logging.info('loading dataset')
    # We load the allocine dataset from allocine
    raw_train_ds, raw_test_ds, raw_validation_ds = datasets.load_dataset('allocine', split=['train[:500]', 'test[:200]', 'validation[:200]'])
    logging.info('dataset loaded')
    show_random_elements(raw_train_ds, 1)

    def encode_examples(examples):
        # token_ids = np.zeros(shape=(len(examples), MAX_SEQ_LEN),
        #                      dtype=np.int32)
        return tokenizer(examples['review'], truncation=True, max_length=MAX_SEQ_LEN)
        # for i, encoded_example in enumerate(encoded_examples["input_ids"]):
        #     token_ids[i, 0:len(encoded_example)] = encoded_example
        # attention_mask = (token_ids != 0).astype(np.int32)
        # return {"input_ids": token_ids, "attention_mask": attention_mask}



    def to_tf_dataset(dataset):
        dataset = dataset.map(encode_examples, batched=True)
        dataset.set_format(type='tensorflow', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        features = {x: dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.model_max_length]) for x in
                    ['input_ids', 'token_type_ids', 'attention_mask']}
        return tf.data.Dataset.from_tensor_slices((features, dataset["label"])).batch(32)

    preprocessor = FlaubertPreprocessor(tokenizer, MAX_SEQ_LEN)
    train_ds, train_labels_ds = preprocessor.transform(raw_train_ds["review"], raw_train_ds["label"])
    test_ds, test_labels_ds = preprocessor.transform(raw_test_ds["review"], raw_test_ds["label"])
    validation_ds, validation_labels_ds = preprocessor.transform(raw_validation_ds["review"], raw_validation_ds["label"])
    logging.info('dataset prepared')

def create_model():
    logging.info("creating model")
    model = TFFlaubertForSequenceClassification.from_pretrained("jplu/tf-flaubert-base-cased")

    opt = tf.keras.optimizers.Adam(learning_rate=5e-6, epsilon=1e-08)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
    return model

def train_model(model):
    logging.info("training model")
    epochs = 2
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=0.02,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
        )
    ]

    history = model.fit(
        train_ds,
        train_labels_ds,
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        validation_data=(validation_ds, validation_labels_ds)
    )
    logging.info("model trained")
    return history

def evaluate_model(model, training_history):
    logging.info("evaluating model")
    loss, accuracy = model.evaluate((test_ds, test_labels_ds))

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    history_dict = training_history.history
    history_dict.keys()

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
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
    logging.info("model evaluated")

def export_model(model):
    model.save(MODEL_PATH, include_optimizer=False)
    model.save_weights(MODEL_PATH + ".hdf5")

def load_model():
    return tf.saved_model.load(MODEL_PATH)

def main():
    # prepare_dataset()
    # model = create_model()
    # training_history = train_model(model)
    # evaluate_model(model, training_history)
    # result_model = export_model(model)
    # result_model = load_model()
    #nlp = pipeline("sentiment-analysis", model="./flaubert-allocine")

    examples = [
            "The movie was great!",
            "The movie was okay.",
            "The movie was terrible...",
            "The movie was terrific!",
            "Le film était pas terrible",
            "Le film était terrible !",
            "Le film était terriblement bien"
        ]
    preprocessor = FlaubertPreprocessor(tokenizer, MAX_SEQ_LEN)
    model = TFFlaubertForSequenceClassification.from_pretrained("jplu/tf-flaubert-base-cased", num_labels=2)
    model.load_weights(MODEL_PATH + '.hdf5')

    config = AutoConfig.from_pretrained("jplu/tf-flaubert-base-cased")
    id2label = {"NEGATIVE": 0, "POSITIVE": 1}
    config.label2Id = id2label
    config.id2Label = id2label
    config._num_labels = 2
    model.config = config

    scores = model.predict(examples)
    # scores = result_model.predict(examples)
    print(np.argmax(scores, axis=1))

main()