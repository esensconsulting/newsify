import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf

class FlaubertEarlyStoppingModel(BaseEstimator):
    def __init__(self, transformers_model, max_epoches, batch_size, validation_data):
        self.model = transformers_model
        self.max_epoches = max_epoches
        self.batch_size = batch_size
        self.validation_data = validation_data

    def fit(self, X, y):
        # Defines early stopper
        early_stopper = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='auto', patience=2,  # only 1 !
            verbose=1, restore_best_weights=True
        )

        # Train model on data subset
        self.model.fit(
            X, y,
            validation_data=self.validation_data,
            epochs=self.max_epoches,
            batch_size=self.batch_size,
            callbacks=[early_stopper],
            verbose=1
        )
        return self

    def predict(self, X):
        scores = self.model.predict(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred