import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FlaubertPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def fit(self, X=None):
        pass

    def transform(self, X, y):
        # 1. Tokenize
        X_encoded = self.encode_values(X)
        # 2. Labels
        y_array = np.array(y)
        return X_encoded, y_array

    def encode_values(self, values):
        token_ids = np.zeros(shape=(len(values), self.max_seq_length),
                             dtype=np.int32)
        for i, value in enumerate(values):
            encoded = self.tokenizer.encode(value, truncation=True, max_length=self.max_seq_length)
            token_ids[i, 0:len(encoded)] = encoded
        attention_mask = (token_ids != 0).astype(np.int32)
        return {"input_ids": token_ids, "attention_mask": attention_mask}

    def fit_transform(self, X, y):
        return self.transform(X, y)
