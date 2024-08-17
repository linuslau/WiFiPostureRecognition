import tensorflow as tf
from tensorflow.keras import layers, models

# Global constants
MAX_EPOCHS = 30  # Maximum number of epochs
MODEL_TYPE = 'cnn'  # Set model type ('cnn', 'cnn_rnn', 'rnn')

class RNN(models.Model):
    def __init__(self, input_len, hidden_len, label_len=8, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.label_len = label_len
        self.lstm = layers.LSTM(hidden_len, return_sequences=True)
        self.dense = layers.Dense(label_len)

    def call(self, X):
        out = self.lstm(X)
        out = out[:, -1, :]  # Extract the output of the last time step
        out = self.dense(out)
        return out

    def get_config(self):
        config = super(RNN, self).get_config()
        config.update({
            "input_len": self.input_len,
            "hidden_len": self.hidden_len,
            "label_len": self.label_len
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CNN_RNN(models.Model):
    def __init__(self, input_len, hidden_len, label_len=8, **kwargs):
        super(CNN_RNN, self).__init__(**kwargs)
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.label_len = label_len

        # CNN layers
        self.conv1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu')
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        self.conv2 = layers.Conv1D(filters=64, kernel_size=3, activation='relu')
        self.pool2 = layers.MaxPooling1D(pool_size=2)
        
        # RNN layer
        self.lstm = layers.LSTM(hidden_len, return_sequences=True)
        
        # Dense layer
        self.dense = layers.Dense(label_len)

    def call(self, X):
        out = self.conv1(X)
        out = self.pool1(out)
        out = self.conv2(X)
        out = self.pool2(out)
        out = self.lstm(out)
        out = out[:, -1, :]  # Extract the output of the last time step
        out = self.dense(out)
        return out

    def get_config(self):
        config = super(CNN_RNN, self).get_config()
        config.update({
            "input_len": self.input_len,
            "hidden_len": self.hidden_len,
            "label_len": self.label_len
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CNN(models.Model):
    def __init__(self, input_len, label_len=8, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.input_len = input_len
        self.label_len = label_len

        self.conv1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu')
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        self.conv2 = layers.Conv1D(filters=64, kernel_size=3, activation='relu')
        self.pool2 = layers.MaxPooling1D(pool_size=2)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(label_len)

    def call(self, X):
        out = self.conv1(X)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.dense(out)
        return out

    def get_config(self):
        config = super(CNN, self).get_config()
        config.update({
            "input_len": self.input_len,
            "label_len": self.label_len
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
