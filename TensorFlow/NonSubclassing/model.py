import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import TimeDistributed

# Global constants
MAX_EPOCHS = 3  # Maximum number of epochs
MODEL_TYPE = 'cnn'  # Set model type ('cnn', 'cnn_rnn', 'rnn')
RETURN_SEQUENCES = True  # LSTM should return the full sequence or just the last output

learning_rate = 0.0001
batch_size = 200

n_steps = 500
input_size = 90
hidden_size = 200
n_classes = 8

# Define class labels
class_labels = ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk", "unknown"]

def build_cnn_model(input_len, label_len=8):
    inputs = tf.keras.Input(shape=(input_len, 90))
    # First convolution and pooling layers
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)
    # Second convolution and pooling layers
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', name='conv2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool2')(x)
    # Flatten and output layers
    x = layers.Flatten(name='flatten')(x)
    outputs = layers.Dense(label_len, name='output')(x)
    model = models.Model(inputs, outputs)
    return model

def build_rnn_model(input_len, hidden_len, label_len=8, return_sequences=RETURN_SEQUENCES):
    inputs = tf.keras.Input(shape=(input_len, hidden_len))
    x = layers.LSTM(hidden_len, return_sequences=return_sequences)(inputs)

    if return_sequences:
        print("return_sequences is True, applying TimeDistributed Dense layer.")
        outputs = TimeDistributed(layers.Dense(label_len))(x)
    else:
        print("return_sequences is False, applying Dense layer.")
        outputs = layers.Dense(label_len)(x)

    model = models.Model(inputs, outputs)
    return model

def build_cnn_rnn_model(input_len, hidden_len, label_len=8, return_sequences=RETURN_SEQUENCES):
    inputs = tf.keras.Input(shape=(input_len, hidden_len))
    # CNN
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # RNN
    x = layers.LSTM(hidden_len, return_sequences=return_sequences)(inputs)
    
    if return_sequences:
        print("return_sequences is True, applying TimeDistributed Dense layer.")
        outputs = TimeDistributed(layers.Dense(label_len))(x)
    else:
        print("return_sequences is False, applying Dense layer.")
        outputs = layers.Dense(label_len)(x)

    model = models.Model(inputs, outputs)
    return model



