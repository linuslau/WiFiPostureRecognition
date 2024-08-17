import tensorflow as tf
import numpy as np

from dataset import get_data, get_cached_data
from model import *  # Import models from model.py
from plot import *  # Import models from plot.py
import config  # Import models from config.py

def prepare_tf_dataset(dataset):
    def generator():
        for data, label in dataset:
            yield data.numpy(), label.numpy()

    return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(500, 90), dtype=tf.float32),
        tf.TensorSpec(shape=(8,), dtype=tf.int8)
    ))

def inference(model, dataset):
    test_loader = prepare_tf_dataset(dataset).batch(1)
    predictions = []
    true_labels = []

    for X, Y in test_loader:
        pred = model(X, training=False)

        # For CNN models, the output is typically 2D (batch_size, num_classes)
        if len(pred.shape) == 2:
            predictions.append(np.argmax(pred, axis=1)[0])  # Get the predicted class index
            true_labels.append(np.argmax(Y, axis=1)[0])  # Get the true class index
        # For RNN models, the output might be 3D (batch_size, sequence_length, num_classes)
        elif len(pred.shape) == 3:
            # Take the prediction from the last time step
            predictions.append(np.argmax(pred[:, -1, :], axis=1)[0])  # Last time step predicted class index
            true_labels.append(np.argmax(Y, axis=1)[0])  # Get the true class index

    return predictions, true_labels

def main():
    # Load the entire model (architecture + weights)
    model = tf.keras.models.load_model('model.h5')

    print("Model architecture and weights loaded successfully!")
    model.summary()

    if config.DEBUG:
        _, test_dataset = get_cached_data()
    else:
        _, test_dataset = get_data()

    predictions, true_labels = inference(model, test_dataset)

    correct_count = 0
    for idx, (pred, true) in enumerate(zip(predictions, true_labels)):
        result = "pass" if pred == true else "fail"
        print(f"Sample {idx}: Predicted class = {class_labels[pred]}, True class = {class_labels[true]} ({result})")
        if pred == true:
            correct_count += 1

    accuracy = correct_count / len(predictions)
    print(f"\nOverall accuracy: {accuracy:.2%}")

    # Plot the results
    plot_results(predictions, true_labels)

if __name__ == "__main__":
    main()
