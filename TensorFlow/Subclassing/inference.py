import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import getData
from model import *  # Import models from model.py

# Global constants
MODEL_TYPE = 'cnn'  # Set model type ('cnn', 'cnn_rnn', 'rnn')

# Define class labels
class_labels = ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk", "unknown"]

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
        predictions.append(np.argmax(pred, axis=1)[0])  # Get the predicted class index
        true_labels.append(np.argmax(Y, axis=1)[0])  # Get the true class index

    return predictions, true_labels

def plot_results(predictions, true_labels):
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels[:cm.shape[0]], yticklabels=class_labels[:cm.shape[0]])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Accuracy per class
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(accuracy_per_class)), accuracy_per_class, tick_label=class_labels[:len(accuracy_per_class)])
    plt.title('Accuracy per Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.show()

def main():
    # Load model architecture
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = tf.keras.models.model_from_json(loaded_model_json, custom_objects={
        'RNN': RNN,
        'CNN_RNN': CNN_RNN,
        'CNN': CNN
    })

    # Call the model once before loading weights to create variables
    dummy_input = tf.zeros((1, 500, 90))  # Assuming the input shape is (500, 90)
    model(dummy_input)

    # Load model weights
    model.load_weights("model_weights.h5")
    print("Model architecture and weights loaded successfully!")
    model.summary()

    _, test_dataset = getData()

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

