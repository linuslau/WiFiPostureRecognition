import tensorflow as tf
from keras import optimizers
from datetime import datetime
import numpy as np

from dataset import get_data, get_cached_data
from model import *  # Import models from model.py
from plot import *  # Import models from plot.py
import config  # Import models from config.py

def get_accuracy(predictions, labels):
    predY = np.argmax(predictions, axis=1)
    trueY = np.argmax(labels, axis=1)
    correct_count = np.sum(predY == trueY)
    total_count = len(trueY)
    acc = correct_count / total_count
    return acc, correct_count, total_count

def prepare_tf_dataset(dataset):
    def generator():
        for data, label in dataset:
            yield data.numpy(), label.numpy()

    return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(500, 90), dtype=tf.float32),
        tf.TensorSpec(shape=(8,), dtype=tf.int8)
    ))

def main():
    print("Initializing parameters")
    train_loss = []
    train_acc = []
    validation_loss = []
    validation_acc = []

    device_name = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    print(f'Running on device: {device_name}')

    with tf.device(device_name):
        if MODEL_TYPE == 'cnn_rnn':
            model = build_cnn_rnn_model(n_steps, input_size, n_classes)
            print("\n" + "="*50)
            print("   MODEL TYPE: CNN + RNN")
            print("="*50 + "\n")
        elif MODEL_TYPE == 'rnn':
            model = build_rnn_model(n_steps, input_size, n_classes)
            print("\n" + "="*50)
            print("   MODEL TYPE: RNN")
            print("="*50 + "\n")
        elif MODEL_TYPE == 'cnn':
            model = build_cnn_model(n_steps, n_classes)
            print("\n" + "="*50)
            print("   MODEL TYPE: CNN")
            print("="*50 + "\n")

        model.summary()

        for layer in model.layers:
            print(f"Layer {layer.name} output shape: {layer.output.shape}")

        model.compile(optimizer=optimizers.Adam(learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

        if config.DEBUG:
            train_dataset, test_dataset = get_cached_data()
        else:
            train_dataset, test_dataset = get_data()

        print("\n --Data loading completed---")

        start_time = datetime.now()
        best_acc = 0.0

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        for epoch in range(MAX_EPOCHS):
            print(f"\nEpoch {epoch+1}/{MAX_EPOCHS} - Training:")

            train_loader = prepare_tf_dataset(train_dataset).batch(batch_size).shuffle(buffer_size=10000)
            test_loader = prepare_tf_dataset(test_dataset).batch(batch_size)

            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

            for X, Y in train_loader:
                with tf.GradientTape() as tape:
                    predictions = model(X)

                    # Adjust the shape of Y to match the output when return_sequences=True
                    if len(predictions.shape) == 3 and predictions.shape[1] == n_steps:
                        if MODEL_TYPE == 'cnn_rnn' and RETURN_SEQUENCES:
                            # Ensure Y's shape matches predictions' shape
                            if len(Y.shape) == 2:
                                Y = tf.expand_dims(Y, axis=1)
                            Y = tf.tile(Y, [1, predictions.shape[1], 1])
                        elif MODEL_TYPE == 'cnn_rnn' and not RETURN_SEQUENCES:
                            Y = tf.reshape(Y, [-1, n_classes])
                        else:
                            # If not cnn_rnn model, still adjust Y's shape to match predictions
                            Y = tf.repeat(Y[:, tf.newaxis, :], repeats=predictions.shape[1], axis=1)

                    loss = loss_fn(Y, predictions)

                grads = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

                epoch_loss_avg.update_state(loss)
                epoch_accuracy.update_state(Y, predictions)

            train_loss.append(epoch_loss_avg.result().numpy())
            train_acc.append(epoch_accuracy.result().numpy())

            print(f'Train Loss: {train_loss[-1]:.3f}, Train Accuracy: {train_acc[-1]:.3f}')

            print("\nValidation:")

            val_accuracy = tf.keras.metrics.CategoricalAccuracy()
            val_loss_avg = tf.keras.metrics.Mean()

            for X, Y in test_loader:
                predictions = model(X)

                # Adjust the shape of Y to match the output when return_sequences=True
                if len(predictions.shape) == 3 and predictions.shape[1] == n_steps:
                    if MODEL_TYPE == 'cnn_rnn' and RETURN_SEQUENCES:
                        # Ensure Y's shape matches predictions' shape
                        if len(Y.shape) == 2:
                            Y = tf.expand_dims(Y, axis=1)
                        Y = tf.tile(Y, [1, predictions.shape[1], 1])
                    elif MODEL_TYPE == 'cnn_rnn' and not RETURN_SEQUENCES:
                        Y = tf.reshape(Y, [-1, n_classes])
                    else:
                        # If not cnn_rnn model, still adjust Y's shape to match predictions
                        Y = tf.repeat(Y[:, tf.newaxis, :], repeats=predictions.shape[1], axis=1)

                loss = loss_fn(Y, predictions)

                val_loss_avg.update_state(loss)
                val_accuracy.update_state(Y, predictions)

            validation_loss.append(val_loss_avg.result().numpy())
            validation_acc.append(val_accuracy.result().numpy())

            print(f'Validation Loss: {validation_loss[-1]:.3f}, Validation Accuracy: {validation_acc[-1]:.3f}')

            if validation_acc[-1] > best_acc:
                best_acc = validation_acc[-1]
                model.save('model.h5')
                print("Model architecture and weights saved successfully!")

        time_interval = datetime.now() - start_time
        print(f"Training completed in {time_interval.seconds:.2f} seconds, Best Accuracy: {best_acc:.3f}")

        if config.PLOT:
            plot_accuracy_and_loss(train_acc, validation_acc, train_loss, validation_loss)

            for data, label in test_dataset:
                X_sample = data.numpy()
                break

            if MODEL_TYPE == 'cnn':
                visualize_feature_maps(model, np.expand_dims(X_sample, axis=0))
                visualize_flatten_dense_layer_output(model, np.expand_dims(X_sample, axis=0))
            elif MODEL_TYPE in ['rnn', 'cnn_rnn']:
                visualize_rnn_activations(model, np.expand_dims(X_sample, axis=0))
                if MODEL_TYPE == 'cnn_rnn':
                    visualize_feature_maps(model, np.expand_dims(X_sample, axis=0))

if __name__ == "__main__":
    main()
