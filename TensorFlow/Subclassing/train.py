import tensorflow as tf
from keras import layers, models, optimizers
import matplotlib.pyplot as plt
from dataset import getData
from datetime import datetime
import numpy as np
from model import *  # Import models from model.py

def draw_train(train_acc, validation_acc, train_loss, validation_loss):
    plt.subplot(2, 1, 1)
    plt.plot(train_acc)
    plt.plot(validation_acc)
    plt.xlabel("n_epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train_acc", "validation_acc"], loc=4)
    plt.ylim([0, 1])

    plt.subplot(2, 1, 2)
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.xlabel("n_epoch")
    plt.ylabel("Loss")
    plt.legend(["train_loss", "validation_loss"], loc=1)
    plt.ylim([0, 2])
    plt.show()

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

    learning_rate = 0.0001
    batch_size = 200
    
    n_steps = 500
    input_size = 90
    hidden_size = 200
    n_classes = 8

    device_name = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    print(f'Running on device: {device_name}')

    with tf.device(device_name):
        if MODEL_TYPE == 'cnn_rnn':
            model = CNN_RNN(input_size, hidden_size, n_classes)
            print("\n" + "="*50)
            print("   MODEL TYPE: CNN + RNN")
            print("="*50 + "\n")
        elif MODEL_TYPE == 'rnn':
            model = RNN(input_size, hidden_size, n_classes)
            print("\n" + "="*50)
            print("   MODEL TYPE: RNN")
            print("="*50 + "\n")
        elif MODEL_TYPE == 'cnn':
            model = CNN(input_size, n_classes)
            print("\n" + "="*50)
            print("   MODEL TYPE: CNN")
            print("="*50 + "\n")

        dummy_input = tf.zeros((1, n_steps, input_size))
        model(dummy_input)
        model.summary()


        model.compile(optimizer=optimizers.Adam(learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

        train_dataset, test_dataset = getData()

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
                # print(f"Training batch - X shape: {X.shape}") 
                with tf.GradientTape() as tape:
                    predictions = model(X, training=True)
                    loss = loss_fn(Y, predictions)

                grads = tape.gradient(loss, model.trainable_variables)
                # Check the existence of gradients and exclude None values.
                grads = [g for g in grads if g is not None]

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
                predictions = model(X, training=False)
                loss = loss_fn(Y, predictions)

                val_loss_avg.update_state(loss)
                val_accuracy.update_state(Y, predictions)

            validation_loss.append(val_loss_avg.result().numpy())
            validation_acc.append(val_accuracy.result().numpy())

            print(f'Validation Loss: {validation_loss[-1]:.3f}, Validation Accuracy: {validation_acc[-1]:.3f}')

            if validation_acc[-1] > best_acc:
                best_acc = validation_acc[-1]
                model_json = model.to_json()
                with open("model.json", "w") as json_file:
                    json_file.write(model_json)

                model.save_weights("model_weights.h5")
                print("Model architecture and weights saved successfully!")

        time_interval = datetime.now() - start_time
        print(f"Training completed in {time_interval.seconds:.2f} seconds, Best Accuracy: {best_acc:.3f}")

        draw_train(train_acc, validation_acc, train_loss, validation_loss)

if __name__ == "__main__":
    main()
