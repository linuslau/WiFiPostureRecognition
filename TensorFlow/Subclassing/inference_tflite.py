import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from model import * # Import models from model.py
from dataset import getData  # Assuming you have this data loading function

# Global constants
MODEL_TYPE = 'cnn'  # Set model type ('cnn', 'cnn_rnn', 'rnn')

# Define class labels
class_labels = ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk", "unknown"]

# Disable TensorFlow progress bar
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

# Load the original Keras model
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Create the model based on the selected type
if MODEL_TYPE == 'cnn_rnn':
    model = model_from_json(loaded_model_json, custom_objects={'CNN_RNN': CNN_RNN})
elif MODEL_TYPE == 'rnn':
    model = model_from_json(loaded_model_json, custom_objects={'RNN': RNN})
elif MODEL_TYPE == 'cnn':
    model = model_from_json(loaded_model_json, custom_objects={'CNN': CNN})

# Call the model once before loading weights to create variables
dummy_input = np.random.rand(1, 500, 90).astype(np.float32)
model(dummy_input)

# Load model weights
model.load_weights('model_weights.h5')

# Load test data
_, test_dataset = getData()

# Prepare TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

total_mse = 0
sample_count = 0

# Iterate over the entire test dataset
for i, (real_input, _) in enumerate(test_dataset):
    real_input = real_input.unsqueeze(0).numpy()  # Add a dimension and convert to NumPy array
    
    # Run inference on the Keras model
    keras_output = model.predict(real_input, verbose=0)  # Disable progress bar for each prediction

    # Run inference on the TFLite model
    interpreter.set_tensor(input_details[0]['index'], real_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    # Calculate the difference
    difference = np.abs(keras_output - tflite_output)
    mse = np.mean(np.square(difference))
    total_mse += mse
    sample_count += 1

    # Print results for the first few samples
    if i < 5:
        keras_label = class_labels[np.argmax(keras_output)]
        tflite_label = class_labels[np.argmax(tflite_output)]
        print(f"Sample {i + 1}:")
        print("Keras model output:", keras_output)
        print("TFLite model output:", tflite_output)
        print("Predicted labels - Keras:", keras_label, ", TFLite:", tflite_label)
        print("Difference:", difference)
        print("MSE for this sample:", mse)
        print("----------")

# Calculate average mean squared error
average_mse = total_mse / sample_count
print(f"Average Mean Squared Error between Keras and TFLite model outputs over {sample_count} samples: {average_mse}")
