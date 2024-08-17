import tensorflow as tf
from tensorflow.keras.models import model_from_json
from model import RNN, CNN_RNN, CNN  # Import models from model.py

# Global constant
MODEL_TYPE = 'cnn'  # Set model type ('cnn', 'cnn_rnn', 'rnn')

# Load model architecture
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load model from JSON based on selected model type
if MODEL_TYPE == 'cnn_rnn':
    model = model_from_json(loaded_model_json, custom_objects={'CNN_RNN': CNN_RNN})
elif MODEL_TYPE == 'rnn':
    model = model_from_json(loaded_model_json, custom_objects={'RNN': RNN})
elif MODEL_TYPE == 'cnn':
    model = model_from_json(loaded_model_json, custom_objects={'CNN': CNN})

# Call the model once before loading weights to create variables
dummy_input = tf.zeros((1, 500, 90))  # Assuming the input shape is (500, 90)
model(dummy_input)

# Load model weights
model.load_weights('model_weights.h5')

# Convert to TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Use TFLite built-in operations
    tf.lite.OpsSet.SELECT_TF_OPS     # Allow using TensorFlow operations
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved as 'model.tflite'")
