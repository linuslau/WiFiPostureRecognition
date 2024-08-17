import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")

# Print model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:")
for detail in input_details:
    print(detail)

print("\nOutput details:")
for detail in output_details:
    print(detail)
