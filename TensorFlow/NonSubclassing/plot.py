import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

from model import *  # Import models from model.py

def plot_feature_map(feature_map, layer_name):
    """
    Plots the feature map and adds annotations to help with understanding.

    Parameters:
    - feature_map: The feature map data
    - layer_name: The name of the current layer
    """
    num_filters = feature_map.shape[-1]
    size = feature_map.shape[1]

    # Calculate grid size
    grid_size = int(num_filters ** 0.5) + (1 if int(num_filters ** 0.5)**2 < num_filters else 0)

    # Create a new figure with the layer name as the window title
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.canvas.manager.set_window_title(f'Feature Maps - {layer_name}')  # Set the window title to the layer name

    for i in range(grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size

        if i < num_filters:
            # Check if the feature map is 2D or 1D
            if len(feature_map.shape) == 3:
                # Use plot to draw 1D data
                ax[row, col].plot(feature_map[0, :, i])
                ax[row, col].set_title(f"Filter {i+1}", fontsize=8)
            else:
                # Use imshow to draw 2D data
                ax[row, col].imshow(feature_map[0, :, :, i], aspect='auto', cmap='viridis')
                ax[row, col].set_title(f"Filter {i+1}", fontsize=8)
            ax[row, col].axis('off')
        else:
            ax[row, col].remove()

    # Add an overall title and simple layer explanation
    plt.suptitle(f"Feature maps in layer: {layer_name}\n"
                 f"Conv layers highlight features, Pool layers reduce dimensions.", fontsize=16)
    plt.show()

def visualize_feature_maps(model, data):
    """
    Visualizes the feature maps of each layer and adds annotations to each image.

    Parameters:
    - model: The trained Keras model
    - data: Input data (shape should match the model input)
    """
    layer_names = [layer.name for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]
    outputs = [model.get_layer(name).output for name in layer_names]

    # Create a new model that returns the outputs of the intermediate layers
    visualization_model = models.Model(inputs=model.input, outputs=outputs)

    # Get the outputs of the intermediate layers
    feature_maps = visualization_model.predict(data)

    for layer_name, feature_map in zip(layer_names, feature_maps):
        print(f"Visualizing {layer_name} output")
        plot_feature_map(feature_map, layer_name)

def plot_accuracy_and_loss(train_acc, validation_acc, train_loss, validation_loss):
    """
    Plot the accuracy and loss curves for the training and validation datasets,
    with added annotations to help better understand these curves.

    Parameters:
    - train_acc: List of accuracy values for the training dataset
    - validation_acc: List of accuracy values for the validation dataset
    - train_loss: List of loss values for the training dataset
    - validation_loss: List of loss values for the validation dataset
    """

    # Create a new figure for the plots
    fig = plt.figure(figsize=(10, 8))

    # Use suptitle to set a title for the figure
    fig.canvas.manager.set_window_title('Training and Validation Metrics')

    # First subplot: Training and Validation Accuracy
    plt.subplot(2, 1, 1)
    plt.plot(train_acc, label="Train Accuracy", color="blue")
    plt.plot(validation_acc, label="Validation Accuracy", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend(loc="best")
    plt.ylim([0, 1])

    # Add annotation at the bottom center
    plt.text(0.5, -0.2, "Monitor if accuracy converges between training and validation",
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="yellow"),
             fontsize=10)

    # Second subplot: Training and Validation Loss
    plt.subplot(2, 1, 2)
    plt.plot(train_loss, label="Train Loss", color="red")
    plt.plot(validation_loss, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend(loc="best")

    # Adjust y-axis limit to better display the data
    plt.ylim([0, max(max(train_loss), max(validation_loss)) + 0.5])

    # Add annotation explaining the behavior of validation loss at the bottom center
    plt.text(0.5, -0.2, "Validation loss should decrease over time and stabilize",
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="yellow"),
             fontsize=10)

    # Automatically adjust subplot spacing
    plt.tight_layout()

    # Display the plots
    plt.show()

def visualize_flatten_dense_layer_output(model, data):
    """
    Visualize the output of the flatten and dense layers.

    Parameters:
    - model: The trained Keras model
    - data: Input data (shape should match the model input)
    """
    # Get the names of the layers to be visualized (e.g., flatten and dense layers)
    dense_layer_names = [layer.name for layer in model.layers if 'flatten' in layer.name or 'output' in layer.name]
    dense_outputs = [model.get_layer(name).output for name in dense_layer_names]

    # Create a new model that returns the outputs of the specified layers
    visualization_model = models.Model(inputs=model.input, outputs=dense_outputs)

    # Predict to get the outputs of these layers
    dense_outputs_values = visualization_model.predict(data)

    for layer_name, output_values in zip(dense_layer_names, dense_outputs_values):
        print(f"Visualizing output for {layer_name}")

        # Determine the display name for the layer
        display_name = "dense" if layer_name == "output" else layer_name

        # Since the outputs of flatten/dense layers are usually 1D, use a line graph to plot them
        fig = plt.figure(figsize=(12, 4))
        plt.plot(output_values.flatten(), label=f'{layer_name} output')
        plt.title(f'{display_name} - Output Visualization')
        plt.xlabel('Index')
        plt.ylabel('Activation')
        plt.legend()

        # Set the window title to reflect the layer being visualized
        fig.canvas.manager.set_window_title(f'{display_name} - Output Visualization')

        plt.show()

def plot_3d_rnn_output(activation):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    time_steps = activation.shape[1]
    units = activation.shape[2]

    X, Y = np.meshgrid(np.arange(time_steps), np.arange(units))
    Z = activation[0, :, :].T  # Get the first batch and transpose for plotting

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Units')
    ax.set_zlabel('Activation')
    plt.title('3D Visualization of RNN Output')
    plt.show()

def visualize_rnn_activations_dbg(model, data):
    print(f"Data shape before predict: {data.shape}")

    # Extract the names and outputs of the RNN layers (LSTM, GRU, etc.)
    layer_names = [layer.name for layer in model.layers if 'lstm' in layer.name or 'rnn' in layer.name]
    outputs = [model.get_layer(name).output for name in layer_names]

    # Create a new model that returns the outputs of the RNN layers
    visualization_model = models.Model(inputs=model.input, outputs=outputs)

    # Get the activations for the input data
    activations = visualization_model.predict(data)

    # Check if activations is a list (in case of multiple layers) or a single output
    if not isinstance(activations, list):
        activations = [activations]

    # Ensure activations have the correct shape and add batch dimension if missing
    for i, activation in enumerate(activations):
        print(f"Activation {i} shape before adjustment: {activation.shape}")
        if len(activation.shape) == 2:  # If 2D, add a batch dimension
            activation = np.expand_dims(activation, axis=0)
        print(f"Activation {i} shape after adjustment: {activation.shape}")

        # Plot the activation (now ensuring it's 3D)
        plot_3d_rnn_output(activation)

def visualize_rnn_activations(model, data):
    """
    Visualize the activations of RNN layers, handling both 1D, 2D, and 3D outputs.
    
    Parameters:
    - model: The Keras model containing RNN layers.
    - data: The input data for which to visualize the activations.
    """
    
    print(f"Data shape before predict: {data.shape}")
    
    # Extract the names and outputs of the RNN layers (LSTM, GRU, etc.)
    layer_names = [layer.name for layer in model.layers if 'lstm' in layer.name or 'rnn' in layer.name]
    outputs = [model.get_layer(name).output for name in layer_names]

    # Create a new model that returns the outputs of the RNN layers
    visualization_model = models.Model(inputs=model.input, outputs=outputs)

    # Get the activations for the input data
    activations = visualization_model.predict(data)

    # Ensure batch dimension is preserved
    # activations = visualization_model.predict(data, batch_size=data.shape[0])

    # activations = visualization_model.predict(data, batch_size=1)

    # print(f"Activation shape after predict: {activations.shape}")

    # Ensure activations have the correct shape
    # Warning!!! activations is totally different from activation
    # This part of code will make 3D to 2D, every issue is introduced by this.
    for i, activation in enumerate(activations):
        print(f"Activation {i} shape after predict: {activation.shape}")

    # Code below also can help convert from 2D to 3D, donm't know the reason
    # At last, this way is good instead of expand_dims
    # Check if activations is a list (in case of multiple layers) or a single output
    if not isinstance(activations, list):
        activations = [activations]

    # This part of code will make 3D to 2D, every issue is introduced by this.
    for layer_name, activation in zip(layer_names, activations):
        print(f"Visualizing RNN activations for {layer_name}")
        print(f"Shape of {layer_name} activation before expanding: {activation.shape}")
        print(f"Activation values: {activation}")  # Log the activation values
        
        # This part of code will make 2D to 3D, but final result will change incorrectly.
        # activation = np.expand_dims(activation, axis=0)
        # print(f"Shape of {layer_name} activation after expanding: {activation.shape}")

        if len(activation.shape) == 1:
            # Handle 1D output (edge case)
            print(f"1D activation detected for {layer_name} with shape {activation.shape}")
            fig = plt.figure(figsize=(10, 5))
            plt.plot(activation, label=f"Activation for {layer_name}")
            plt.title(f"{layer_name} Activation (1D Output)")
            plt.xlabel("Units")
            plt.ylabel("Activation Value")
            plt.legend()
            fig.canvas.manager.set_window_title(f'{layer_name} - Output Visualization')
            plt.show()

        elif len(activation.shape) == 2:
            # Handle 2D output: (batch_size, units)
            print(f"2D activation detected for {layer_name} with shape {activation.shape}")
            fig = plt.figure(figsize=(10, 5))
            plt.plot(activation[0], label=f"Activation for {layer_name}")
            plt.title(f"{layer_name} Activation (2D Output)")
            plt.xlabel("Units")
            plt.ylabel("Activation Value")
            plt.legend()
            fig.canvas.manager.set_window_title(f'{layer_name} - Output Visualization')
            plt.show()

        elif len(activation.shape) == 3:
            # Handle 3D output: (batch_size, sequence_length, units)
            print(f"3D activation detected for {layer_name} with shape {activation.shape}")
            num_units = activation.shape[-1]
            fig = plt.figure(figsize=(12, 6))
            for i in range(min(num_units, 8)):  # Limit to first 8 units for clarity
                plt.plot(activation[0, :, i], label=f"Unit {i+1}")
            plt.title(f"{layer_name} Activation (3D Output)")
            plt.xlabel("Sequence length")
            plt.ylabel("Activation")
            plt.legend()
            fig.canvas.manager.set_window_title(f'{layer_name} - Output Visualization')
            plt.show()
            # Plot the activation (now ensuring it's 3D)
            plot_3d_rnn_output(activation)

        else:
            print(f"Unexpected shape for {layer_name}: {activation.shape}. Skipping visualization.")
            
def plot_results(predictions, true_labels):
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Ensure that the accuracy per class and class_labels have the same length
    cm_len = cm.shape[0]
    if cm_len < len(class_labels):
        adjusted_class_labels = class_labels[:cm_len]
    else:
        adjusted_class_labels = class_labels

    # Accuracy per class
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(accuracy_per_class)), accuracy_per_class, tick_label=adjusted_class_labels)
    plt.title('Accuracy per Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.show()