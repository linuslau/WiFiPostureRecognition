# WiFiPostureRecognition

- [WiFiPostureRecognition](#wifiposturerecognition)
  - [Example Usage](#example-usage)
  - [Overview](#overview)
  - [File Structure](#file-structure)
  - [Features](#features)
  - [Getting Started](#getting-started)
  - [Applications](#applications)

## Example Usage
```
pip install -r requirements.txt
python train.py  # Train the model
python inference.py  # Perform model inference
```

## Overview
WiFiPostureRecognition is a project focused on recognizing human postures using Wi-Fi signal data. By leveraging deep learning techniques and Wi-Fi Channel State Information (CSI), this project aims to classify and identify various human postures without the need for cameras or wearable sensors.

## File Structure
- `model.py`: Defines the structures for CNN, RNN, and CNN-RNN models.
- `rain.py`: Used for model training, with support for selecting different model architectures (CNN, RNN, CNN-RNN).
- `inference.py`: Loads trained models and performs inference, outputting prediction results and confusion matrices.

## Features
- Multiple Model Support:
  - CNN: Ideal for processing Wi-Fi CSI data with spatial and temporal features.
  - RNN: Suitable for modeling time-series data, capturing the temporal changes in Wi-Fi signals.
  - CNN-RNN: Combines the strengths of CNN and RNN, first extracting spatial features using CNN and then processing temporal features with RNN, making it suitable for complex posture recognition tasks.
- Flexible Model Architecture:
  - Users can easily switch between CNN, RNN, and CNN-RNN models based on task requirements, providing versatility for different posture recognition scenarios.
- Two Frameworks: Supports both TensorFlow and PyTorch implementations, allowing developers to choose their preferred framework.
- Two Approaches in TensorFlow: The project provides both subclassed and non-subclassed models for flexibility in architecture design.
- Wi-Fi Sensing: Utilizes Wi-Fi CSI data for non-intrusive human posture recognition.
- Deep Learning Models: Implements CNN-RNN and RNN models in both subclassed and non-subclassed formats.
- Model Conversion: Includes scripts for converting .h5 models to TensorFlow Lite (.tflite) format for deployment on MCU platforms like STM32.
- Batch Scripts: Provides batch scripts for various tasks such as training, inference, and model conversion, simplifying the workflow.

## Getting Started
- Install Dependencies: Ensure you have the necessary Python libraries installed by running:

    `pip install -r requirements.txt`

- Choose Your Framework:
  - PyTorch: Contains PyTorch-based implementations for posture recognition.
  - TensorFlow: Contains TensorFlow-based implementations, with both subclassed and non-subclassed models.

- Choose Your Model Type:
  - NonSubclassing: This directory contains scripts and models without subclassing, providing a simpler, more traditional approach.
  - Subclassing: This directory contains more advanced scripts and models that use subclassing for greater flexibility.
        
- Prepare Data: Place your CSI data in the appropriate directory according to the project structure.
- Train the Model: Run the train.py script from either NonSubclassing or Subclassing directory depending on your chosen model type.

- Run Inference: Use inference.py to perform posture recognition on your test data.

- Model Conversion (Optional): If using the subclassed model, you can convert the trained model to TensorFlow Lite format using the provided scripts for deployment on resource-constrained devices like the STM32 MCU.

## Applications
Smart homes and environments
Healthcare monitoring
Human-computer interaction
Security and surveillance systems