# Neural Network MNIST Classifier
This project implements a neural network for classifying handwritten digits from the MNIST dataset. It includes scripts for data loading and preprocessing (load_and_pre.py), defining the neural network model (Neural_Net.py), and testing and visualizing the results (test_visualization.py).

# Installation
No specific installation steps are required apart from having Python and necessary libraries installed. Ensure you have numpy and matplotlib installed, as they are crucial for the project.

# Usage
## Step 1: Data Loading and Preprocessing
load_and_pre.py is used for loading and preprocessing the MNIST dataset. It includes functions for reducing and sorting the training data.

To load and preprocess the data, simply run the script. It will prepare the data for training and testing.

## Step 2: Neural Network Definition
Neural_Net.py defines the neural network models. It includes classes for model parameters (ModelParameters), training parameters (TrainingParameters), a single-layer neural network (SingleLayerNN), and a neural network with a hidden layer (HiddenLayerNN).

Import the necessary classes from this script into the main script where one plan to train and evaluate the neural network.

## Step 3: Testing and Visualization
test_visualization.py is for testing and visualizing the performance of the neural network models. It includes a TestVisualization class with methods for plotting accuracies during training, visualizing misclassified images, testing on reduced training data, and experimenting with different batch sizes.

Create an instance of TestVisualization by passing the model and training parameters.
Call the appropriate methods for testing and visualization as per your requirement.

test_visualization.py is the core script for testing and visualizing the network's performance. It includes several functionalities:

### 1. Accuracy Plotting:
Implements the neural network model to classify MNIST images and plots the accuracy on test data for every 'n' iterations.
### 2. Misclassified Images Visualization:
For each class, visualizes the top 10 misclassified images with the highest scores.
### 3. Reduced Training Data:
Reduces training data to one example per class and plots the accuracy curve.
### 4. Batch Size Comparison:
Tests the original model with different mini-batch sizes (1, 10, 100) and plots the results.
### 5. Sorted Data Training:
Trains the model on sorted data and plots the results.
### 6. Hidden Layer Model Training:
Trains a model with an added hidden layer and plots the accuracy curve.
