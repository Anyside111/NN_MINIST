from matplotlib import pyplot as plt
import numpy as np

import load_and_pre as lp


# Part 1: Single-layer Neural Network
# model parameters class
class ModelParameters:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes
        self.W, self.b = self.initialize_parameters(input_size, num_classes)

    def initialize_parameters(self, input_size, num_classes):
        # shape of W: (num_classes, input_size); shape of b: (num_classes, 1)
        W = np.random.randn(num_classes, input_size) * 0.01
        b = np.zeros((num_classes, 1))
        return W, b


# training parameters class
class TrainingParameters:
    def __init__(self, images_train, labels_train, images_test, labels_test, learning_rate=0.05, batch_size=10):
        self.images_train = images_train
        self.labels_train = labels_train.flatten()
        self.images_test = images_test
        self.labels_test = labels_test.flatten()
        self.learning_rate = learning_rate
        self.batch_size = batch_size


# NN class
class SingleLayerNN:
    def __init__(self, model_parameters, training_parameters):
        self.W = model_parameters.W
        self.b = model_parameters.b
        self.train_params = training_parameters

    # Softmax
    @staticmethod
    def softmax(z):
        e_z = np.exp(z - np.max(z)) # for numerical stability
        return e_z / e_z.sum(axis=0)

    # cross-entropy loss
    @staticmethod
    def compute_loss(Y, E):
        # print(f"E shape: {E.shape}, Y shape: {Y.shape}")
        m = Y.shape[0]
        if m == 1:
            log_likelihood = -np.log(E[Y, 0])  # single label
        else:
            log_likelihood = -np.log(E[Y, range(m)])  # select the values of E that correspond to labels
        return np.sum(log_likelihood) / m  # average loss over the batch

    # forward propagation
    def forward_propagation(self, X):
        Z = np.dot(self.W, X) + self.b
        return self.softmax(Z)


    # backward propagation
    def backward_propagation(self, X, Y, E, m):
        dZ = E.copy()
        if m == 1:
            dZ[Y, 0] -= 1
        else:
            dZ[Y, range(m)] -= 1
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        return dW, db

    # update network parameters
    def update_parameters(self, dW, db):
        self.W -= self.train_params.learning_rate * dW
        self.b -= self.train_params.learning_rate * db

    # inference
    def predict(self, X):
        E = self.forward_propagation(X)
        return np.argmax(E, axis=0)

    # train the model
    def train_model(self, test_during_training=True):
        total_loss = 0
        test_accuracies = []
        total_samples = self.train_params.images_train.shape[0]
        num_iterations = total_samples // self.train_params.batch_size

        for i in range(num_iterations):
            # Adjust evaluation frequency
            if i < 50:
                eval_frequency = 10
            elif i < 100:
                eval_frequency = 20
            else:
                eval_frequency = 100
            #eval_frequency = 100 # use for different batch_size
            # Mini-batch Gradient Descent
            start_index = i * self.train_params.batch_size
            end_index = min(start_index + self.train_params.batch_size, total_samples)

            X_batch = self.train_params.images_train[start_index:end_index].T
            Y_batch = self.train_params.labels_train[start_index:end_index].flatten()
            m_batch = X_batch.shape[1]

            # forward propagation
            E = self.forward_propagation(X_batch)

            # Compute loss
            loss = self.compute_loss(Y_batch, E)
            total_loss += loss

            # backward propagation
            dW, db = self.backward_propagation(X_batch, Y_batch, E, m_batch)
            self.update_parameters(dW, db)

            # print loss and Evaluate on test data every eval_frequency iterations for current model
            if test_during_training and (i % eval_frequency == 0 or i == num_iterations - 1):
                test_predictions = self.predict(self.train_params.images_test.T)
                test_acc = np.mean(test_predictions == self.train_params.labels_test)
                test_accuracies.append(test_acc)
                avg_loss = total_loss / (m_batch * (i + 1)) # cumulative losses
                print(f"Iteration {i+1}: Average Loss = {avg_loss}, Test Accuracy = {test_acc}")
                total_loss = 0 # Reset the total loss for the next set of iterations

        return test_accuracies

    def test_model(self):
        test_predictions = self.predict(self.train_params.images_test.T)
        test_acc = np.mean(test_predictions == self.train_params.labels_test)
        print(f"Test Accuracy: {test_acc}")
        return test_acc


    def train_model_reduced_data(self, num_evaluations=10):
        total_loss = 0
        test_accuracies = []
        total_samples = self.train_params.images_train.shape[0]
        original_batch_size = self.train_params.batch_size
        self.train_params.batch_size = 1
        num_iterations = min(total_samples, num_evaluations)

        for i in range(num_iterations):
            start_index = i
            end_index = i + 1

            X_batch = self.train_params.images_train[start_index:end_index].T
            Y_batch = self.train_params.labels_train[start_index:end_index].flatten()
            m_batch = X_batch.shape[1]

            # forward propagation
            E = self.forward_propagation(X_batch)

            # Compute loss
            loss = self.compute_loss(Y_batch, E)
            total_loss += loss

            # backward propagation
            dW, db = self.backward_propagation(X_batch, Y_batch, E, m_batch)
            self.update_parameters(dW, db)

            # Evaluate on test data for current model
            test_predictions = self.predict(self.train_params.images_test.T)
            test_acc = np.mean(test_predictions == self.train_params.labels_test)
            test_accuracies.append(test_acc)
            avg_loss = total_loss / (i + 1)  # cumulative losses
            print(f"Iteration {i+1}: Average Loss = {avg_loss}, Test Accuracy = {test_acc}")
            self.train_params.batch_size = original_batch_size

        return test_accuracies



# Part 6: add a hidden layer
class MultiLayerNN(SingleLayerNN):
    def __init__(self, model_parameters, training_parameters, hidden_size=100):
        super(MultiLayerNN, self).__init__(model_parameters, training_parameters)
        self.hidden_size = hidden_size
        self.W1, self.b1, self.W2, self.b2 = self.initialize_parameters_with_hidden_layer(model_parameters)

    def initialize_parameters_with_hidden_layer(self, model_parameters):
        W1 = np.random.randn(self.hidden_size, self.train_params.images_train.shape[1]) * 0.01
        b1 = np.zeros((self.hidden_size, 1))
        W2 = np.random.randn(model_parameters.num_classes, self.hidden_size) * 0.01
        b2 = np.zeros((model_parameters.num_classes, 1))
        return W1, b1, W2, b2

    def forward_propagation(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.maximum(0, Z1)  # ReLU activation
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def backward_propagation(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[1]
        # the same as in single-layer NN
        dZ2 = A2.copy()
        if m == 1:
            dZ2[Y, 0] -= 1
        else:
            dZ2[Y, range(m)] -= 1
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        # ReLU derivative
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.train_params.learning_rate * dW1
        self.b1 -= self.train_params.learning_rate * db1
        self.W2 -= self.train_params.learning_rate * dW2
        self.b2 -= self.train_params.learning_rate * db2

    def predict(self, X):
        _, _, _, A2 = self.forward_propagation(X)
        return np.argmax(A2, axis=0)

    def train_model(self, test_during_training=True):
        total_loss = 0
        test_accuracies = []
        total_samples = self.train_params.images_train.shape[0]
        num_iterations = total_samples // self.train_params.batch_size

        for i in range(num_iterations):
            # Adjust evaluation frequency
            if i < 50:
                eval_frequency = 10
            elif i < 100:
                eval_frequency = 20
            else:
                eval_frequency = 100
            # Mini-batch Gradient Descent
            start_index = i * self.train_params.batch_size
            end_index = min(start_index + self.train_params.batch_size, total_samples)
            X_batch = self.train_params.images_train[start_index:end_index].T
            Y_batch = self.train_params.labels_train[start_index:end_index].flatten()
            m_batch = X_batch.shape[1]

            # forward propagation
            Z1, A1, Z2, A2 = self.forward_propagation(X_batch)
            # Compute loss
            loss = self.compute_loss(Y_batch, A2)
            total_loss += loss
            # backward propagation
            dW1, db1, dW2, db2 = self.backward_propagation(X_batch, Y_batch, Z1, A1, Z2, A2)
            self.update_parameters(dW1, db1, dW2, db2)

            # print loss and Evaluate on test data every eval_frequency iterations for current model
            if test_during_training and (i % eval_frequency == 0 or i == num_iterations - 1):
                test_predictions = self.predict(self.train_params.images_test.T)
                test_acc = np.mean(test_predictions == self.train_params.labels_test)
                test_accuracies.append(test_acc)
                avg_loss = total_loss / (m_batch * (i + 1))
                print(f"Iteration {i+1}: Average Loss = {avg_loss}, Test Accuracy = {test_acc}")

        return test_accuracies

