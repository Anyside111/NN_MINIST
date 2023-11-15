import numpy as np
from matplotlib import pyplot as plt

import load_and_pre as lp
import Neural_Net as sln


class TestVisualization:
    def __init__(self, model_params, training_params):
        self.model = model_params
        self.training_params = training_params


    # part 1-1: Plot the accuracies during one epoch training
    def plot_accuracies_during_one_epoch(self):
        neural_network = sln.SingleLayerNN(self.model, self.training_params)
        print("Epoch 1")
        test_accuracies = neural_network.train_model()
        np.savez('model_parameters_one_epoch.npz', W=neural_network.W, b=neural_network.b)
        plt.plot(test_accuracies, label='Test Accuracy during Training')
        plt.title('Accuracy of the model of a single-layer neural network during one epoch')
        plt.xlabel('Evaluation Point')
        plt.ylabel('Accuracy')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend()
        plt.show()


    # Part 1-2: Plot the accuracies during 10 epochs training
    def plot_accuracies_during_epochs(self, num_epochs=10):
        neural_network = sln.SingleLayerNN(self.model, self.training_params)
        epoch_accuracies = []
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}")
            neural_network.train_model(test_during_training=False) # won't evaluate during training
            test_acc = neural_network.test_model() # test per epoch
            epoch_accuracies.append(test_acc)
        np.savez('model_parameters_ten_epoch.npz', W=neural_network.W, b=neural_network.b)
        #print(f"Test Accuracy: {epoch_accuracies}")
        plt.plot(range(1, num_epochs + 1), epoch_accuracies, label='Test Accuracy per Epoch')
        plt.title('Accuracy of the model of a single-layer neural network during epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend()
        plt.show()

    # Part 2: Visualize misclassified images
    def visualize_misclassified_images(self, model_parameters_file):
        data = np.load(model_parameters_file)
        W = data['W']
        b = data['b']

        neural_network = sln.SingleLayerNN(self.model, self.training_params)
        neural_network.W = W
        neural_network.b = b

        neural_network = sln.SingleLayerNN(self.model, self.training_params)
        test_predictions = neural_network.predict(self.training_params.images_test.T)
        E = neural_network.forward_propagation(self.training_params.images_test.T)
        confidence_scores = np.max(E, axis=0)

        misclassified = {i: [] for i in range(10)}
        for idx, (true_label, predicted_label) in enumerate(zip(self.training_params.labels_test, test_predictions)):
            if true_label != predicted_label:
                misclassified[true_label].append((idx, predicted_label, confidence_scores[idx]))

        for i in range(10):
            misclassified[i].sort(key=lambda x: x[2], reverse=True)
            misclassified[i] = misclassified[i][:10]

        for class_idx in misclassified:
            plt.figure(figsize=(15, 1.5))
            for j, (idx, pred_label, score) in enumerate(misclassified[class_idx]):
                image = self.training_params.images_test[idx].reshape(28, 28)
                plt.subplot(1, 10, j + 1)
                plt.imshow(image, cmap='gray')
                plt.title(f"True: {class_idx}\nPred: {pred_label}\nScore: {score:.2f}")
                plt.axis('off')
            plt.tight_layout()
            plt.show()


    # Part 3: Reduce the training data
    def test_reduced_training_data(self):
        reduced_images_train, reduced_labels_train = lp.reduce_training_data(
            self.training_params.images_train, self.training_params.labels_train, num_examples_per_class=1)

        reduced_train_params = sln.TrainingParameters(reduced_images_train, reduced_labels_train,
                                                  self.training_params.images_test, self.training_params.labels_test,
                                                  learning_rate=0.05, batch_size=1)
        neural_network = sln.SingleLayerNN(self.model, reduced_train_params)
        test_accuracies = neural_network.train_model_reduced_data(num_evaluations=10)

        if test_accuracies:
            print(f"Test Accuracy with reduced training data: {test_accuracies[0]}")
        else:
            print("No test accuracies recorded.")
        np.savez('model_parameters_reduced_data.npz', W=neural_network.W, b=neural_network.b)

        x_vals = list(range(1, 11, 1))
        plt.plot(x_vals, test_accuracies, label='Test Accuracy on reduced training data')
        plt.title('Accuracy of the model of a neural network with reduced training data')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend()
        plt.show()


    # Part 4: Different batch sizes
    def test_different_batch_sizes(self, total_evaluations):
        batch_sizes = [1, 10, 100]
        batch_size_accuracies = {}

        for batch_size in batch_sizes:
            self.training_params.batch_size = batch_size
            neural_network = sln.SingleLayerNN(self.model, self.training_params)
            batch_accuracies = []
            num_iterations = total_evaluations // batch_size
            num_epochs = total_evaluations//num_iterations

            for _ in range(num_epochs):
                test_acc = neural_network.train_model()
                batch_accuracies.extend(test_acc)

            batch_size_accuracies[batch_size] = batch_accuracies

            x_vals = range(1, total_evaluations+1)
            for current_batch_size, accuracies in batch_size_accuracies.items():
                plt.plot(x_vals, accuracies[:total_evaluations], label=f'Batch Size = {current_batch_size}')

            plt.title('Accuracy of the model of a neural network with different batch sizes')
            plt.xlabel('Evaluation point')
            plt.ylabel('Accuracy')
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.legend()
            plt.show()


    # Part 5: Sort the training data
    def test_sorted_training_data(self):
        sorted_images_train, sorted_labels_train = lp.sort_sample_order(self.training_params.images_train,
                                                                        self.training_params.labels_train)
        self.training_params.images_train = sorted_images_train
        self.training_params.labels_train = sorted_labels_train

        neural_network = sln.SingleLayerNN(self.model, self.training_params)
        test_accuracies = neural_network.train_model()
        np.savez('model_parameters_sorted_data.npz', W=neural_network.W, b=neural_network.b)

        plt.plot(test_accuracies, label='Test Accuracy on sorted training data')
        plt.title('Accuracy of the model of a neural network on sorted training data')
        plt.xlabel('Evaluation Point')
        plt.ylabel('Accuracy')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend()
        plt.show()

    # Part 6: Plot the accuracies of the model with one hidden layer
    # part 6-1: Plot the accuracies during one epoch training
    def plot_accuracies_during_one_epoch_multi(self):
        neural_network = sln.MultiLayerNN(self.model, self.training_params)
        print("Epoch 1")
        test_accuracies = neural_network.train_model()
        print(test_accuracies)
        np.savez('model_parameters_add_hidden_layer.npz', W=neural_network.W, b=neural_network.b)
        plt.plot(test_accuracies, label='Test Accuracy during Training')
        plt.title('Accuracy of the model of a Multi-layer neural network during one epoch')
        plt.xlabel('Evaluation Point')
        plt.ylabel('Accuracy')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend()
        plt.show()


    # Part 6-2: Plot the accuracies during 10 epochs training
    def plot_accuracies_during_epochs_multi(self, num_epochs=10):
        neural_network = sln.MultiLayerNN(self.model, self.training_params)
        epoch_accuracies = []
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}")
            neural_network.train_model(test_during_training=False) # won't evaluate during training
            test_acc = neural_network.test_model() # test per epoch
            epoch_accuracies.append(test_acc)
        print(epoch_accuracies)
        np.savez('model_parameters_ten_epoch.npz', W=neural_network.W, b=neural_network.b)
        #print(f"Test Accuracy: {epoch_accuracies}")
        plt.plot(range(1, num_epochs + 1), epoch_accuracies, label='Test Accuracy per Epoch')
        plt.title('Accuracy of the model of a Multi-layer neural network during epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend()
        plt.show()




if __name__ == "__main__":

    images_train, images_test, labels_train, labels_test = lp.images_train, lp.images_test, lp.labels_train, lp.labels_test
    # images_train = images_train[0:1000, :] # for debug
    # labels_train = labels_train[0:1000, :]
    model_params = sln.ModelParameters(input_size=784, num_classes=10)
    train_params = sln.TrainingParameters(images_train, labels_train, images_test, labels_test, learning_rate=0.05, batch_size=10)
    tv = TestVisualization(model_params, train_params)


    #tv.plot_accuracies_during_one_epoch()
    #tv.plot_accuracies_during_epochs(num_epochs=10)

    #tv.test_different_batch_sizes(600)
    #tv.visualize_misclassified_images(model_parameters_file = "D:/github_projects/NN_MINIST/NN_MINISTmodel_parameters_one_epoch.npz")
    #tv.test_sorted_training_data()
    #tv.test_reduced_training_data()
    tv.plot_accuracies_during_epochs_multi()
    #tv.plot_accuracies_during_one_epoch_multi()


