from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat


#Loading the data
M = loadmat('D:/ECS271_ML/NN_MINIST/MNIST_digit_data.mat')
images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']
#just to make all random sequences on all computers the same.
np.random.seed(1)

#randomly permute data points inds
inds = np.random.permutation(images_train.shape[0])
images_train = images_train[inds]
labels_train = labels_train[inds]

inds = np.random.permutation(images_test.shape[0])
images_test = images_test[inds]
labels_test = labels_test[inds]


# Part 3: reduce the training data
def reduce_training_data(images_train, labels_train, num_examples_per_class=1):
    reduced_images_train = []
    reduced_labels_train = []

    for i in range(10):
        class_indices = np.where(labels_train == i)[0]
        selected_indices = np.random.choice(class_indices, num_examples_per_class, replace=False)
        reduced_images_train.append(images_train[selected_indices])
        reduced_labels_train.append(labels_train[selected_indices])

    reduced_images_train = np.vstack(reduced_images_train)
    reduced_labels_train = np.vstack(reduced_labels_train)

    return reduced_images_train, reduced_labels_train


# Part 5: sort the training data
def sort_sample_order(images_train, labels_train):
    sorted_indices = np.argsort(labels_train)
    sorted_images_train = images_train[sorted_indices]
    sorted_labels_train = labels_train[sorted_indices]
    sorted_images_train = np.squeeze(sorted_images_train)
    return sorted_images_train, sorted_labels_train




# # if you want to use only the first 1000 data points.
# images_train = images_train[0:1000,:]
# labels_train = labels_train[0:1000,:]
#
#
# # show the 10'th train image
# i=10
# im = images_train[i,:].reshape((28,28),order='F')
# plt.imshow(im)
# plt.title('Class Label:'+str(labels_train[i][0]))
# plt.show()
