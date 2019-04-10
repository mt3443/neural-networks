import pickle
import numpy as np
from nn import NeuralNetwork

mnist_data = pickle.load(open('data/mnist.p', 'rb'))
digits_train = mnist_data[0] / 255
labels_train = mnist_data[1]
digits_test = mnist_data[2] / 255
labels_test = mnist_data[3]

n_input_nodes = 784
n_output_nodes = 10
hidden_layers = [100]

n = NeuralNetwork(n_input_nodes, n_output_nodes, hidden_layers)

n_epochs = 20
n.train(digits_train, labels_train, n_epochs)

# print error and implement learning rate