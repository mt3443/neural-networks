from mnist import MNIST
import numpy as np
from nn import NeuralNetwork

mndata = MNIST('data/')
mndata.gz = True
digits_train, labels_train = mndata.load_training()
digits_test, labels_test = mndata.load_testing()

n_input_nodes = 784
n_output_nodes = 10
hidden_layers = [100]

n = NeuralNetwork(n_input_nodes, n_output_nodes, hidden_layers)

n_epochs = 20
n.train(digits_train, labels_train, n_epochs)

# print error and implement learning rate