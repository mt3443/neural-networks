from mnist import MNIST
import numpy as np
from nn import NeuralNetwork

mndata = MNIST('data/')
mndata.gz = True
train_input, train_output = mndata.load_training()
test_input, test_output = mndata.load_testing()

# n_input_nodes = 784
# n_output_nodes = 10
# hidden_layers = [100]

n_input_nodes = 3
n_output_nodes = 2
hidden_layers = [4]

train_input = [[1, 0, 2], [2, 4, 6], [2, 3, 1], [1, 2, 4], [2, 1, 4], [0, 0, 2]]
train_output = [0, 1, 0, 1, 1, 1]

n = NeuralNetwork(n_input_nodes, n_output_nodes, hidden_layers)

n_epochs = 1
n.train(train_input, train_output, n_epochs, batch_size=2)