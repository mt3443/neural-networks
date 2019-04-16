from nn import NeuralNetwork
from mnist import MNIST

mndata = MNIST('data/')
mndata.gz = True
train_input, train_output = mndata.load_training()
test_input, test_output = mndata.load_testing()

n_input_nodes = 784
n_output_nodes = 10
hidden_layers = [100]

# n_input_nodes = 3
# n_output_nodes = 4
# hidden_layers = [4]

# train_input = [[1, 0, 4], [2, 3, 5], [0, 0, 1], [2, 1, 3], [4, 3, 5], [1, 0, 2], [3, 1, 2]]
# train_output = [2, 1, 0, 2, 3, 2, 1]

n = NeuralNetwork(n_input_nodes, n_output_nodes, hidden_layers)

n_epochs = 1000
n.train(train_input, train_output, n_epochs, batch_size=20000)