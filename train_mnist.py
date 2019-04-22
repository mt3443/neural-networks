from nn import NeuralNetwork
from mnist import MNIST

# get mnist training data
print('Loading MNIST data...', end=' ', flush=True)
mndata = MNIST('data/mnist/')
mndata.gz = True
train_input, train_output = mndata.load_training()
print('done', flush=True)

# neural network parameters
n_input_nodes = 784
n_output_nodes = 10

# number and size of hidden layers
# example: for a hidden layer of 50 nodes
#          followed by a hidden layer of 25
#          nodes, write: hidden_layers = [50, 25]
hidden_layers = [50]

# create neural network
print('Creating neural network...', end=' ', flush=True)
n = NeuralNetwork(n_input_nodes, n_output_nodes, hidden_layers)
print('done', flush=True)

# training parameters
n_epochs = 1000
batch_size = 6000

# train the network
n.train(train_input, train_output, n_epochs, batch_size)