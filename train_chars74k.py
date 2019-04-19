import pickle
from nn import NeuralNetwork

# get chars74k training data
print('Loading chars74k data...', end=' ', flush=True)
train_input = pickle.load(open('data/chars74k/train_input.p', 'rb'))
train_output = pickle.load(open('data/chars74k/train_output.p', 'rb'))
print('done', flush=True)

# neural network parameters
n_input_nodes = 1200
n_output_nodes = 52

# number and size of hidden layers
# example: for a hidden layer of 50 nodes
#          followed by a hidden layer of 25
#          nodes, write: hidden_layers = [50, 25]
hidden_layers = [128, 64]

# create neural network
print('Creating neural network...', end=' ', flush=True)
n = NeuralNetwork(n_input_nodes, n_output_nodes, hidden_layers)
print('done', flush=True)

# training parameters
n_epochs = 2000
batch_size = 500

# train the network
n.train(train_input, train_output, n_epochs, batch_size)