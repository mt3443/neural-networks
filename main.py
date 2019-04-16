from nn import NeuralNetwork
from mnist import MNIST

mndata = MNIST('data/')
mndata.gz = True
train_input, train_output = mndata.load_training()
test_input, test_output = mndata.load_testing()

n_input_nodes = 784
n_output_nodes = 10
hidden_layers = [100]

n = NeuralNetwork(n_input_nodes, n_output_nodes, hidden_layers)

n_epochs = 1000
n.train(train_input, train_output, n_epochs, batch_size=6000)

n.test(test_input, test_output)