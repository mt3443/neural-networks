from nn import NeuralNetwork
from mnist import MNIST
import pickle
import sys

if len(sys.argv) != 2:
	print('Error! Usage: python3 test_mnist.py <path to weights file>')
else:
	# get specified weights
	weights = pickle.load(open(sys.argv[1], 'rb'))

	# create a nerual network with those weights
	n = NeuralNetwork(weights=weights)

	# get mnist test data
	mndata = MNIST('data/')
	mndata.gz = True
	test_input, test_output = mndata.load_testing()

	# test neural network
	n.test(test_input, test_output)