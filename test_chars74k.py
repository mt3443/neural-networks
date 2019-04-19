from nn import NeuralNetwork
import pickle
import sys

if len(sys.argv) != 2:
	print('Error! You must specify which model you want to test.')
	print('Usage: python3 test_chars74k.py <path to weights file>')
	print('Example: python3 test_chars74k.py pickle/300_chars74k_weights.p')
else:
	# get specified weights
	weights = pickle.load(open(sys.argv[1], 'rb'))

	# create a nerual network with those weights
	n = NeuralNetwork(weights=weights)

	# get chars74k test data
	test_input = pickle.load(open('data/chars74k/test_input.p', 'rb'))
	test_output = pickle.load(open('data/chars74k/test_output.p', 'rb'))

	# test neural network
	n.test(test_input, test_output)