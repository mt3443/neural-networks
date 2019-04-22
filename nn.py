import random
import numpy as np
import sys
import pickle

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)

class NeuralNetwork:

	# description: neural network (multilayer perceptron) constructor
	# param: n_input_nodes (int), number of input nodes
	# param: n_output_nodes (int), number of output nodes
	# param: hidden_layers (list of ints), representation of the hidden layers.
	#        the number of elements in the list is the number of hidden layers.
	#        the value at each index is the number of nodes in that layer.
	# param: weights (3d array of floats), neural network weights, meant to be
	#        used to load a previously trained model
	def __init__(self, n_input_nodes=None, n_output_nodes=None, hidden_layers=None, weights=None):
		if n_input_nodes is None and n_output_nodes is None and hidden_layers is None and weights is None:
			sys.exit('Error! You must specify neural network dimensions (n_input_nodes, n_output_nodes, hidden_layers) or pass in an array of weights')
		
		self.n_input_nodes = n_input_nodes
		self.n_output_nodes = n_output_nodes
		self.hidden_layers = hidden_layers
		self.weights = weights

		# assign initial weights if none are given (i.e. if network is not trained)
		if weights is None:
			self.weights = self.getInitialWeights()
		else:
			# set network dimensions to match given weights
			self.n_input_nodes = weights[0].shape[0]
			self.n_output_nodes = weights[-1].shape[1]

			self.hidden_layers = []
			for i in range(len(weights) - 1):
				self.hidden_layers.append(weights[i].shape[1])

		self.activations = [None] * (len(self.hidden_layers) + 1)


	# description: randomly initializes network weights 
	def getInitialWeights(self):
		weights = []

		for i in range(len(self.hidden_layers) + 1):
			if i == 0:
				weights.append(np.random.randn(self.n_input_nodes, self.hidden_layers[0]))
			elif i != len(self.hidden_layers):
				weights.append(np.random.randn(self.hidden_layers[i - 1], self.hidden_layers[i]))
			else:
				weights.append(np.random.randn(self.hidden_layers[-1], self.n_output_nodes))

		return np.array(weights)


	# description: implementation of feedforward algorithm. given an input, this produces
	#              the neural network's output
	# param: x (list of floats), neural network input vector
	def feedforward(self, x):
		self.activations[0] = sigmoid(np.dot(x, self.weights[0]))
		
		for i in range(len(self.hidden_layers) - 1):
			self.activations[i + 1] = sigmoid(np.dot(self.activations[i], self.weights[i + 1]))

		self.activations[-1] = sigmoid(np.dot(self.activations[-2], self.weights[-1]))


	# description: implementation of backpropagation algorithm. this function "trains" the
	#              neural network. errors between produced and expected output backpropagate
	#              through the neural network and adjust weights
	# param: x (list of floats), neural network input
	# param: y (list of floats), expected neural network output
	# param: learning_rate (float), how large of an adjustment to make to the weights
	def backpropagation(self, x, y, learning_rate):

		# record errors in each layer, starting with output layer
		deltas = []
		deltas.append((y - self.activations[-1]) * sigmoid_derivative(self.activations[-1]))
		for i in range(len(self.hidden_layers), 0, -1):
			deltas.append((np.dot(deltas[-1], self.weights[i].T)) * sigmoid_derivative(self.activations[i - 1]))
		deltas.reverse()

		# for every set of weights in the network
		for i in range(len(self.weights)):

			# get current layer
			if i == 0:
				layer = x
			else:
				layer = self.activations[i - 1]

			# get corresponding error
			delta = deltas[i]

			# force matrix dimensions to be compatible with dot product
			layer.shape = (len(layer), 1)
			delta.shape = (1, len(delta))

			# update weights
			self.weights[i] += learning_rate * np.dot(layer, delta)


	# description: top-level function that trains the neural network using SGD
	# param: x (list of inputs), training data inputs
	# param: y (list of outputs), training data outputs
	# param: epochs (int), number of times the neural network is trained
	# param: batch_size (int), number of input samples to use in each epoch
	# param: learning_rate (float), how large of an adjustment to make on each epoch
	def train(self, x, y, epochs, batch_size, learning_rate=1):

		# normalize inputs
		print('Normalizing input data...', end=' ', flush=True)
		x = np.array(x)
		x = x / np.amax(x)
		print('done', flush=True)

		print('Training started with batch_size={}, epochs={}, hidden_layers={}'.format(batch_size, epochs, self.hidden_layers), flush=True)

		for i in range(epochs):
			i % (epochs / 100) == 0 and print('\r{}% complete...'.format(int(100 * i / epochs)), end='')

			# get a random batch
			inputs, temp_outputs = self.getBatch(x, y, batch_size)

			# convert outputs to one hot notation
			outputs = []
			for j in range(batch_size):
				a = np.zeros(self.n_output_nodes, dtype=np.int8)
				a[temp_outputs[j]] = 1
				outputs.append(a)

			# train the network on current batch
			for j in range(len(inputs)):
				self.feedforward(inputs[j])
				self.backpropagation(inputs[j], outputs[j], learning_rate)

		print('\nTraining complete')

		# save network weights
		h = str(self.hidden_layers[0])
		t = 'mnist/' if self.n_output_nodes == 10 else 'chars74k/'
		for i in range(len(self.hidden_layers) - 1):
			h = h + 'x' + str(self.hidden_layers[i + 1])
		file_path = 'weights/' + t + h + '.p'
		pickle.dump(self.weights, open(file_path, 'wb'))
		print('Weights saved to \'' + file_path + '\'')


	# description: selects a random set of training samples
	# param: x (list of inputs), training data inputs
	# param: y (list of outputs), training data outputs
	# param: batch_size (int), number of samples to return
	def getBatch(self, x,  y, batch_size):
		inputs = []
		outputs = []

		sample_indices = random.sample(range(len(x)), batch_size)

		for index in sample_indices:
			inputs.append(x[index])
			outputs.append(y[index])

		return np.array(inputs), outputs


	# description: classifies handwritten digit inputs
	# param: x (list of floats), neural network input sample
	def predict(self, x):
		self.feedforward(x)
		return np.argmax(self.activations[-1])


	# description: checks the accuracy of the neural network.
	#              tests the network on given testing data
	# param: x (list of inputs), testing data inputs
	# param: y (list of outputs), testing data outputs
	def test(self, x, y):

		# counter recording number of correct classifications
		correct = 0

		# normalize inputs
		x = np.array(x)
		x = x / np.amax(x)

		incorrect = []

		# classify test data
		for i in range(len(x)):
			prediction = self.predict(x[i])
			if prediction == y[i]:
				correct += 1
			else:
				incorrect.append(i)

		# record which inputs were incorrectly classified
		h = str(self.hidden_layers[0])
		t = 'mnist/' if self.n_output_nodes == 10 else 'chars74k/'
		for i in range(len(self.hidden_layers) - 1):
			h = h + 'x' + str(self.hidden_layers[i + 1])
		file_path = 'incorrect/' + t + h + '.p'
		pickle.dump(incorrect, open(file_path, 'wb'))
		print('Incorrectly classified sample indices saved to \'' + file_path + '\'')

		print('Testing accuracy:', round(correct / len(x), 5))


# suppress occasional np overflow warnings
np.warnings.filterwarnings('ignore')