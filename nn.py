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
	def __init__(self, n_input_nodes, n_output_nodes, hidden_layers, weights=None):
		self.n_input_nodes = n_input_nodes
		self.n_output_nodes = n_output_nodes
		self.hidden_layers = hidden_layers
		self.activations = [None] * (len(hidden_layers) + 1)
		self.weights = weights

		# assign initial weights if none are given (i.e. if network is not trained)
		if weights is None:
			self.weights = self.getInitialWeights()
		else:
			# ensure given weights match network dimensions
			self.checkWeights()

	# description: randomly initializes network weights 
	def getInitialWeights(self):
		weights = []

		# seed rng for reproducible results
		np.random.seed(1)

		for i in range(len(self.hidden_layers) + 1):
			if i == 0:
				weights.append(np.random.randn(self.n_input_nodes, self.hidden_layers[0]))
			elif i != len(self.hidden_layers):
				weights.append(np.random.randn(self.hidden_layers[i - 1], self.hidden_layers[i]))
			else:
				weights.append(np.random.randn(self.hidden_layers[-1], self.n_output_nodes))

		# reset rng (deseed)
		np.random.seed()

		return np.array(weights)

	# description: verifies that given weights match the shape of the neural network
	def checkWeights(self):
		expected_weights = self.getInitialWeights()

		if expected_weights.shape != self.weights.shape:
			sys.exit('Error: given weights do not match neural network dimensions')

		for i in range(len(self.weights)):
			if expected_weights[i].shape != self.weights[i].shape:
				sys.exit('Error: given weights do not match neural network dimensions')

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
	def backpropagation(self, x, y):
		deltas = []

		deltas.append((y - self.activations[-1]) * sigmoid_derivative(self.activations[-1]))

		for i in range(len(self.hidden_layers), 0, -1):
			deltas.append((np.dot(deltas[-1], self.weights[i].T)) * sigmoid_derivative(self.activations[i - 1]))
			
		deltas.reverse()

		for i in range(len(self.weights)):
			if i == 0:
				layer = x
			else:
				layer = self.activations[i - 1]

			layer.shape = (len(layer), 1)

			delta = deltas[i]
			delta.shape = (1, len(delta))

			self.weights[i] += np.dot(layer, delta)

	# description: top-level function that trains the neural network
	# param: x (list of inputs), training data inputs
	# param: y (list of outputs), training data outputs
	# param: epochs (int), number of times the neural network is trained
	# param: batch_size (int), number of input samples to use in each epoch
	def train(self, x, y, epochs=1000, batch_size=500):

		# normalize inputs
		x = np.array(x)
		x = x / np.amax(x)

		print('Training started\nBatch size: {}\nTotal epochs: {}'.format(batch_size, epochs))

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
				self.backpropagation(inputs[j], outputs[j])

		print('\nTraining complete')

		# save network weights
		pickle.dump(self.weights, open('pickle/mnist_weights.p', 'wb'))

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
		pickle.dump(incorrect, open('pickle/mnist_incorrect.p', 'wb'))

		print('Testing accuracy:', correct / len(x))

np.warnings.filterwarnings('ignore')