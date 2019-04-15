import random
import numpy as np
import sys

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)

class NeuralNetwork:

	# hidden_layers = array of ints, index = layer number, int = number of nodes in that layer
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

	def checkWeights(self):
		expected_weights = self.getInitialWeights()

		if expected_weights.shape != self.weights.shape:
			sys.exit('Error: given weights do not match neural network dimensions')

		for i in range(len(self.weights)):
			if expected_weights[i].shape != self.weights[i].shape:
				sys.exit('Error: given weights do not match neural network dimensions')

	def feedforward(self, x):
		self.activations[0] = sigmoid(np.dot(x, self.weights[0]))
		
		for i in range(len(self.hidden_layers) - 1):
			self.activations[i + 1] = sigmoid(np.dot(self.activations[i], self.weights[i + 1]))

		self.activations[-1] = sigmoid(np.dot(self.activations[-2], self.weights[-1]))

	def calculate_error(self, y):
		error = 0

		for i in range(self.n_output_nodes):
			error += 0.5 * ((self.activations[-1][i] - y[i]) ** 2)

		return error

	def backpropagation(self, x, y):
		error = (y - self.activations[-1]) * sigmoid_derivative(self.activations[-1])

		a = np.dot(error, self.weights[1].T) * sigmoid_derivative(self.activations[0])
		dw1 = np.dot(x, a)
		dw2 = np.dot(self.activations[0], error)

		self.weights[0] += dw1
		self.weights[1] += dw2

	def train(self, x, y, epochs, batch_size=500):

		for i in range(epochs):
			print('Epoch', i + 1)

			# get a batch
			inputs, outputs = self.getBatch(x, y, batch_size)

			# normalize inputs
			temp_inputs = np.array(inputs) / 255

			# convert outputs to one hot
			temp_outputs = []
			for j in range(batch_size):
				a = np.zeros(self.n_output_nodes, dtype=np.int8)
				a[outputs[j]] = 1
				temp_outputs.append(a)

			# train network
			for j in range(batch_size):
				self.feedforward(temp_inputs[j])
				self.backpropagation(temp_inputs[j], temp_outputs[j])

	def getBatch(self, x,  y, batch_size):
		inputs = []
		outputs = []

		sample_indices = random.sample(range(len(x)), batch_size)

		for index in sample_indices:
			inputs.append(x[index])
			outputs.append(y[index])

		return inputs, outputs

	def predict(self, x):
		self.feedforward(x)
		return np.argmax(self.activations[-1])