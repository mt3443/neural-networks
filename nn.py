import numpy as np
import sys

class NeuralNetwork:

	# hidden_layers = array of ints, index = layer number, int = number of nodes in that layer
	def __init__(self, n_input_nodes, n_output_nodes, hidden_layers, weights=None):
		self.n_input_nodes = n_input_nodes
		self.n_output_nodes = n_output_nodes
		self.hidden_layers = hidden_layers
		self.layers = [None] * (len(hidden_layers) + 1)
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
				weights.append(np.random.rand(self.n_input_nodes, self.hidden_layers[0]))
			elif i != len(self.hidden_layers):
				weights.append(np.random.rand(self.hidden_layers[i - 1], self.hidden_layers[i]))
			else:
				weights.append(np.random.rand(self.hidden_layers[-1], self.n_output_nodes))

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

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		return x * (1 - x)

	def forward(self, x):
		self.layers[0] = self.sigmoid(np.dot(x, self.weights[0]))

		for i in range(len(self.hidden_layers) - 1):
			self.layers[i + 1] = self.sigmoid(np.dot(self.layers[i], self.weights[i + 1]))

		self.layers[-1] = self.sigmoid(np.dot(self.layers[-2], self.weights[-1]))

	def backpropagation(self, x, y):
		# implement me

	def train(self, x, y, epochs):
		for i in range(epochs):
			for j in range(len(x)):
				temp_input = x[j].flatten()
				temp_output = np.zeros(10)
				temp_output[y[j]] = 1
				self.forward(temp_input)
				self.backpropagation(temp_input, temp_output)

	def predict(self, x):
		self.forward(x)
		return np.argmax(self.layers[-1])