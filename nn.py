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

	def get_error(self, y):
		return np.sum(0.5 * np.square(y - self.activations[-1]))

	def backpropagation(self, x, y):
		output_delta = (y - self.activations[-1]) * sigmoid_derivative(self.activations[-1])
		hidden_delta = (np.dot(output_delta, self.weights[1].T)) * sigmoid_derivative(self.activations[0])

		x_t = x
		x_t.shape = (len(x), 1)

		hidden_delta_t = hidden_delta
		hidden_delta_t.shape = (1, len(hidden_delta))

		hidden_layer_t = self.activations[0]
		hidden_layer_t.shape = (len(self.activations[0]), 1)

		output_delta_t = output_delta
		output_delta_t.shape = (1, len(output_delta))

		self.weights[0] += np.dot(x_t, hidden_delta_t)
		self.weights[1] += np.dot(hidden_layer_t, output_delta_t)

	def train(self, x, y, epochs=1000, batch_size=500):

		# normalize inputs
		x = np.array(x)
		x = x / np.amax(x)

		print('Training started\nBatch size:{}\nTotal epochs:{}'.format(batch_size, epochs))

		for i in range(epochs):
			i % 100 == 0 and print('Current epoch: {}'.format(i))

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

		print('Training complete')

	def getBatch(self, x,  y, batch_size):
		inputs = []
		outputs = []

		sample_indices = random.sample(range(len(x)), batch_size)

		for index in sample_indices:
			inputs.append(x[index])
			outputs.append(y[index])

		return np.array(inputs), outputs

	def predict(self, x):
		self.feedforward(x)
		return np.argmax(self.activations[-1])

	def test(self, x, y):
		# counter recording number of correct classifications
		correct = 0

		# normalize inputs
		x = np.array(x)
		x = x / np.amax(x)

		# classify test data
		for i in range(len(x)):
			prediction = self.predict(x[i])
			if prediction == y[i]:
				correct += 1

		print('Accuracy:', correct / len(x))