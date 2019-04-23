## EECS 738 Project 3: Neural Networks by Matthew Taylor

### Overview

The purpose of this project was to create a neural network that can solve interesting problems. Two datasets were used to ensure the neural network was general and robust, specifically the MNIST and Chars74k datasets. A simple multilayer perceptron was implemented from scratch and trained on these datasets. After training, the neural network was able to achieve a testing accuracy of up to 96%.

### Approach

The MNIST dataset contains samples of handwritten digits 0 through 9. The neural network is simply meant to classify a given digit. Solving this problem is the neural network equivalent of "Hello World". The Chars74k dataset contains handwritten letters, capital and lowercase A through Z, and as with the MNIST dataset, the neural network classifies these letters. Admittedly, these datasets are very similar. However, there are some key differences that will become apparent later.

The neural network's inputs are the pixels of the grayscale images of the handwritten digits or letters. As you might have guessed, the outputs of this network are the possible classes. Ideally, only one of these output nodes will stand out from all others. This unique node represents the neural network's guess as to which class the given input belongs to. The neural network created in this project is incredibly dynamic and can be instantiated with any number of inputs, outputs, and hidden layers. 

Training the neural network utilizes a method called stochastic gradient descent. Given the pool of training data, the neural network collects a subset of these training inputs called a batch. For every sample in this batch, the neural network uses forward propagation to get some result. Initially, this result is completely random. Then, through the use of a loss function, the neural network can get an idea of how it should adjust the parameters (or weights) it used to arrive at its decision. This readjustment of these weights is called backpropagation, which is the foundation of how a neural network learns. After forward and backward propagation occurs for each sample in the current batch, that iteration (or epoch) is considered complete. Then, another epoch begins with a new random batch of inputs. Once the network is trained, it can use simple forward propagation to classify a given input.

### How To Run

This project was written in Python 3.7.2 and requires modules that can be installed using the command:
```
pip3 install numpy python-mnist
```
Once the dependencies are installed, we can create neural networks. **Note: Trained neural networks are saved in this repository. There is no need to train them yourself.** However, if you would like to train the neural network yourself, you can do so with the following commands.

**Important: To change the number of hidden layers and the number of nodes in each hidden layer, you must edit the 'hidden_layers' variable in `train_mnist.py` or `train_chars74k.py`. You can also edit the batch size and number of epochs used for training to achieve different results.**

To train a model with the MNIST dataset:
```
python3 train_mnist.py
```

To train a model with the Chars74k dataset:
```
python3 train_chars74k.py
```

The commands above generate pickled files containing the weights used by the neural networks. Take note of the file path that is printed after training completes, as it is required for testing. To test the neural networks, you can execute the following commands:
```
python3 test_mnist.py <path to MNIST weights file>
```

or
```
python3 test_chars74k.py <path to Chars74k weights file>
```

Examples:
```
python3 test_mnist.py weights/mnist/100.p
```

and
```
python3 test_chars74k.py weights/chars74k/50x50.p
```
