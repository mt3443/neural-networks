## EECS 738 Project 3: Neural Networks by Matthew Taylor

### Overview

The purpose of this project was to create a neural network that can solve interesting problems. Two datasets were used to ensure the neural network was general and robust, specifically the MNIST and Chars74k datasets. A simple multilayer perceptron was implemented from scratch and trained on these datasets. After training, the neural network was able to achieve a testing accuracy of up to 96%.

### Approach

The MNIST dataset contains samples of handwritten digits 0 through 9. The neural network is simply meant to classify a given digit. Solving this problem is the neural network equivalent of "Hello World". The Chars74k dataset contains handwritten letters, capital and lowercase A through Z, and as with the MNIST dataset, the neural network classifies these letters. Admittedly, these datasets are very similar. However, there are some key differences that will become apparent later.

The neural network's inputs are the pixels of the grayscale images of the handwritten digits or letters. As you might have guessed, the outputs of this network are the possible classes. Ideally, only one of these output nodes will stand out from all others. This unique node represents the neural network's guess as to which class the given input belongs to. The neural network created in this project is incredibly dynamic and can be instantiated with any number of inputs, outputs, and hidden layers. 
