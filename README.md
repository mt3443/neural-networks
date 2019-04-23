## EECS 738 Project 3: Neural Networks by Matthew Taylor

### Overview

The purpose of this project was to create neural networks that can solve interesting problems. Two datasets were used to ensure the neural network was general and robust, specifically the MNIST and Chars74k datasets. A simple multilayer perceptron was implemented from scratch and trained on these datasets. After training, the neural network was able to achieve a testing accuracy of up to 96%.

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

Training the neural network pickles the final weights and prints the path to the file the weights are stored in. Take note of the file path that is printed after training completes, as it is required for testing. To test the neural networks, you can execute the following commands:
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

### Results

The MNIST and Chars74k testing examples shown above have accuracies of 96.3% and 37.5%, respectively. These were the highest values I could achieve for each dataset. The test scripts record samples that are incorrectly classified by the neural network so they can be analyzed. Here are some incorrectly classified samples from the MNIST dataset:

![](https://i.imgur.com/K8GulNj.png) ![](https://i.imgur.com/Ludensr.png) ![](https://i.imgur.com/AbXTJxj.png)

The images above are meant to be 2, 8, and 1. But clearly, they look nothing like they're supposed to. Many test samples in the MNIST dataset share this flaw. After seeing these samples, it's understandable that the neural network could not classify 100% of the images correctly.

The Chars74k dataset had much less impressive results. Although the idea behind this dataset is practically identical to MNIST, there are some key differences between the two. The MNIST dataset contains 60,000 training samples and 10,000 testing samples, while the Chars74k dataset only had about 2,000 training samples and 500 testing samples. Perhaps the relatively small number of training samples couldn't provide the neural network a useful source upon which to develop an aptitude for classifying the testing samples. Another factor worth mentioning is that Chars74k samples were not processed as the MNIST samples had been. The MNIST sample digits were scaled and centered so each would have a similar size and location in the frame. This was not done in the Chars74k dataset, which may have led to the results we saw.

### Optimizations, Improvements, and Future Work

Given more time, I would have trained more models with a wider variety of hidden layers, nodes in each hidden layer, batch sizes, and number of epochs. I essentially chose network dimensions randomly to see which would provide the highest testing accuracy. Developing a systematic way to determine the best neural network dimensions by recording and plotting test and training accuracy would have taken an extremely long time (as each model takes about 10-30 minutes to train, depending on the dimensions), but would have offered the best results.

Next, I would like to have implemented a convolutional neural network alongside this multilayer perceptron, in order to compare the two. Since CNNs famously provide state-of-the-art performance in image processing applications, I'm curious to see how high the test accuracy could have gone with these two datasets.

Lastly, I also wish I added one additional class to the output of each neural network. This output would serve as a sort of "none of the above" class. For instance, if the neural network trained on the MNIST dataset were given an image of an X, something it's never seen before, it wouldn't classify it as any of the 10 digits it knows. However, this may have required restructuring or reclassifying input data to allow the neural network to be capable of such a feat. This is one of many steps that can be taken to create a more robust and dynamic neural network.
