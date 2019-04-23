import pickle
from mnist import MNIST
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

mndata = MNIST('data/mnist/')
mndata.gz = True
test_input, test_output = mndata.load_testing()

incorrect = pickle.load(open(sys.argv[1], 'rb'))

while True:
    index = int(input('>> '))
    print(test_output[incorrect[index]])
    plt.imshow(np.array(test_input[incorrect[index]]).reshape(28,28), cmap='gray')
    plt.show()