"""Example code using my neural net on the 10 class, MNIST dataset."""
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as matplotlib
import Functions as F
import NeuralNet as NN
import math as math

# Data prep
mnist = input_data.read_data_sets("/tmp/data/")
X_train = mnist.train.images
p = np.random.permutation(len(X_train))
X_train = X_train.transpose()
y_train = np.matrix(mnist.train.labels.astype("int"))
y_train = y_train.transpose()
y = np.zeros([10, 55000])
for i in range(55000):
    y[y_train[i], i] = 1
y_train = y
X_test = mnist.test.images
X_test = X_test.transpose()
y_test = np.matrix(mnist.test.labels.astype("int"))
y_test = y_test.transpose()
y = np.zeros([10, 10000])
for i in range(10000):
    y[y_test[i], i] = 1
y_test = y

# Shuffle data
X_train = X_train.transpose()
p = np.random.permutation(len(X_train))
X_train = X_train[p]
X_train = X_train.transpose()
y_train = y_train.transpose()
y_train = y_train[p]
y_train = y_train.transpose()

# Neural Net
net = NN.NeuralNet(784, [300, 100], 10, learning_rate = 0.000000001, momentum = 0.8)

#Graph prep
num_epoch = 2
batch_size = 100
num_batch = math.ceil(X_train.shape[1] / batch_size)
error_rates = [-1 for i in range(num_epoch * num_batch)]
plt.ion()
graph = plt.plot(np.linspace(0, num_epoch, num_epoch * num_batch), error_rates)[0]
plt.ylim(0, 100)
plt.xlim(0, num_epoch)
plt.ylabel('PCC')
plt.xlabel('Epoch')
plt.title('0.000000001')

# Train and graph
for e in range(num_epoch):
    print("Epoch: ", e + 1, "/", num_epoch)
    for b in range(0, X_train.shape[1], batch_size):
        print(b)
        net.train(X_train[:, b:b+batch_size], y_train[:, b:b+batch_size])
    
        yhat = net.test(X_test)
        error = F.pcc(yhat, y_test)
        error_rates[e * num_batch + int(b / batch_size)] = error
        graph.set_ydata(error_rates)
        plt.draw()
        plt.pause(0.01)
        print("PCC: ", error)




