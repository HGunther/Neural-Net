"""Example code using my neural network on the two class problem."""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as matplotlib
import Functions as F
import NeuralNet as NN
import math as math

# Read data
ifile = open("Example_Two_Class_Data.dat", 'r')
data = []
for line in ifile.readlines():
    split = line.strip().split()
    data.append([float(split[0]), float(split[1]), 0,1])
    data.append([float(split[2]), float(split[3]), 1,0])
ifile.close()

# Shuffle data
data = np.matrix(data)
np.random.shuffle(data)
data = data.transpose()

# Partition data into test and train
test = data[:, :int(data.shape[1]*0.2)]
train = data[:, int(data.shape[1]*0.2):]

# Graph prep
num_epoch = 150
evaluation = F.pcc
plt.ion()
plt.xlim(0, num_epoch)
plt.xlabel('Epoch')
if evaluation == F.pcc:
    plt.yscale('linear')
    plt.ylim(0, 100)
    plt.ylabel('PCC')
elif evaluation == F.mse:
    plt.yscale('log')
    plt.ylim(1e-3, 1)
    plt.ylabel('MSE')
error_rates = [-1 for i in range(num_epoch)]
graph = plt.plot(range(0,num_epoch), error_rates)[0]

# Create neural net
net = NN.NeuralNet(2, [5, 10], 2, learning_rate = 0.01, momentum = 0.8)

# Train and graph
for e in range(num_epoch):
    print("Epoch: ", e+1, "/", num_epoch)
    net.train(train[0:2, :], train[2:4, :])

    yhat = net.test(test[0:2, :])
    error = evaluation(yhat, test[2:4, :])
    error_rates[e] = error
    graph.set_ydata(error_rates)
    plt.draw()
    plt.pause(0.01)
    print("PCC: ", error)
