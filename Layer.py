import numpy as np

class NNLayer:
    g = None
    g_prime = None
    w = None
    bias = None
    x = None
    h = None
    p = None
    w_updates = None
    b_updates = None
    learning_rate = None
    momentum = None
    previous_w_update = None
    previous_b_update = None
    
    def __init__(self, num_inputs, num_nodes, function_pair, learning_rate, momentum):
        self.g = function_pair[0]
        self.g_prime = function_pair[1]
        self.w = np.matrix(np.random.rand(num_inputs, num_nodes))
        self.bias = np.matrix(np.zeros([num_nodes, 1]))
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.previous_w_update = np.matrix(np.zeros([num_inputs, num_nodes]))
        self.previous_b_update = np.matrix(np.zeros([num_nodes, 1]))
    
    def apply(self, x):
        self.x = np.matrix(x)
        self.p = np.matmul(self.w.transpose(), self.x) + self.bias
        self.h = self.g(self.p)
        return np.matrix(self.h)
    
    def batch_train(self, deriv, is_last = False):
        if is_last:
            intermed = np.matmul(self.g_prime(self.p), deriv)
        else:
            intermed = np.multiply(self.g_prime(self.p), deriv)
        m = np.ones(self.w.shape)
        m = np.multiply(m, intermed.transpose())
        m = np.multiply(self.x, m)
        self.w_updates.append(m)
        self.b_updates.append(intermed)
        return np.matmul(self.w, intermed)
    
    def batch_update(self):
        w_increment = np.multiply(self.learning_rate, sum(self.w_updates))/len(self.w_updates) + np.multiply(self.momentum, self.previous_w_update)
        self.w = self.w + w_increment
        self.previous_w_update = w_increment
        b_increment = np.multiply(self.learning_rate, sum(self.b_updates))/len(self.b_updates) + np.multiply(self.momentum, self.previous_b_update)
        self.bias = self.bias + b_increment
        self.previous_b_update = b_increment
        return
    
    def new_batch(self):
        self.w_updates = []
        self.b_updates = []
    
    def print_state(self):
        print("Weights" , self.w)
        print("Bias ", self.bias)
