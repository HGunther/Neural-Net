import Functions as F
import Layer as Layer
import numpy as np

class NeuralNet:
    num_inputs = 0
    layer_sizes = []
    layers = []
    
    def __init__(self, num_inputs, hidden_layer_sizes, num_outputs, learning_rate = 1, hidden_function = F.ReLU_tuple, output_function = F.Softmax_tuple, momentum = 0):
        self.num_inputs = num_inputs
        self.layer_sizes = [num_inputs] + hidden_layer_sizes + [num_outputs]
        self.layers = []
        for i in range(1, len(self.layer_sizes) - 1):
            self.layers.append( Layer.NNLayer( \
                                              num_inputs = self.layer_sizes[i - 1], \
                                              num_nodes = self.layer_sizes[i], \
                                              function_pair = hidden_function, \
                                              learning_rate = learning_rate, \
                                              momentum = momentum))
        self.layers.append( Layer.NNLayer( \
                                          num_inputs = self.layer_sizes[-2], \
                                          num_nodes = self.layer_sizes[-1], \
                                          function_pair = output_function, \
                                          learning_rate = learning_rate, 
                                          momentum = momentum))
    
    def train(self, x, y):
        # Prepare each layer for batch learning
        for l in self.layers:
            l.new_batch()
        # Enter each data pair
        for i in range(x.shape[1]):
            yhat = self.test(np.matrix(x[:, i].squeeze()).transpose())
            deriv = - F.Cross_Entropy_Deriv(np.matrix(y[:, i].squeeze()).transpose())
            deriv = self.layers[-1].batch_train(deriv, is_last = True)
            for l in self.layers[-2::-1]:
                deriv = l.batch_train(deriv)
        # Perform the updates
        for l in self.layers[::-1]:
            l.batch_update()
    
    def test(self, x):
        h = x
        for l in self.layers:
            h = l.apply(h)
        return h
    
    def print_state(self):
        """Used for debuging"""
        for i in range(len(self.layers)):
            print("Layer ", i)
            self.layers[i].print_state()