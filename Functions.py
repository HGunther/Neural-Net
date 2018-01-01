import numpy as np
import math as math

"""Mean Squared Error"""
def mse(x, y):
    error = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            error += math.pow(x[i,j] - y[i,j], 2)
    return error / float(x.size)

"""Percent Correctly Classified"""
def pcc(yhat, y):
    errors = 0
    for i in range(yhat.shape[1]):
        if np.array(np.round(yhat[:, i])).argmax() != np.array(y[:, i]).argmax():
            errors += 1
    return (1 - errors/float(yhat.shape[1])) * 100

"""Rectified Linear Unit"""
def ReLU(p):
    answer = np.matrix(np.zeros(p.shape))
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            answer[i,j] = max(0, p[i,j])
    return answer

def ReLU_Deriv(p):
    answer = np.matrix(np.zeros([p.shape[0], p.shape[0]]))
    for i in range(p.shape[0]):
        answer[i,i] = 1 if p[i,0] > 0 else 0
    return answer

def ReLU_Deriv_Diag(p):
    """For the special case in this program where I want the returned values as a vector rather than a diagonal matrix to optimize the calculations."""
    answer = np.matrix(np.zeros([p.shape[0], 1]))
    for i in range(p.shape[0]):
        answer[i,0] = 1 if p[i,0] > 0 else 0
    return answer

ReLU_tuple = (ReLU, ReLU_Deriv_Diag)

"""Softmax"""
def Softmax(p):
    answer = np.matrix(np.zeros(p.shape))
    maximums = p.max(0)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            answer[i,j] = math.pow(math.e, p[i,j] - maximums[0,j])
    denom = answer.sum(0)
    for i in range(answer.shape[0]):
        for j in range(answer.shape[1]):
            answer[i,j] = answer[i,j] / denom[0,j]
    return answer

def Softmax_Deriv(p):
    sm = Softmax(p)
    answer = np.matrix(np.zeros([p.shape[0], p.shape[0]]))
    for i in range(p.shape[0]):
        for j in range(p.shape[0]):
            answer[i,j] = 1 - sm[i,0] if i == j else - sm[i,0]
    return answer

Softmax_tuple = (Softmax, Softmax_Deriv)

"""Cross Entropy"""
def Cross_Entropy(y, yhat):
    total = 0
    for i in range(y.shape(0)):
        total += y[i] * math.ln(yhat[i])
    return - total

def Cross_Entropy_Deriv(y):
    return - y

Cross_Entropy_tuple = (Cross_Entropy, Cross_Entropy_Deriv)

"""Sigmoid"""
def Sigmoid(p):
    answer = np.matrix(np.zeros(p.shape))
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            answer[i,j] = (math.pow(math.e, p[i,j])) / (math.pow(math.e, p[i,j]) + 1)
    return answer

def Sigmoid_Deriv(p):
    return 1 - Sigmoid(p)

Sigmoid_tuple = (Sigmoid, Sigmoid_Deriv)
