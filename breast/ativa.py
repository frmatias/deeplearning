# -*- coding: utf-8 -*-
import numpy as np

def stepFunction(soma):
    if(soma >=1):
        return 1
    return 0

def sigmoidFunction(soma):
    return 1/(1 + np.exp(-soma))

def tanFunction(soma):
    return (np.exp(soma) - np.exp(-soma))/(np.exp(soma) + np.exp(-soma))

def relu(soma):
    if(soma >= 0):
        return soma
    return 0
def linearFunction(soma):
    return soma

def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum()

valor = [9,2,1.3]
print(softmax(valor))
