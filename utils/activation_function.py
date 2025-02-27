import numpy as np 


class Activation_functions():

    def __init__(self, classificationFunctionChoice = 'sigmoid'):
        self.clasificationFunction = classificationFunctionChoice

    def activate(self, z):
        if self.clasificationFunction == 'sigmoid':
            return(self.sigmoid(z))
        if self.clasificationFunction == 'relu':
            return(self.ReLu(z))
               
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def ReLu(self, z):
        return (np.maximum(0, z))
    
        