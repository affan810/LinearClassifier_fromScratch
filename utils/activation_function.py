import numpy as np 


class Activation_functions():
    '''
    Initialize the object with str 'sigmoid' or relu
    '''

    def __init__(self, classificationFunctionChoice = 'sigmoid'):
        self.clasificationFunction = classificationFunctionChoice

    #activate method is added because Activation_functions(classificationFunctionChoice='sigmoid', z=linear_model) created an instance of the class instead of returning the activation value directly 
    def activate(self, z):
        if self.clasificationFunction == 'sigmoid':
            return(self.sigmoid(z))
        if self.clasificationFunction == 'relu':
            return(self.ReLu(z))
               
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def ReLu(self, z):
        return (np.maximum(0, z))
    
        