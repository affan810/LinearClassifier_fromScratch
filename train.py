import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.generate_data import DataGenration 
from utils.activation_function import Activation_functions
from utils.plotting import plot

'''
Classification using function (choices 1: sigmoid , 2: Relu)
'''


'''
choose from Sigmoid or ReLu activation functions and update the classifierchoice variable
'''
classifierchoice = 'sigmoid'


# Logistic Regression Class
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate                 #learning rate for gradient decent
        self.epochs = epochs                               #number of iterations
        self.weights = None                                #initial weight
        self.bias = None                                   #initial bias 

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            model = np.dot(X, self.weights) + self.bias
            activation = Activation_functions(classificationFunctionChoice=classifierchoice)
            predictions = activation.activate(model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            #deletes the object activation to avoid extented memory usage
            del activation

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        activation = Activation_functions(classifierchoice)
        y_predicted = activation.activate(linear_model)    
        return [1 if i > 0.5 else 0 for i in y_predicted] #created boundery of >0.5 for classification

# Generate synthetic data
data1 = DataGenration(noise_level='low')
X, Y = data1.linearClassificationData()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.4f}')



plot(model, X_test, y_test, classifierFunction = 'sigmoid')
