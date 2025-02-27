import matplotlib.pyplot as plt
import numpy as np

def plot(model, X_test, y_test, classifierFunction = 'Logistic'):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = model.predict(xx.ravel().reshape(-1, 1))
    Z = np.array(Z).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_test[:, 0], y_test, c=y_test, edgecolors='k')
    plt.title(f"{classifierFunction} Regression Decision Boundary")
    plt.show()