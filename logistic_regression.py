import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.e**-z)

def linear_regression(X, thetas):
    return X @ thetas

def logistic_regression(X, thetas):
    return sigmoid(linear_regression(X, thetas))

def cross_entropy(h, y):
    h = np.clip(h, 0.000000001, 0.99999999)
    return np.mean(- y * np.log(h) - (1 - y) * np.log(1 - h) )

def mse(h, y):
    return np.mean((h - y)**2)

def logistic_regression_derivative(h, y, X):
    m = len(X)
    return (1/m) * ((h - y) @ X)

def gradient_descent_logistic_regression(X, y, initial_thetas, alpha, iterations):
    trained_thetas = initial_thetas.copy()

    error_history = []

    for i in range(iterations):
        h = logistic_regression(X, trained_thetas)
        error = cross_entropy(h, y)
        error_history.append(error)

        trained_thetas -= alpha * logistic_regression_derivative(h, y, X)

    return trained_thetas, error_history

def gradient_descent_linear_regression(X, y, initial_thetas, alpha, iterations):
    trained_thetas = initial_thetas.copy()

    error_history = []

    for i in range(iterations):
        h = linear_regression(X, trained_thetas)
        error = mse(h, y)
        error_history.append(error)

        trained_thetas -= alpha * logistic_regression_derivative(h, y, X)

    return trained_thetas, error_history

def visualize_error_history(error_history):
  fig, ax = plt.subplots(figsize=(14,7))

  ax.plot(error_history, label="Cross Entropy")

  upper_ylim = np.quantile(error_history, 0.99)
  ax.set_ylim(-0.001, upper_ylim)

  ax.set_xlabel("iterations")
  ax.set_ylabel("Cross Entropy")
  ax.set_title("Learning Curve")
  fig.legend()

def accuracy(h, y):
    return np.mean(h.round() == y)
