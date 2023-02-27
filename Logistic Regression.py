## Logistic Regression from scratch
## Kashaf Ali 2023

import numpy as np
import pandas as pd


class Logistic_regression:
    # Class constructor
    def __init__(self):
        self.w = None  # logistic regression weights
        self.saved_w = []  # Since this is a small problem, we can save the weights
        #  at each iteration of gradient descent to build our
        #  learning curve
        self.saved_c = []
        self.saved_c_test = []
        # returns nothing
        pass

    def prepare_x(self, X):
        # apends a column of ones to the beginning of the input matrix X
        x = np.array(X)
        ones = np.ones(shape=(x.shape[0], 1))
        c = np.concatenate((ones, x), axis=1)
        # returns the X with a new feature of all ones (a column that is the new column 0)
        return c

    # Method for calculating the sigmoid function of w^T X for an input set of weights
    def sigmoid(self, X, w):
        z = np.dot(self.prepare_x(X), w)
        sig = 1 / (1 + np.exp(-z))  # z = w^T X + b
        # returns the value of the sigmoid
        return sig

    # Cost function for an input set of weights
    def cost(self, X, y, w):
        m = self.prepare_x(X).shape[0]
        y = np.array(y)
        cost = (
            np.matmul(-y.T, np.log(self.sigmoid(X, w)))
            - np.matmul((1 - y.T), np.log(1 - self.sigmoid(X, w)))
        ) / m
        # returns the average cross entropy cost
        return cost

    # Update the weights in an iteration of gradient descent
    def gradient_descent(self, X, y, w, lr):
        # returns a scalar of the magnitude of the Euclidean norm
        #  of the change in the weights during one gradient descent step
        m = self.prepare_x(X).shape[0]
        y = np.array(y)
        dw = (np.matmul(self.prepare_x(X).T, (self.sigmoid(X, self.w) - y))) / m
        # magnitude of the step gradient descent takes
        mag = lr * np.linalg.norm(dw)
        # update the weights
        self.w = self.w - lr * dw
        # returns the updated weights
        return self.w, mag

    # Fit the logistic regression model to the data through gradient descent
    def fit(self, X, y, w_init, lr, delta_thresh=1e-6, max_iter=5000, verbose=False):
        # Note the verbose flag enables you to print out the weights at each iteration
        #  (optional - but may help with one of the questions)
        self.w = w_init
        self.saved_w = []
        self.saved_c = []
        # for each iteration of gradient descent
        for i in range(max_iter):
            # update the weights
            self.w, mag = self.gradient_descent(X, y, self.w, lr)
            # save the weights
            self.saved_w.append(self.w.copy())
            # save the cost
            self.saved_c.append(self.cost(X, y, self.w))
            # save the cost for the test data
            if verbose:
                print(self.w)
            # if the magnitude of the change in the weights is less than the threshold
            #  then stop the gradient descent
            if mag < delta_thresh:
                break

    # Use the trained model to predict the confidence scores
    # (prob of positive class in this case)
    def predict_proba(self, X):
        # returns the confidence score for the each sample
        p = self.sigmoid(X, self.w)
        return p

    # Use the trained model to make binary predictions
    def predict(self, X, thresh=0.5):
        # returns a binary prediction for each sample
        predictions = self.predict_proba(X)
        predicted_class = np.where(predictions >= thresh, 1, 0)
        return predicted_class

    # Stores the learning curves from saved weights from gradient descent
    def learning_curve(self, X, y):
        # returns the value of the cost function from each step in gradient descent
        #  from the last model fitting process
        self.saved_c_test = []
        # for each weight in the saved weights
        for i in range(len(self.saved_w)):
            self.saved_c_test.append(self.cost(X, y, self.saved_w[i]))
        return self.saved_c_test
