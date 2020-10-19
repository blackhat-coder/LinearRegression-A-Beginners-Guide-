import numpy as np
import pandas as pd


class LinearRegression:
    def cost_function(self, theta, target, feautures):
        m = len(target)

        cost = np.square(np.dot(feautures, theta) - target).sum()

        return cost / (2 * m)
    
    def fit(self, X, Y):
        return np.dot(np.linalg.pinv(np.dot(np.transpose(X) , X)) , np.dot(np.transpose(X) , Y))
    
    def predict(self, feautures):
        return np.dot(feautures, self.fit())

