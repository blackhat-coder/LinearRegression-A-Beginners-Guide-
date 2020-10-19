import numpy as np
import pandas as pd


class LinearRegression:

    @staticmethod
    def cost_function(self, theta, target, feautures):
        self.theta = theta
        self.target = target
        self.feautures = feautures

        m = len(target)

        cost = np.square(np.dot(feautures, theta) - target).sum()

        return cost / (2 * m)
    
    @staticmethod
    def fit(self, X: "Feautures", Y: "Target"):

        model_fit = np.dot(np.linalg.pinv(np.dot(np.transpose(X) , X)) , np.dot(np.transpose(X) , Y))

        self.b = model_fit[0]

        self.w = model_fit[1:]

        return model_fit
    
    def predict(self, feautures):
        return np.dot(feautures, self.fit())

    def __repr__(self):
        print("Linear Model")
