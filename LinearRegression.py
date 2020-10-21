import numpy as np
import pandas as pd


class Linear_Regression:
    def cost_function(self, theta, target, feautures):
        # computes vectorized cost function
        self._theta = theta
        self._target = target
        self._feautures = feautures

        m = len(target)

        cost = np.square(np.dot(feautures, theta) - target).sum()

        return cost / (2 * m)
    
    def fit(self, X: 'feautures', Y:'target'):
        # computes the model parameters using normal equation method
        
        self.model_fit = np.dot(np.linalg.pinv(np.dot(np.transpose(X) , X)) , np.dot(np.transpose(X) , Y))

        self.b = self.model_fit[0]

        self.w = self.model_fit[1:]

        return self.model_fit
    
    def predict(self, feautures):
        return np.dot(feautures, self.model_fit)

    def __repr__(self):
        print("Linear Modell")
