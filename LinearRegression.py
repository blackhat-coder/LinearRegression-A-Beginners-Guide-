import numpy as np
import pandas as pd


class LinearRegression:


    def cost_function(self, theta, target, feautures):
        m = len(target)

        cost = np.square(np.dot(feautures, theta) - target).sum()

        return cost / (2 * m)
