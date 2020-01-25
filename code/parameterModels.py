# models for predicting parameters

import sklearn


class LinearModel:
    def __init__(self, features, normalize=True):
        self.lr = sklearn.linear_model.LinearRegression(normalize=normalize)
        self.features = features
    
    def train(self, X, y):
        self.lr.fit(X, y)
    
    def predict(self, x):
        return self.lr.predict(x)
    
    def getFetures(self):
        return self.features