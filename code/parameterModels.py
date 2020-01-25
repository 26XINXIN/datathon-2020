# models for predicting parameters

from sklearn.linear_model import LinearRegression


class LinearModel:
    def __init__(self, features, normalize=True):
        self.lr = LinearRegression(normalize=normalize)
        self.features = features
    
    def train(self, X, y):
        self.lr.fit(X, y, verbose=True)
    
    def predict(self, x):
        return self.lr.predict(x)
    
    def getFetures(self):
        return self.features
    
    def score(self, X, y):
        return self.lr.score(X, y)