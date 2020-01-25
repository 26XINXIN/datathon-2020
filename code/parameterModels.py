# models for predicting parameters

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class LinearModel:
    def __init__(self, features, normalize=True):
        self.lr = LinearRegression(normalize=normalize)
        self.features = features
    
    def train(self, X, y):
        self.lr.fit(X, y)
    
    def predict(self, x):
        return self.lr.predict(x)
    
    def getFetures(self):
        return self.features
    
    def score(self, X, y):
        return self.lr.score(X, y)

class RandomForest:
    def __init__(self, ntree=500):
        self.rg = RandomForestRegressor(n_estimators=500)
        