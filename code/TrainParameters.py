import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from parameterModels import *
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def read_data(path):
    data = list()
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            data.append([float(v) for v in line])
    return np.asarray(data)

def read_population(path):
    data = pd.read_csv(path)
    return data[["Country Name", "2003"]].values
    

def extract_training_data(data, population):
    population = {c[0]: c[1] for c in population}
    dS = data[1:, :] - data[:-1, :]
    dS = np.concatenate((np.zeros((1, data.shape[1])), dS), axis=0)
    S = data
    I = np.zeros(data.shape)
    countries =  ['Canada', 'China', 'France', 'Germany', 'Hong Kong', 'Italy', 'Ireland', 
                  'Singapore', 'Spain', 'Switzerland', 'Taiwan', 'Thailand', 'United Kingdom', 
                  'United States', 'Vietnam', 'Australia', 'Brazil', 'Bulgaria', 'Macau', 'Indonesia', 
                  'Kuwait', 'Malaysia', 'Mongolia', 'New Zealand', 'Philippines', 'Poland', 'South Korea', 
                  'Romania', 'South Africa', 'Sweden', 'Colombia', 'Finland', 'India', 'Belgium', 'Japan', 
                  'Russia', 'Slovenia']
    for i in range(data.shape[0]):
        for j, c in enumerate(countries):
            I[i, j] = population.get(c, 6731000) - data[i, j]
    dI = I[1:, :] - I[:-1, :]
    dI = np.concatenate((np.zeros((1, data.shape[1])), dI), axis=0)
    return S[:-1,:], I[:-1,:] / 1000000, dS[:-1,:], dI[:-1,:] / 1000000, - dS[1:,:] / (S[:-1,:] * I[:-1,:] / 1000000 + 0.001) * (S[:-1,:] + I[:-1,:] / 1000000) # last dS and dI 

def plotFactor(X, y):
    lr = LinearRegression()
    lr.fit(X.reshape(-1, 1), y)
    x = np.linspace(np.min(X), np.max(X), 100)
    fig, ax = plt.subplots()
    ax.plot(x, lr.predict(x.reshape(-1, 1)), 'r-')
    ax.plot(X, y, 'b.')
    # ax.set_ylim(-100, 100)
    plt.show()
    

if __name__ == "__main__":
    data = read_data('../data/sars.csv')
    population = read_population('../data/population.csv')
    I, S, dS, dI, beta_c = extract_training_data(data, population)
    print(S)
    S = S.reshape(-1,)
    I = I.reshape(-1,)
    dS = dS.reshape(-1,)
    dI = dI.reshape(-1,)
    beta_c = beta_c.reshape(-1,)
    print(S)
    
    plotFactor(S, beta_c)

    X = np.asarray(list(zip(S, I, dS, dI)))
    y = beta_c
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    predictor = LinearModel(["S", "I", "dS", "dI"])
    predictor.train(X_train, y_train)
    print(predictor.score(X_test, y_test))