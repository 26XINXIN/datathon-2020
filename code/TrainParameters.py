import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from parameterModels import *
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy import signal

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
    dI = data[1:, :] - data[:-1, :]
    dI = np.concatenate((np.zeros((1, data.shape[1])), dI), axis=0)
    I = data
    S = np.zeros(data.shape)
    countries =  ['Canada', 'China', 'France', 'Germany', 'Hong Kong', 'Italy', 'Ireland', 
                  'Singapore', 'Spain', 'Switzerland', 'Taiwan', 'Thailand', 'United Kingdom', 
                  'United States', 'Vietnam', 'Australia', 'Brazil', 'Bulgaria', 'Macau', 'Indonesia', 
                  'Kuwait', 'Malaysia', 'Mongolia', 'New Zealand', 'Philippines', 'Poland', 'South Korea', 
                  'Romania', 'South Africa', 'Sweden', 'Colombia', 'Finland', 'India', 'Belgium', 'Japan', 
                  'Russia', 'Slovenia']
    for i in range(data.shape[0]):
        for j, c in enumerate(countries):
            if c == 'Hong Kong':
                S[i, j] = 6731000 - data[i, j]
            elif c == 'Taiwan':
                S[i, j] = 22605000 - data[i, j]
            elif c == 'Macau':
                S[i, j] = 460147 - data[i, j]
            elif c == 'South Korea':
                S[i, j] = 47890000 - data[i, j] 
            elif c == 'Russia':
                S[i, j] = 144600000 - data[i, j]
            else:
                S[i, j] = population[c] - data[i, j]
    dS = S[1:, :] - S[:-1, :]
    dS = np.concatenate((np.zeros((1, data.shape[1])), dS), axis=0)
    return S[:-1,:], I[:-1,:], dS[:-1,:], dI[:-1,:], - dS[1:,:] / (S[:-1,:] * I[:-1,:] + 0.001) * (S[:-1,:] + I[:-1,:]) # last dS and dI 

def plotFactor(X, y):
    lr = LinearRegression()
    lr.fit(X.reshape(-1, 1), y)
    x = np.linspace(np.min(X), np.max(X), 100)
    fig, ax = plt.subplots()
    ax.plot(x, lr.predict(x.reshape(-1, 1)), 'r-')
    ax.plot(X, y, 'b.')
    # ax.set_ylim(-100, 100)
    plt.show()

def datafilter(T, S, I, dS, dI, y):
    nt, ns, ni, nds, ndi, ny = list(), list(), list(), list(), list(), list()
    for t, s, i, ds, di, yy in zip(T, S, I, dS, dI, y):
        if i != 0 and yy >= 0 and di < 100:
            nt.append(t)
            ns.append(s)
            ni.append(i)
            nds.append(ds)
            ndi.append(di)
            ny.append(yy)
    return np.asarray(nt), np.asarray(ns), np.asarray(ni), np.asarray(nds), np.asarray(ndi), np.asarray(ny)

if __name__ == "__main__":
    data = read_data('../data/sars.csv')
    population = read_population('../data/population.csv')
    S, I, dS, dI, beta_c = extract_training_data(data, population)
    beta_c = signal.savgol_filter(beta_c, 5, 2, axis=0)
    t = np.repeat(np.arange(beta_c.shape[0]).reshape(-1, 1), beta_c.shape[1], axis=1).reshape(-1,)
    S = S.reshape(-1,)
    I = I.reshape(-1,)
    dS = dS.reshape(-1,)
    dI = dI.reshape(-1,)
    
    beta_c = beta_c.reshape(-1,)
    

    t, S, I, dS, dI, beta_c = datafilter(t, S, I, dS, dI, beta_c)
    
    beta_c = np.log(beta_c+0.001)
    
    # plt.plot(beta_c, '.')
    # plt.show()
    plotFactor(S, beta_c)
    
    X = np.asarray(list(zip(t, S, I, dS, dI)))
    y = beta_c
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # predictor = LinearModel(["S", "I", "dS", "dI"])
    # predictor.train(X_train, y_train)
    # print(predictor.score(X_test, y_test))
    # plt.plot(predictor.predict(X_test), y_test, '.')
    # print(y_test)
    rg = RandomForestRegressor(n_estimators=500)
    rg.fit(X_train, y_train)
    print(rg.score(X_test, y_test))
    plt.plot(rg.predict(X_test), y_test, '.')
    plt.show()
    