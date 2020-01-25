from epiModel import *
import numpy as np
from joblib import dump, load
from TrainParameters import *

class Runner:
    # areaInfo: list of tuples, first being the country's name, second being the country's population
    # transition_matrix: m[i,j] being the number of people travelling from i to j
    # initInfection: [country_name , number of initially infected people]
    # beta: probability of infection per contact
    # c: probability of people meeting other people
    def __init__(self, transition_matrix, population, source, param_est=None, beta=0.2, c=0.25):
        self.areas = list()
        self.n_countries = len(population)
        self.param_est = param_est
        self.transition_matrix = transition_matrix
        for country, popu in population.items():
            epi = SI(country, beta, c)
            epi.setInitial(popu - source.get(country, 0), source.get(country, 0), 0)
            self.areas.append(epi)
            if country not in self.transition_matrix.index.values:
                zeros = np.zeros(self.transition_matrix.shape[0])
                self.transition_matrix[country] = zeros
                self.transition_matrix.loc[country] = np.zeros(self.transition_matrix.columns.shape)
                self.transition_matrix.loc[country, country] = 1.0
        for country in self.transition_matrix.index:
            if country not in population:
                self.transition_matrix = self.transition_matrix.drop(columns=country)
                self.transition_matrix = self.transition_matrix.drop(index=country)
        self.timeStamp = 0
        

    def next(self):
        sIn = np.zeros((len(self.areas), ))
        iIn = np.zeros((len(self.areas), ))
        sOut = np.asarray([a.S for a in self.areas])
        iOut = np.asarray([a.I for a in self.areas])

        for a in self.areas:
            sIn += a.S * self.transition_matrix.loc[a.name,:].values
            iIn += a.I * self.transition_matrix.loc[a.name,:].values

        for a, si, so, ii, io in zip(self.areas, sIn, sOut, iIn, iOut):
            if self.param_est:
                # print(a.name, self.timeStamp, a.S, a.I, a.dS, a.dI)
                beta_c = self.param_est.predict([[self.timeStamp, a.S, a.I, a.dS, a.dI]])[0]
                a.nextState(si, so, ii, io, min(np.exp(beta_c), 1))
            else:
                a.nextState(si, so, ii, io)
        self.timeStamp += 1

    def getState(self):
        stateMap = {}
        for area in self.areaMap:
            stateMap[area] = self.areaMap[area].history[-1]
        return stateMap

    def getHistory(self):
        return {area.name: area.history for area in self.areas}


if __name__ == '__main__':
    data = read_data('../data/sars.csv')
    population = read_population('../data/population.csv')
    population = {c: p for c, p in population if c != 'Not classified'}
    with open("../data/countries.txt") as f:
        countries = [line.strip('\n') for line in f]
    transition_matrix = pd.read_csv('../data/trans.csv', index_col=False, header=None, names=countries)
    transition_matrix = transition_matrix.set_index(pd.Index(countries))
    # print(transition_matrix)
    source = {"China": 50}
    beta = 0.5
    c = 0.25
    r = Runner(transition_matrix, population, source, 
               param_est=load('../model/randomForest500'), beta=beta, c=c)
    
    for i in range(10):
        print("step" + str(i))
        r.next()

    hist = {}
    history = r.getHistory()
    for c, h in history.items():
        hist[c] = [hh["I"] for hh in h]
    
    with open("../data/simulation1.csv", "w") as f:
        f.write("country,infection\n")
        for c, data in hist.items():
            f.write(c + ',' + ','.join([str(d) for d in data]) + '\n')

    import matplotlib.pyplot as plt

    for c, d in hist.items():
        plt.plot(d, label=c)
    plt.legend()
    plt.show()
    











