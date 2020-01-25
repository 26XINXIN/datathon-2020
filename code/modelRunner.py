from epiModel import *
import numpy as np

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
        for country, popu in population.items():
            epi = SI(country, beta, c)
            epi.setInitial(popu - source.get(country, 0), source.get(country, 0), 0)
            self.areas.append(epi)
            
        self.transition_matrix = np.asarray(transition_matrix)
        self.timeStamp = 0

    def next(self):
        sIn = np.zeros((len(self.areas), ))
        iIn = np.zeros((len(self.areas), ))
        sOut = np.asarray([a.S for a in self.areas])
        iOut = np.asarray([a.I for a in self.areas])

        for i, a in enumerate(self.areas):
            sIn += a.S * self.transition_matrix[i,:] 
            iIn += a.I * self.transition_matrix[i,:]

        for a, si, so, ii, io in zip(self.areas, sIn, sOut, iIn, iOut):
            if self.param_est:
                a.nextState(si, so, ii, io, self.param_est.predict([self.timeStamp])[0])
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
    population = {"China": 100, "United States": 100, "Mars": 100}
    transition_matrix = np.asarray([
        [0.87, 0.1, 0.03],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    source = {"China": 5}
    beta = 0.5
    c = 0.25
    r = Runner(transition_matrix, population, source, beta=beta, c=c)
    
    for _ in range(100):
        r.next()

    hist = {}
    history = r.getHistory()
    for c, h in history.items():
        hist[c] = [hh["I"] for hh in h]
    
    import matplotlib.pyplot as plt

    plt.plot(hist['China'], label="cn")
    plt.plot(hist['United States'], label="us")
    plt.plot(hist['Mars'], label="ms")
    plt.legend()
    plt.show()
    











