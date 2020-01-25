from epiModel import *
import numpy as np

class Runner:
    # areaInfo: list of tuples, first being the country's name, second being the country's population
    # transition_matrix: m[i,j] being the number of people travelling from i to j
    # initInfection: [country_name , number of initially infected people]
    # beta: probability of infection per contact
    # c: probability of people meeting other people
    def __init__(self, areaInfo, transition_matrix, initInfection, beta, c):
        self.areaMap = {}
        self.areas = []
        for areaName, areaPopulation in areaInfo:
            areaOb = SI(beta, c)
            infected = 0 if initInfection[0] != areaName else initInfection[1]
            areaOb.setInitial(areaPopulation - infected, infected, 0)
            self.areaMap[areaName] = areaOb
            self.areas.append(areaOb)
        self.transition_matrix = transition_matrix
        self.timeStamp = 0

    def next(self):
        sOut = np.zeros((len(self.areas), ))
        iOut = np.zeros((len(self.areas), ))
        sIn = np.zeros((len(self.areas), ))
        iIn = np.zeros((len(self.areas), ))
        
        
        for i, a in enumerate(self.areas):
            sOut[i] = np.sum(self.transition_matrix[i,:]) * a.S
            iOut[i] = np.sum(self.transition_matrix[i,:]) * a.I
            sIn += a.S * self.transition_matrix[i,:]
            iIn += a.I * self.transition_matrix[i,:]

        for a, si, so, ii, io in zip(self.areas, sIn, sOut, iIn, iOut):
            a.nextState(si, so, ii, io)
        self.timeStamp += 1

    def getState(self):
        stateMap = {}
        for area in self.areaMap:
            stateMap[area] = self.areaMap[area].history[-1]
        return stateMap

    def getHistory(self):
        historyList = []
        for t in range(self.timeStamp):
            stateMap = {}
            for area in self.areaMap:
                stateMap[area] = self.areaMap[area].history[t]
            historyList.append(stateMap)
        return historyList


if __name__ == '__main__':
    areaInfo = [("China", 100), ("United States", 100), ("Mars", 100)]
    transition_matrix = np.asarray([
        [0.87, 0.1, 0.03],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    initInfection = ["China", 5]
    beta = 0.5
    c = 0.25
    r = Runner(areaInfo,transition_matrix,initInfection,beta,c)
    hist = []
    for _ in range(100):
        r.next()
    history = r.getHistory()
    for h in history:
        hist.append(
            {
                "China": h["China"]["I"],
                "United States": h["United States"]["I"],
                "Mars": h["Mars"]["I"],
            })
    import matplotlib.pyplot as plt
    cn = [h['China'] for h in hist]
    us = [h['United States'] for h in hist]
    ms = [h['Mars'] for h in hist]
    plt.plot(cn, label="cn")
    plt.plot(us, label="us")
    plt.plot(ms, label="ms")
    plt.legend()
    plt.show()
    











