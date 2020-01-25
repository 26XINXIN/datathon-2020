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
        for i,a in enumerate(self.areas):
            sIn, sOut, iIn, iOut = 0,0,0,0
            for j,b in enumerate(self.areas):
                sOut += self.transition_matrix[i,j] * (1 - a.ratio) * a.N
                sIn += self.transition_matrix[j,i] * (1 - b.ratio) * b.N
                iOut += self.transition_matrix[i,j] * a.ratio * a.N
                iIn += self.transition_matrix[j,i] * b.ratio * b.N
            a.nextState(sIn,sOut,iIn,iOut)
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
        [87, 10, 3],
        [10, 80, 10],
        [10, 10, 80]
    ])
    initInfection = ["China", 5]
    beta = 0.5
    c = 0.25
    r = Runner(areaInfo,transition_matrix,initInfection,beta,c)
    while (input() != '.'):
        r.next()













