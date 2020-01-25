import matplotlib.pyplot as plt
import numpy as np
from epiModel import SI
import pandas as pd

def runSi(beta=0.19, c=0.1):
    si = SI(name='world', beta=beta, c=beta)
    si.setInitial(66594397472.0-167, 167, 0)
    for i in range(120):
        si.nextState(0, 0, 0, 0)
    return si.history

SIdata = runSi()
SIdata = [d['I'] for d in SIdata]


data = list()
k = 0
with open("../data/real_sars1.csv") as f:
    for line in f:
        if len(line.split(',')) <= 2:
            continue
        if k == 0:
            row = line.strip().split(',')
            k = len(row) - 1
            data.append([float(v) for v in row[1:]])
        else:
            row = line.strip().rsplit(',', k)
            data.append([float(v) for v in row[1:]])

total = list()
for j in range(len(data[0])):
    total.append(sum([data[i][j] for i in range(len(data))]))

true_data = np.loadtxt('../data/sars.csv', delimiter=',')
true_total = np.sum(true_data, axis=1)


plt.plot(true_total, label='true data')
plt.plot(total, label='learned SI')
plt.plot(SIdata, label='classical SI')
plt.legend()
plt.show()

def read_population(path):
    data = pd.read_csv(path)
    return data[["Country Name", "2003"]].values
population = read_population('../data/population.csv')
total_population = 0
for c, p in population:
    if c == 'Not classified': continue
    total_population += p
print(total_population)