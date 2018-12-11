# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:49:51 2018

@author: OH YEA
"""
import csv
import gridworld as gw
import numpy as np
from matplotlib import pyplot

gw.SetBoard()
trials = 100
nEpisodes = 100
alpha = 0.2
lamb = 0.5

theReturns = np.zeros((trials, nEpisodes))
theReturns[0, :] = np.asarray(gw.RunSimulation(nEpisodes, alpha, lamb))
xValues = list(range(len(theReturns)))

for x in range(trials-1):
    theReturns[x+1, :] = np.asarray(gw.RunSimulation(nEpisodes, alpha, lamb))
    
theFinal = np.mean(theReturns, axis=0)
xValues = list(range(len(theFinal)))

pyplot.plot(xValues, theFinal)
pyplot.show()

with open('gridworld-sarsa.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for entry in range(trials):
        writer.writerow(theReturns[entry, :])