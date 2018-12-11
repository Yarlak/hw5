# -*- coding: utf-8 -*-
"""
Created on Sat Dec 08 23:01:40 2018

@author: OH YEA
"""

import csv
import mountaincar as mc
import numpy as np
from matplotlib import pyplot


#trials = 2
#nEpisodes = 100
#alpha = 0.01
#lamb = 0.5
#order = 2

trials = 100
nEpisodes = 100
alphas = [0.0025]
orders = [2]
epsilons = [0.9]
lamb = 0.15

for epsilon in epsilons:
    for alpha in alphas:
        for order in orders:
    
            theReturns = np.zeros((trials, nEpisodes))
            theReturns[0, :] = np.asarray(mc.RunSimulation(nEpisodes, alpha, order, epsilon, lamb))
            xValues = list(range(len(theReturns)))
            
            for x in range(trials-1):
                theReturns[x+1, :] = np.asarray(mc.RunSimulation(nEpisodes, alpha, order, epsilon, lamb))
                
            theFinal = np.mean(theReturns, axis=0)
            xValues = list(range(len(theFinal)))
            
            pyplot.plot(xValues, theFinal)
            pyplot.title(str(order) + "-" + str(alpha) + "-" + str(epsilon))
            pyplot.show()
            
            theName = "mountaincar-s-" + str(order) + "-" + str(alpha) + "-" + str(epsilon) + ".csv"
            with open(theName, 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for entry in range(trials):
                    writer.writerow(theReturns[entry, :])