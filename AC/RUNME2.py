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

trials = 5
nEpisodes = 100
alphas = [0.0001, 0.001]
orders = [5, 6, 7]
epsilons = [0.9]
lambs = [0.05, 0.1]
betas = [0.0001, 0.001]

for epsilon in epsilons:
    for alpha in alphas:
        for order in orders:
            for beta in betas:
                for lamb in lambs:
                    print("NEW SET")
                    try:
                        theReturns = np.zeros((trials, nEpisodes))
                        theReturns[0, :] = np.asarray(mc.RunSimulation(nEpisodes, alpha, order, epsilon, lamb, beta))
                        xValues = list(range(len(theReturns)))
                        
                        for x in range(trials-1):
                            theReturns[x+1, :] = np.asarray(mc.RunSimulation(nEpisodes, alpha, order, epsilon, lamb, beta))
                            
                        theFinal = np.mean(theReturns, axis=0)
                        xValues = list(range(len(theFinal)))
                        
                        pyplot.plot(xValues, theFinal)
                        pyplot.title(str(order) + "-" + str(alpha) + "-" + str(epsilon) + "-" + str(lamb) + "-" + str(beta))
                        pyplot.show()
                        
                        theName = "mountaincar-ac-" + str(order) + "-" + str(alpha) + "-" + str(epsilon) + "-" + str(lamb) + "-" + str(beta) + ".csv"
                        with open(theName, 'wb') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',')
                            for entry in range(trials):
                                writer.writerow(theReturns[entry, :])
                    except:
                        print("bad hyperparamters")