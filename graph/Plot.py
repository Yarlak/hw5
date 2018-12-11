# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:17:27 2018

@author: OH YEA
"""

from matplotlib import pyplot
import numpy as np
import csv

def ReadResults(whichOne, episodes, trials):
    theReturns = np.zeros((trials, episodes))
    with open(whichOne, 'rb') as csvfile:
        theReader = csv.reader(csvfile)
        readCount = 0
        for row in theReader:
            theReturns[readCount, :] = row
            readCount += 1
    
    return GetData(theReturns)

def GetData(theData):
    theMean = np.mean(theData, axis=0)
    theStdDev = np.std(theData, axis=0)
    return theMean, theStdDev

cpf_q_mean, cpf_q_std = ReadResults('cpf_final_q.csv', 100, 100)
cpf_s_mean, cpf_s_std = ReadResults('cpf_sarsa.csv', 100, 100)

gw_q_mean, gw_q_std = ReadResults('gridworld-q.csv', 100, 100)
gw_s_mean, gw_s_std = ReadResults('gridworld-sarsa.csv', 100, 100)

cpp_q_mean, cpp_q_std = ReadResults('cppol-q.csv', 100, 100)
cpp_s_mean, cpp_s_std = ReadResults('cppol-sarsa.csv', 100, 100)

cpf_t_q_mean, cpf_t_q_std = ReadResults('cpf_tentrials_q.csv', 100, 10)

xValues = list(range(100))

f1 = pyplot.figure()
f2 = pyplot.figure()
f3 = pyplot.figure()
f4 = pyplot.figure()
f5 = pyplot.figure()
f6 = pyplot.figure()
f7 = pyplot.figure()
f8 = pyplot.figure()
f9 = pyplot.figure()
f10 = pyplot.figure()
f11 = pyplot.figure()
f12 = pyplot.figure()

a1 = f1.add_subplot(111)
a1.plot(xValues, cpf_q_mean, 'k-')
a1.set_title("Cart Pole Q-Learning with Fourier")
a1.fill_between(xValues, cpf_q_mean-cpf_q_std, cpf_q_mean+cpf_q_std)
f1.show()

a2 = f2.add_subplot(111)
a2.plot(xValues, cpf_s_mean, 'k-')
a2.set_title("Cart Pole Sarsa with Fourier")
a2.fill_between(xValues, cpf_s_mean-cpf_s_std, cpf_s_mean+cpf_s_std)
f2.show()

a3 = f3.add_subplot(111)
a3.plot(xValues, cpf_s_mean, xValues, cpf_q_mean)
a3.set_title("Cart Pole Sarsa and Q-Learning with Fourier")
a3.legend(["Sarsa", "Q-Learning"])
f3.show()

a4 = f4.add_subplot(111)
a4.plot(xValues, cpp_q_mean, 'k-')
a4.set_title("Cart Pole Q-Learning with Polynomial")
a4.fill_between(xValues, cpp_q_mean-cpf_q_std, cpf_q_mean+cpf_q_std)
f4.show()

a5 = f5.add_subplot(111)
a5.plot(xValues, cpp_q_mean, 'k-')
a5.set_title("Cart Pole Q-Learning with Polynomial")
f5.show()

a6 = f6.add_subplot(111)
a6.plot(xValues, cpp_s_mean, 'k-')
a6.set_title("Cart Pole Sarsa with Polynomial")
a6.fill_between(xValues, cpp_s_mean-cpf_s_std, cpf_s_mean+cpf_s_std)
f6.show()

a7 = f7.add_subplot(111)
a7.plot(xValues, cpp_q_mean, 'k-')
a7.set_title("Cart Pole Sarsa with Polynomial")
f7.show()

a8 = f8.add_subplot(111)
a8.plot(xValues, gw_q_mean, 'k-')
a8.set_title("Gridworld with Q-Learning")
a8.fill_between(xValues, gw_q_mean - gw_q_std, gw_q_mean + gw_q_std)
f8.show()

a9 = f9.add_subplot(111)
a9.plot(xValues, gw_s_mean, 'k-')
a9.set_title("Gridworld with Sarsa")
a9.fill_between(xValues, gw_s_mean - gw_s_std, gw_s_mean + gw_s_std)
f9.show()

a10 = f10.add_subplot(111)
a10.plot(xValues, gw_s_mean, xValues, gw_q_mean)
a10.set_title("Gridworld with Sarsa and Q-Learning")
a10.legend(["Sarsa", "Q-Learning"])
f10.show()

a11 = f11.add_subplot(111)
a11.plot(xValues, cpf_t_q_mean, 'k-')
a11.set_title("Cart Pole Q-Learning with Fourier (ten trials)")
a11.fill_between(xValues, cpf_t_q_mean - cpf_t_q_std, cpf_t_q_mean + cpf_t_q_std)
f11.show()

a12 = f12.add_subplot(111)
a12.plot(xValues, cpf_t_q_mean, xValues, cpf_q_mean)
a12.set_title("Cart Pole Q-Learning with 3rd and 4th order Fourier")
a12.legend(["4th Order", "3rd Order"])
f12.show()

pyplot.show()