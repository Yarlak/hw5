# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:02:56 2018

@author: OH YEA
"""

import numpy as np
import math

#******************************
def UpdateEVal(theE, theGamma, theLamb, thePhi):
    tE = theE
    gamma = theGamma
    lamb = theLamb
    
    tE = gamma * lamb * tE + thePhi
    
    return tE

#******************************
def GetNextAction(x, v, theC, theW, epsilon):
    
    tX = x
    tV = v
    
    posActions = np.array([-1, 0, 1])
    tempVal = ExploreVal(tX, tV, theW, theC)
    
    aIndex = GetAction(tempVal, epsilon)
    
    chosenAction = posActions[aIndex]
    return chosenAction
    
    
def GetAction (tVal, theEp):
    epsilon = theEp
    tempA = tVal
    if (CheckEntriesEqual(tempA)):
        tempChoice = np.array([0, 1, 2])
        theThing = np.random.choice(tempChoice)
        return theThing
    else: 
        theMax = tempA[np.argmax(tempA)]
        theIndices = GetIndices(tempA, theMax, EpsilonGreedy(epsilon))
        return np.random.choice(theIndices)
 

def GetIndices(theArray, maxValue, whichOne):
    maxIndices = []
    otherIndices = []
    iCounter = 0
    for x in theArray:
        if (x == maxValue):
            maxIndices.append(iCounter)
        else:
            otherIndices.append(iCounter)
        iCounter += 1
    
    if (whichOne == 0):
        return maxIndices
    else:
        return otherIndices

def EpsilonGreedy(theEp):
    dice = np.random.rand()
    epsilon = theEp
    if (dice < epsilon):
        return 0
    else:
        return 1

def CheckEntriesEqual(tArray):
    checkArray = tArray
    for x in range(len(checkArray)-1):
        if (checkArray[x+1] != checkArray[x]):
            return False
    return True
#******************************
def ExploreVal(x, v, theW, theC):
   
    theX = x
    theV = v
    
    tempQ = np.zeros(3)
    
    x1, v1 = GetNextState(theX, theV, -1)
    x2, v2 = GetNextState(theX, theV, 0)
    x3, v3 = GetNextState(theX, theV, 1)
    
    tempQ[0], phi1 = GetVal(x1, v1, theC, theW)
    tempQ[1], phi2 = GetVal(x2, v2, theC, theW)
    tempQ[2], phi2 = GetVal(x3, v3, theC, theW)
    
    return tempQ
    
#******************************
def UpdateWeights(w, tAlpha, tTDE, thePhi, theE):
    theAlpha = tAlpha
    theTDE = tTDE
    tPhi = thePhi
    eT = theE
    
    tempW = w
    tempW += theAlpha * theTDE * eT
    
    return tempW

def CalculateTDE(x1, v1, x2, v2, currentReward, theC, theW):
    
    val1, phi1 = GetVal(x1, v1, theC, theW)
    val2, phi2 = GetVal(x2, v2, theC, theW)
    
    TDE = currentReward + val2 * 1 - val1
    
    return TDE, phi1

def GetVal(x, v, tC, tW):
    
    tempW = tW
    tPhi = GetPhi(x, v, tC)
    
    val = np.dot(tempW, tPhi)
    
    return val, tPhi

def GetPhi(x, v, tC):
    
    tX = x
    tV = v
    
    nState = NormalizeState(tX, tV)
    
    tempC = tC
    
    tPhi = np.zeros(tempC.shape[0])
    
    for u in range(tempC.shape[0]):
        tInside = np.dot(tempC[u,:], nState)
        tPhi[u] = np.cos(tInside)
        
    return tPhi
    
    
def NormalizeState(x, v):
        
    minX = -1.2
    maxX = 0.5
    minV = -0.07
    maxV = 0.07
    
    theX = x
    theV = v
    
    theX = (theX - minX)/(maxX - minX)
    theV = (theV - minV)/(maxV - minV)
    
    stateVect = np.array([theX, theV])
    
    return stateVect





#*******************************
def ResetC(order):
    
    tempOrder = order + 1
    c = np.zeros((tempOrder**2,2), dtype=int)
    counter = 0
    
    for i in range(tempOrder):
        for j in range(tempOrder):
           c[counter, :] = np.array([i, j])
           counter += 1
    c = c * np.pi
    
    w = np.zeros(c.shape[0])
    
    return c, w
    

def GetNextState (x, v, a):
    tempX = x
    tempV = v
    tempA = a
    vNext = tempV + 0.001 * tempA - 0.0025 * np.cos(3*tempX)
    xNext = tempX + vNext
    
    if (vNext > 0.07):
        vNext = 0.07
    elif (vNext < -0.07):
        vNext = -0.07
    
    if (xNext > 0.5):
        xNext = 0.5
        vNext = 0
    elif (xNext < -1.2):
        xNext = -1.2
        vNext = 0
    
    return xNext, vNext

    
def GetReward(x):
    tempX = x
    endIt = False
    tempReward = -1
    if (tempX == 0.5):
        endIt = True
        tempReward = 0
    
    return endIt, tempReward


def RunEpisode(alpha, order, theC, theW, epsilon, theLamb, beta, theTheta):
    x = -0.5
    v = 0
    c = theC
    w = theW
    lamb = theLamb
    endEpisode = False
    totalReward = 0
    eVal = np.zeros(w.size)
    eTheta = np.zeros(w.size * 3)
    theta = theTheta
    tCount = 0
    while (endEpisode == False):
        newPhi = GetPhi(x, v, c)
        a, theProbs = PolicyTime(theta, newPhi)
        #a = GetNextAction(x, v, c, w, epsilon)
        x2, v2 = GetNextState(x, v, a)
        endEpisode, r = GetReward(x2)
        
        tempTDE, tempPhi = CalculateTDE(x, v, x2, v2, r, c, w)
        eVal = UpdateEVal(eVal, 1, theLamb, tempPhi)
        w = UpdateWeights(w, alpha, tempTDE, tempPhi, eVal)
        
        tempLogPol = GetLogPol(theProbs, tempPhi, 1)
        eTheta = UpdateEVal(eTheta, 1, theLamb, tempLogPol)
        theta += beta * tempTDE * eTheta
        
        totalReward += r
        tCount += 1
        
        if (tCount > 15000):
            raise ValueError('Bad time')
        x = x2
        v = v2
                
    return totalReward, w

def PolicyTime(theta, phi):
    
    tempProbs = []
    theExps = GetSoftmaxItems(theta, phi)
    tempSum = np.sum(theExps)
    

    for theExp in theExps:
        
        if (math.isnan(tempSum)):
            raise ValueError('Hello')
        tempProbs.append(theExp/tempSum)
        
        
    
    dice = np.random.rand()
        
    chosenAction = 0
    
    if (dice < tempProbs[0]):
        chosenAction = 0
    elif (dice < tempProbs[0] + tempProbs[1]):
        chosenAction = 1
    else:
        chosenAction = 2
        
    return chosenAction, tempProbs
    

def GetLogPol(theProbs, thePhi, theA):
    
    a = theA
    phiSize = thePhi.size
    tempLogPol = np.zeros(3 * phiSize)
    phi = thePhi
    probs = theProbs
    
    
    for i in range(3):
        tempI =  i * phiSize
        if (i == a):
            tempLogPol[tempI:tempI + phiSize] = (1 - probs[i]) * phi
        else:
            tempLogPol[tempI:tempI + phiSize] = probs[i] * phi
    
    return tempLogPol
    

def GetSoftmaxItems(theTheta, thePhi):
    
    theta = theTheta
    phi = thePhi
    
    calcs = []
    theSize = theta.size/3
    
    for i in range(3):
        tempI = i * theSize
        tInside = np.dot(theta[tempI:tempI + theSize], thePhi)
        theExp = np.exp(tInside)
        calcs.append(theExp)    
    
    return calcs
    
def RunSimulation(nEpisodes, alpha, order, epsilon, theLamb, beta):
    
    theReturns = []
    c, w = ResetC(order)
    theta = np.zeros(w.size * 3)
    
    for episode in range(nEpisodes):
        tempReturn, w = RunEpisode(alpha, order, c, w, epsilon, theLamb, beta, theta)
        theReturns.append(tempReturn)
        print("DONE")
    return theReturns




        