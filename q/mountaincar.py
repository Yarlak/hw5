# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:02:56 2018

@author: OH YEA
"""

import numpy as np

#******************************
def GetNextAction(x, v, theC, theW, epsilon):
    
    tX = x
    tV = v
    
    posActions = np.array([-1, 0, 1])
    tempQ = ExploreQ(tX, tV, theW, theC)
    
    aIndex = GetAction(tempQ, epsilon)
    
    chosenAction = posActions[aIndex]
    return chosenAction
    
    
def GetAction (tQ, theEp):
    epsilon = theEp
    tempA = tQ
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
def ExploreQ(x, v, theW, theC):
   
    theX = x
    theV = v
    
    tempQ = np.zeros(3)
    
    x1, v1 = GetNextState(theX, theV, -1)
    x2, v2 = GetNextState(theX, theV, 0)
    x3, v3 = GetNextState(theX, theV, 1)
    
    tempQ[0], phi1 = GetQ(x1, v1, theC, theW)
    tempQ[1], phi2 = GetQ(x2, v2, theC, theW)
    tempQ[2], phi2 = GetQ(x3, v3, theC, theW)
    
    return tempQ
    
#******************************
def UpdateWeights(w, tAlpha, tTDE, thePhi):
    theAlpha = tAlpha
    theTDE = tTDE
    tPhi = thePhi
    
    tempW = w
    tempW += theAlpha * theTDE * tPhi
    
    return tempW

def CalculateTDE(x1, v1, x2, v2, currentReward, theC, theW):
    
    q1, phi1 = GetQ(x1, v1, theC, theW)
    #q2, phi2 = GetQ(x2, v2, theC, theW)

    q2 = np.amax(ExploreQ(x2, v2, theW, theC))
    
    TDE = currentReward + q2 * 1 - q1
    
    return TDE, phi1

def GetQ(x, v, tC, tW):
    
    tX = x
    tV = v
    
    nState = NormalizeState(tX, tV)
    
    tempW = tW
    tempC = tC
    
    tPhi = np.zeros(tempC.shape[0])
    
    for u in range(tempC.shape[0]):
        tInside = np.dot(tempC[u,:], nState)
        tPhi[u] = np.cos(tInside)
    
    q = np.dot(tempW, tPhi)
    
    return q, tPhi
    
    
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


def RunEpisode(alpha, order, theC, theW, epsilon):
    x = -0.5
    v = 0
    c = theC
    w = theW
    endEpisode = False
    totalReward = 0
    
    tCount = 0
    while (endEpisode == False):
        a = GetNextAction(x, v, c, w, epsilon)
        x2, v2 = GetNextState(x, v, a)
        endEpisode, r = GetReward(x2)
        
        tempTDE, tempPhi = CalculateTDE(x, v, x2, v2, r, c, w)
            
        w = UpdateWeights(w, alpha, tempTDE, tempPhi)

        totalReward += r
        tCount += 1
        x = x2
        v = v2
                
    
    return totalReward, w
    
def RunSimulation(nEpisodes, alpha, order, epsilon):
    
    theReturns = []
    c, w = ResetC(order)
    
    for episode in range(nEpisodes):
        tempReturn, w = RunEpisode(alpha, order, c, w, epsilon)
        theReturns.append(tempReturn)
    return theReturns




        