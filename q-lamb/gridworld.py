# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 21:08:20 2018

@author: OH YEA
"""
import numpy as np

#***********************
def UpdateET(theE, theGamma, theLamb):
    tE = theE
    gamma = theGamma
    lamb = theLamb
    
    tE = gamma * lamb * tE
    
    return tE

#***********************
def GetAction (tQ, tState):
    otherQ = tQ
    tempA = otherQ[tState - 1, :]
    if (CheckEntriesEqual(tempA)):
        tempChoice = np.array([0, 1, 2, 3])
        theThing = np.random.choice(tempChoice)
        return theThing
    else: 
        theMax = tempA[np.argmax(tempA)]
        theIndices = GetIndices(tempA, theMax, EpsilonGreedy())
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

def EpsilonGreedy():
    dice = np.random.rand()
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
#***********************
def GetReward(stateIndex, t, tGamma):
    reward = 0
    state = board[stateIndex[0], stateIndex[1]]
    gamma = tGamma
    
    if (state == 21):
        reward = -10 * gamma**t
    elif (state == 23):
        reward = 10 * gamma**t
    
    return reward

def GetNextState(action, stateIndex, t, tGamma):
    currentAction = action
    gamma = tGamma
    dice = np.random.random()
    tempStateIndex = np.array([stateIndex[0], stateIndex[1]])
    stays = False
    reward = 0
    
    if (dice < 0.8):
        currentAction = currentAction
    elif (dice < 0.85):
        if (currentAction + 1 >=4):
            currentAction = 0
        else:
            currentAction += 1
    elif (dice < 0.9):
        if (currentAction - 1 < 0):
            currentAction = 3
    else:
        stays = True
    tempStateIndex += actions[currentAction, :]

    if (tempStateIndex[0] < 0 or tempStateIndex[0] > 4 or tempStateIndex[1] < 0 or tempStateIndex[1] > 4 or board[tempStateIndex[0], tempStateIndex[1]] == -1 or stays):
        tempStateIndex = np.array([stateIndex[0], stateIndex[1]])
    else:
        reward = GetReward(tempStateIndex, t, gamma)
    return tempStateIndex, reward


def SetBoard():
    counter = 1
    for y in range(5):
        for x in range(5):
            if ([y,x] == [2, 2] or [y, x] == [3, 2]):
                board[y, x] = -1
            else:
                board[y, x] = counter
                counter += 1

def UpdateQ(tempQ, tempReward, s1, a1, s2, a2, alpha, tGamma, theE):
    newQ = tempQ
    TDE = GetTDE(tempQ, tempReward, s1 - 1, a1, s2 - 1, a2, tGamma)
    for state in range(23):
        newQ[state, a1] += TDE * alpha * theE[state]
    return newQ

def GetStateIndex(indices):
    return board[indices[0], indices[1]]
    
def GetTDE(tempQ, tempReward, s1, a1, s2, a2, tGamma):
    gamma = tGamma
    TDE = tempReward + gamma*np.amax(tempQ[s2, :]) - tempQ[s1, a1]
    return TDE

def RunSimulation(nEpisodes, alpha, theLamb):
    lamb = theLamb
    Q = np.zeros((23, 4)) + 2
    theReturns = []
    gamma = 0.9
    for i in range(nEpisodes):
        eT = np.zeros(23)
        theState = np.array([0, 0])
        tempReturn = 0
        t = 0
        tempAction = int(GetAction(Q, GetStateIndex(theState)))
        while(board[theState[0], theState[1]] != 23):
            nextState, tempReward = GetNextState(tempAction, theState, t, gamma)
            nextAction = int(GetAction(Q, GetStateIndex(nextState)))
            eT = UpdateET(eT, gamma, lamb)
            eT[GetStateIndex(theState)-1] += 1
            Q = UpdateQ(Q, tempReward, GetStateIndex(theState), tempAction, GetStateIndex(nextState), nextAction, alpha, gamma, eT)
            tempAction = nextAction
            theState = nextState
            tempReturn += tempReward
            t+=1
        theReturns.append(tempReturn)
    
    return theReturns

#AU - AR - AD - AL
epsilon = 0.9
actions = np.array([[-1, 0], [0, 1], [1, 0], [0,-1]])
board = np.zeros([5,5], dtype = int)