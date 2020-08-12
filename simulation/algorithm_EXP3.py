'''
description: Implementation of the EXP3 algorithm
'''

import numpy as np
from math import exp
from sys import float_info
from copy import deepcopy

class EXP3:
    def __init__(self, numAction, seed = 0):
        ''' initializes all attributes '''
        self.numAction = numAction
        self.gamma = 1
        self.weight = [1] * numAction
        self.probability = [0] * numAction

        # to handle changes in action availability
        self.actionAvailabilityStatus = [1] * self.numAction
        # np.random.seed(seed)
        # end __init__

    ''' ################################################################################################################################################################### '''
    def updateProbabilityDistribution(self, t, changeInActionAvailability, currentActionAvailabilityStatus, currentActionIndex, deviceID):
        '''
        description: updates the probability distribution over available actions
        args:        self
        return:      None
        '''
        if changeInActionAvailability:
            if t == 1: self.weight = currentActionAvailabilityStatus
            else:
                self.weight = [x * y for x, y in zip(self.weight, currentActionAvailabilityStatus)]  # set weight of actions no longer available to zero
                # set weight of newly discovered actions to maxWeight
                maxWeight = max(self.weight) if max(self.weight) > 0 else 1
                self.weight = [self.weight[i] if currentActionAvailabilityStatus[i] == 1 and self.weight[i] > 0 else maxWeight if currentActionAvailabilityStatus[i] == 1 and self.weight[i] == 0 else 0 for i in range(self.numAction)]
                self.weight = [self.weight[i] / max(self.weight) if self.weight[i] / max(self.weight) > 0 else (float_info.min * float_info.epsilon) if self.weight[i] > 0 else 0 for i in range(len(self.weight))]  # normalize the weights
            self.actionAvailabilityStatus = deepcopy(currentActionAvailabilityStatus)
        # self.probability = list(((1 - self.gamma) * (weight/sum(self.weight))) + (self.gamma/self.numAction) for weight in self.weight)
        self.probability = list(((1 - self.gamma) * (weight / sum(self.weight))) + (self.gamma / sum(self.actionAvailabilityStatus)) if weight != 0 else 0 for weight in self.weight)
        # if sum(self.probability) != 1.0: print("gamma:", self.gamma, ", weight:", self.weight, ", prob:", self.probability, ", sum:", sum(self.probability)); input()
        # end updateProbabilityDistribution

    ''' ################################################################################################################################################################### '''
    def chooseAction(self, currentTimeSlot, numAgent, currentActionIndex, deviceID):
        '''
        description: selects an action at random from the probability distribution
        args:        self, current time slot, number of agents
        return:      index of the chosen action (index according to the list of weight/probability)
        '''
        try:
            return np.random.choice(list(range(self.numAction)), p=self.probability)
        except: print("t=", currentTimeSlot, ", sum:", sum(self.probability), ", prob:", self.probability)
        # end chooseAction

    ''' ################################################################################################################################################################### '''
    def updateWeight(self, t, chosenActionIndex, gain, maxGain, previousActionIndex, deviceID):
        '''
        description: updates the weight of the action selected
        args:        self, current time slot - used to compute gain, index at which the weight for the chosen action is stored, the gain observed, maximum gain available
        return:      None
        '''
        self.gamma = EXP3.computeGamma(self, t + 1)                                         # update gamma
        scaledGain, estimatedGain = EXP3.computeEstimatedGain(self, gain, maxGain, chosenActionIndex)   # compute scaled gain and estimated gain
        # self.weight[chosenActionIndex] *= exp((self.gamma * estimatedGain)/self.numAction)              # update weight of chosen action
        # self.weight = [w/max(self.weight) if w / max(self.weight) > 0 else (float_info.min * float_info.epsilon) for w in self.weight] # normalize the weights
        self.weight[chosenActionIndex] *= exp((self.gamma * estimatedGain) / sum(self.actionAvailabilityStatus))    # update weight of chosen action
        for i in range(len(self.weight)):
            if self.weight[i] > 0: self.weight[i] = self.weight[i] / max(self.weight) if self.weight[i] / max(self.weight) > 0 else (float_info.min * float_info.epsilon)
        # self.weight = [self.weight[i] / max(self.weight) if self.weight[i] / max(self.weight) > 0 else (float_info.min * float_info.epsilon) if self.weight[i] > 0 else 0 for i in range(len(self.weight))]  # normalize the weights
        # end updateWeight

    ''' ################################################################################################################################################################### '''
    def computeGamma(self, t):
        '''
        description: computes the value of gamma based on t, without the need to know the horizon
        args:        self, current time step t
        returns:     value of gamma for the current time step
        '''
        return t ** (-1 / 10)
        # end computeGamma

    ''' ################################################################################################################################################################### '''
    def computeEstimatedGain(self, gain, maxGain, index):
        '''
        description: computes the scaled gain and estimated gain for the action selected
        args:        self, gain observed from chosen action, maximum gain available, index at which probability of chosen action is stored
        return:      estimated gain of chosen network
        '''
        scaledGain = gain/maxGain
        return scaledGain, scaledGain/self.probability[index]
        # end computeEstimatedGain

    ''' ################################################################################################################################################################### '''
    def getAttributeName(self):
        '''
        decsription: returns the list of attributes (of this class) used by EXP3 algorithm
        args:        self
        return:      list of attribute names (strings)
        '''
        attributeList = ["gamma"]
        for i in range(1, self.numAction + 1): attributeList.append("weight %d" %(i))
        for i in range(1, self.numAction + 1): attributeList.append("probability %d" % (i))
        return attributeList
        # end getAttributeList

    ''' ################################################################################################################################################################### '''
    def getAttributeValue(self):
        '''
        description: returns a list of values of all the attributes (of this class) used by EXP3 algorithm
        args:        self
        return:      list of attribute values
        '''
        return [self.gamma] + self.weight + self.probability
        # end getAttributeValue
    # end class EXP3