'''
description: Implementation of the Exponentially Weighted Average (EWA) algorithm with full information
'''

import numpy as np
from math import exp
from sys import float_info

class FullInformation:
    def __init__(self, numAction, eta, seed = 0):
        ''' initializes all attributes '''
        self.numAction = numAction
        self.eta = eta
        self.weight = [1] * numAction
        self.probability = [0] * numAction
        # np.random.seed(seed)
        # end __init__

    ''' ################################################################################################################################################################### '''
    def updateProbabilityDistribution(self, t):
        '''
        description: updates the probability distribution over available actions
        args:        self
        return:      None
        '''
        self.probability = list((weight/sum(self.weight)) for weight in self.weight)
        # end updateProbabilityDistribution

    ''' ################################################################################################################################################################### '''
    def chooseAction(self, currentTimeSlot, numAgent, currentActionIndex):
        '''
        description: selects an action at random from the probability distribution
        args:        self, current time slot, number of agents
        return:      index of the chosen action (index according to the list of weight/probability)
        '''
        return np.random.choice(list(range(self.numAction)), p=self.probability)
        # end chooseAction

    ''' ################################################################################################################################################################### '''
    def updateWeight(self, scaledGainPerNetwork):
        '''
        description: updates the weight of each action
        args:        self, current time slot - used to compute gain, index at which the weight for the chosen action is stored, the gain observed, maximum gain available
        return:      None
        '''
        scaledLossPerNetwork = FullInformation.computeLoss(self, scaledGainPerNetwork)    # compute loss
        self.weight = list((w * exp(-1 * self.eta * scaledLoss)) for w, scaledLoss in zip(self.weight, scaledLossPerNetwork))           # update weight of each action
        self.weight = [w/max(self.weight) if w / max(self.weight) > 0 else (float_info.min * float_info.epsilon) for w in self.weight]  # normalize the weights
        # end updateWeight

    ''' ################################################################################################################################################################### '''
    def computeLoss(self, scaledGainPerNetwork):
        '''
        description: computes the loss for each action
        args:        self, gain observed from chosen action, maximum gain available, index at which probability of chosen action is stored
        return:      estimated gain of chosen network
        '''
        scaledLossPerNetwork = list((max(scaledGainPerNetwork) - gain) for gain in scaledGainPerNetwork)
        return scaledLossPerNetwork
        # end computeLoss

    ''' ################################################################################################################################################################### '''
    def getAttributeName(self):
        '''
        decsription: returns the list of attributes (of this class) used by EWA algorithm
        args:        self
        return:      list of attribute names (strings)
        '''
        attributeList = []
        for i in range(1, self.numAction + 1): attributeList.append("weight %d" %(i))
        for i in range(1, self.numAction + 1): attributeList.append("probability %d" % (i))
        return attributeList
        # end getAttributeList

    ''' ################################################################################################################################################################### '''
    def getAttributeValue(self):
        '''
        description: returns a list of values of all the attributes (of this class) used by EWA algorithm
        args:        self
        return:      list of attribute values
        '''
        return self.weight + self.probability
        # end getAttributeValue
    # end class EXP3