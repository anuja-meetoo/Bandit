'''
description: Implementation of an optimal random algorithm that assumes prior knowledge of the actual gain of each action,
             and, in each time slot, selects an action randomly from a probability distribution equal to the ratios of the gains.
'''

import numpy as np
from math import exp
from sys import float_info

class FullInformation:
    def __init__(self, numAction, eta, seed = 0):
        ''' initializes all attributes '''
        self.numAction = numAction
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
        self.probability = []
        # end updateProbabilityDistribution

    ''' ################################################################################################################################################################### '''
    def chooseAction(self, currentTimeSlot, numAgent):
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
        return
        # end updateWeight

    ''' ################################################################################################################################################################### '''
    def getAttributeName(self):
        '''
        decsription: returns the list of attributes (of this class) used by EWA algorithm
        args:        self
        return:      list of attribute names (strings)
        '''
        attributeList = []
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
        return self.probability
        # end getAttributeValue
    # end class EXP3