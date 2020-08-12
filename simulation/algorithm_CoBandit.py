'''
description: Implementation of the Co-Bandit algorithm
'''

import numpy as np
from math import exp
from sys import float_info
from utility_method import getListIndex, combineObservation
from multiprocessing import Lock

lock = Lock()

class CoBandit:
    sharedObservation = ""                              # observations shared among agents
    def __init__(self, numAction, eta, maxTimeUnheardOfAcceptable, transmitProbability, listenProbability, seed = 0):
        ''' initializes all attributes '''
        self.numAction = numAction
        self.eta = eta
        self.weight = [1] * numAction
        self.probability = [0] * numAction
        self.message = ""                               # message to be shared; includes one's own observation and feedback received from neighbors
        self.transmitProbability = transmitProbability
        self.listenProbability = listenProbability
        self.recentGainHistoryPerAction = {}            # gain that was (or could be) observed from each action over the past few time slots
        self.actionDetailHistory = []                   # observation made/received for the past DELAY time slots - to update weight; an element has details for one time slot
        self.timeLastHeard = [-1] * self.numAction      # time slot each action was last heard of
        self.maxTimeUnheardOfAcceptable = maxTimeUnheardOfAcceptable    # maximum number of time slots a network can be unheard of
        self.exploreActionUnheardOf = False             # depending on whether the algorithm has explored an action unheard of in the current time slot
        self.numAgentPerAction = [-1] * self.numAction  # last value I know of
        # np.random.seed(seed)
        # end __init__

    ''' ################################################################################################################################################################### '''
    def updateProbabilityDistribution(self, t):
        '''
        description: updates the probability distribution over available actions
        args:        self
        return:      None
        '''
        self.probability = list((weight / sum(self.weight)) for weight in self.weight)
        # end updateProbabilityDistribution

    ''' ################################################################################################################################################################### '''
    def chooseAction(self, currentTimeSlot, numAgent):
        '''
        description: selects an action; an action that has not been chosen/heard of since a significant amount of time, or one at random from the probability distribution
        args:        self, current time slot, number of agents in the setting
        return:      index of the chosen action (index according to the list of weight/probability)
        '''
        actionUnheardOfList, actionUnheardOfProbability = CoBandit.mustExploreActionUnheardOf(self, currentTimeSlot, numAgent)
        if self.exploreActionUnheardOf == True: return np.random.choice(actionUnheardOfList, p=actionUnheardOfProbability)
        else: return np.random.choice(list(range(self.numAction)), p=self.probability)
        # end chooseAction

    ''' ################################################################################################################################################################### '''
    def updateWeight(self, currentTimeSlot, index, gain, maxGain):
        '''
        description: updates the weight of the action selected
        args:        self, current time slot - used to compute gain, index at which the weight for the chosen action is stored, the gain observed, maximum gain available
        return:      None
        '''
        scaledGainPerNetwork = []
        scaledLossPerNetwork = CoBandit.computeLoss(self, scaledGainPerNetwork)  # compute loss
        self.weight = list((w * exp(-1 * self.eta * scaledLoss)) for w, scaledLoss in zip(self.weight, scaledLossPerNetwork))               # update weight of each action
        self.weight = [w / max(self.weight) if w / max(self.weight) > 0 else (float_info.min * float_info.epsilon) for w in self.weight]    # normalize the weights
        # end updateWeight

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
    def mustTransmit(self):
        '''
        description: determines whether to transmit one's observation and feedback received
        args:        self
        return:      True of False depending on whether need to transmit or not
        '''
        if self.exploreActionUnheardOf == True: return True
        return np.random.choice([False, True], p=[1 - self.transmitProbability, self.transmitProbability])  # select and return an action
        # end mustTransmit

    ''' ################################################################################################################################################################### '''
    def transmit(self):
        '''
        description: simulates braodcasting of observations made and received from neighbors; appends them to a shared string, while dropping duplicates
        args:        self, observation(s) to be shared (as a string of values separated by ";")
        returns:     True or False, depending on whether the device transmits in the current time slot
        '''
        global lock

        if CoBandit.mustTransmit(self):
            with lock: CoBandit.sharedObservation = combineObservation(CoBandit.sharedObservation, self.message)
            return True
        return False
        # end transmit

    ''' ################################################################################################################################################################### '''
    def mustListen(self, transmit):
        '''
        description: determines whether there is a need to listen to broadcast
        args:        self, whether the device shared its observation and feedback received from others in the current time slot - in which case it won't listen
        return:      True of False depending on whether need to listen or not
        '''
        if transmit == True: return False
        return np.random.choice([False, True], p=[1 - self.listenProbability, self.listenProbability])  # select and return an action
        # end mustTransmit

    ''' ################################################################################################################################################################### '''
    def listen(self, transmit):
        '''
        description: simulates listening for feedback from neighbors; retrieves the value from a shared string
        args:        self, whether the device shared its observation and feedback received from others in the current time slot - in which case it won't listen
        returns:     observations shared during the current sub-time slot (as a string seperated by ";")
        '''
        result = ""
        if CoBandit.mustListen(self, transmit): result = CoBandit.sharedObservation
        print("heard, ", result)
        # end listen

    ''' ################################################################################################################################################################### '''
    def mustExploreActionUnheardOf(self, currentTimeSlot, numAgent):
        '''
        description: determines whether to explore an action unheard of for a significant amount of time
        args:        self, the current time slot - used to determine if any action has not been heard over a significant amount of time
        return:      list of actions unheard of over a significant amount of time, probability distribution of the actions unheard of over a significant amount of time
        '''
        actionUnheardOfList = []; actionUnheardOfProbability = []

        if (min(self.timeLastHeard) == -1 and currentTimeSlot > self.maxTimeUnheardOfAcceptable) or \
                (min(self.timeLastHeard) != -1 and (currentTimeSlot - min(self.timeLastHeard)) > self.maxTimeUnheardOfAcceptable):

            # build a list of indices of actions unheard of for more than self.maxTimeUnheardOfAcceptable time slots
            for i in list(range(self.numAction)):
                if currentTimeSlot - self.timeLastHeard[i] > self.maxTimeUnheardOfAcceptable: actionUnheardOfList.append(i)

            # any one of the unheard of network will be selected with equal probability
            exploreProbability = len(actionUnheardOfList)/numAgent
            self.exploreActionUnheardOf = np.random.choice([False, True], p=[1 - exploreProbability, exploreProbability])
            actionUnheardOfProbability = [1 / len(actionUnheardOfList)] * len(actionUnheardOfList)
            print("exploring? ", self.exploreActionUnheardOf)
        return actionUnheardOfList, actionUnheardOfProbability
        # end mustExploreActionUnheardOf

    ''' ################################################################################################################################################################### '''
    def getAttributeName(self):
        '''
        decsription: returns the list of attributes (of this class) used by Co-Bandit algorithm
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
        description: returns a list of values of all the attributes (of this class) used by Co-Bandit algorithm
        args:        self
        return:      list of attribute values
        '''
        return self.weight + self.probability
        # end getAttributeValue
    # end class EXP3