'''
description: Implementation of the Periodic EXP4 algorithm
             Optimizations include (a) experts are aware of what networks are available at each time slot
'''

import numpy as np
from math import exp
from sys import float_info
from math import ceil
import problem_instance
import traceback
from termcolor import colored
from copy import deepcopy

class PeriodicEXP4:
    def __init__(self, numAction, numTimeSlot, numRepeat, problemInstance, periodOption, deviceID, optimization = False, seed = 0):
        ''' initializes all attributes '''
        self.numAction = numAction
        self.gamma = 1
        self.numTimeSlotPerRepetition = numTimeSlot // numRepeat
        self.partitionFunctionList = problem_instance.PERIOD_OPTIONS[periodOption]
        self.label = {}  # store list of first time slots in each period for each partition function; used to get label for a time slot; only for the first repetition
        self.b = []; PeriodicEXP4.initializeB(self, problemInstance, numAction, deviceID, optimization, numTimeSlot, numRepeat) #, numTimeSlot, numRepeat, problemInstance, periodOption, optimization)
        self.probability = [0] * numAction
        # np.random.seed(seed)
        print("optimization:", optimization)
        # end __init__

    ''' ################################################################################################################################################################### '''
    def initializeB(self, problemInstance, numAction, deviceID, optimization, numTimeSlot, numRepeat):
        '''
        description: initilizes the 3d list b to store the "weight" of by playing arm i during time slots assigned label l by partition function f (for each f, l anf i)
        args:        problem instance chosen, ID of the device running the algorithm, whether or not the optimized version of the algorithm is being considered
        return:      the 3d list b
        '''
        # initialize the 3D list b, and compute the first time slot in each period for each partition function
        for numPeriod in self.partitionFunctionList:
            self.b.append([[0 for col in range(numPeriod)] for row in range(numAction)]) # b[f][i][l] - 3D list; for each partition, 2D list - row refers to an action, col refers to label
            self.label.update({numPeriod: [1 + (x * self.numTimeSlotPerRepetition // numPeriod) for x in range(numPeriod)]})

        if optimization == True:
            # optimization == True => we assume that the experts are aware of which actions are available at each time slots and recommend only available actions
            # get time of change in action availability
            timeOfChangeInActionAvailability = problem_instance.getTimeOfChangeInNetworkAvailability(problemInstance, deviceID) # throughout simulation run
            timeOfChangeInActionAvailability = [x for x in timeOfChangeInActionAvailability if x <= numTimeSlot/numRepeat]

            for partitionFunctionIndex, numPeriod in enumerate(self.partitionFunctionList):      # for each partition
                # if any time of change in action availability not in the list of first time slot in each partition,
                # (1) add it to the list, and (2) add a new element to the 2D b list for that partition
                additionalTimeOfChange = list(set(timeOfChangeInActionAvailability) - set(self.label.get(numPeriod)))
                if additionalTimeOfChange != []:
                    self.label.update({numPeriod: sorted(self.label.get(numPeriod) + additionalTimeOfChange)})
                    for i in range(len(self.b[partitionFunctionIndex])): self.b[partitionFunctionIndex][i] += ([0] * len(additionalTimeOfChange))

            # set the value of self.b to zero instead of one for action i when it is not available
            for partitionFunctionIndex, numPeriod in enumerate(self.partitionFunctionList):  # for each partition
                for labelIndex, time in enumerate(sorted(self.label.get(numPeriod))):
                    for actionID in range(1, numAction + 1):
                        if not problem_instance.isNetworkAccessible(problemInstance, time, actionID, deviceID):
                            self.b[partitionFunctionIndex][actionID - 1][labelIndex] = -1
        # end initializeB

    ''' ################################################################################################################################################################### '''
    def updateProbabilityDistribution(self, t, changeInActionAvailability, currentActionAvailabilityStatus, currentActionIndex, deviceID):
        '''
        description: updates the probability distribution over available actions
        args:        self, current time slot
        return:      None
        '''
        # compute r
        r = [0] * self.numAction
        for i in range(self.numAction):
            for partitionFunctionIndex, partitionFunction in enumerate(self.partitionFunctionList):
                currentLabel = PeriodicEXP4.getLabel(self, t, partitionFunction)  # probability update is done at the beginning of time slot; label might have changed
                if self.b[partitionFunctionIndex][i][currentLabel - 1] != -1:
                    tmpSumBLabels = 0
                    for label in range(1, len(self.b[partitionFunctionIndex][0]) + 1): #partitionFunction + 1): # b[f][i][l]
                        if label != currentLabel: tmpSumBLabels += max([self.b[partitionFunctionIndex][actionIndex][label - 1] for actionIndex in range(self.numAction)])
                    tmpR = self.b[partitionFunctionIndex][i][currentLabel - 1] + tmpSumBLabels
                    if tmpR > r[i]: r[i] = tmpR
                else: r[i] = "-"
        r = [x - max([y for y in r if y != "-"]) if x != "-" else "-" for x in r]; r = [exp(x) if x != "-" else 0 for x in r]

        # compute probability
        self.probability = list(x/sum(r) for x in r)
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
        except:
            print(colored("prob:" + str(self.probability), "blue"))
            traceback.print_exc()
        # end chooseAction

    ''' ################################################################################################################################################################### '''
    def updateWeight(self, t, chosenActionIndex, gain, maxGain, prevNetworkSelected, deviceID):
        '''
        description: updates the "weight" for selecting an action in time slots  during which a particular network is selected
        args:        self, current time slot - used to compute gain, index at which the weight for the chosen action is stored, the gain observed, maximum gain available
        return:      None
        '''

        self.gamma = PeriodicEXP4.computeGamma(self, t + 1)  # update gamma
        scaledGain, estimatedGain = PeriodicEXP4.computeEstimatedGain(self, gain, maxGain, chosenActionIndex)  # compute scaled gain and estimated gain

        # update b
        for partitionFunctionIndex, partitionFunction in enumerate(self.partitionFunctionList):
            currentLabel = PeriodicEXP4.getLabel(self, t, partitionFunction)
            self.b[partitionFunctionIndex][chosenActionIndex][currentLabel - 1] += ((self.gamma * estimatedGain) / self.numAction)
        # end updateWeight

    ''' ################################################################################################################################################################### '''
    def getLabel(self, t, numPartition):
        '''
        description: determines the label assigned to the current time slot by a given partition function
        args:        self, current time step t, number of partitions used by the partition function
        returns:     the label assigned to the current time slot by the partition function (which divides the time equally into period time slots)
        '''
        equivalentTimeSlotInFirstRepetition = self.numTimeSlotPerRepetition if t % self.numTimeSlotPerRepetition == 0 else t % self.numTimeSlotPerRepetition
        if equivalentTimeSlotInFirstRepetition >= sorted(self.label.get(numPartition))[-1]: return len(self.label.get(numPartition))
        else: return sorted(self.label.get(numPartition)).index(max(i for i in sorted(self.label.get(numPartition)) if i <= equivalentTimeSlotInFirstRepetition)) + 1
        # end getLabel

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
        return [self.gamma] + self.probability # + [deepcopy(self.b)]
        # end getAttributeValue
    # end class PeriodicEXP4