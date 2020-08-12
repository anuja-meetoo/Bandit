'''
description: Implementation of the Smart Periodic EXP4 algorithm (optimizations of Periodic EXP4)
             Optimizations include
                (a) experts are aware of what action are available at each time slot
                (b) block concept
                (c) greedy policy at start
                (d) switch back mechanism
                (e) reset block length when there is a significant decline in the quality of the network being chosen most often for a long duration of time
'''

import numpy as np
from math import exp
from sys import float_info
from math import ceil, factorial
import problem_instance
import traceback
from termcolor import colored
from copy import deepcopy
import random
import pandas

class SmartPeriodicEXP4:
    def __init__(self, numAction, numTimeSlot, numRepeat, problemInstance, periodOption, deviceID, beta = 0.1, optimization = False, maxTimeSlotConsideredPreviousBlock = 8,
                 seed = 0, rollingAverageWindowSize = 8, percentageDeclineForReset = 50, maxTimeSlotConsideredPreferredNetwork  = 40):
        ''' initializes all attributes '''
        self.numAction = numAction
        self.gamma = 1
        self.numTimeSlotPerRepetition = numTimeSlot // numRepeat
        self.partitionFunctionList = problem_instance.PERIOD_OPTIONS[periodOption]
        self.label = {}  # store list of first time slots in each period for each partition function; used to get label for a time slot; only for the first repetition
        self.b = []; SmartPeriodicEXP4.initializeB(self, problemInstance, numAction, deviceID, optimization, numTimeSlot, numRepeat) #, numTimeSlot, numRepeat, problemInstance, periodOption, optimization)
        self.probability = [0] * numAction

        # initial exploration in random order
        self.actionToExplore = []                               # the indices of the actions
        self.actionAvailabilityStatus = [0] * self.numAction

        # for block concept
        self.beta = beta
        self.blockIndex = 1
        self.blockLength = 0
        self.numBlockActionSelected = [0] * self.numAction      # keeps track of number of blocks in which each action has been selected
        self.probabilityCurrentBlock = 0

        # for hybrid
        self.maxProbabilityDifference = 1 / (self.numAction - 1)
        self.blockLengthPerAction = [1] * self.numAction        # block length initially chosen for each action
        self.blockLengthForHybrid = 0
        self.cumulativeGainPerAction = [0] * self.numAction     # cumulative scaled gain for each action
        self.numTimeSlotActionSelected = [0] * self.numAction   # keeps track of number of time slots in which each action has been selected

        # for switch back mechanism
        self.switchBack = False
        self.actionSelectedPreviousBlock = -1
        self.gainPerTimeSlotCurrentBlock = []
        self.gainPerTimeSlotPreviousBlock = []
        self.maxTimeSlotConsideredPreviousBlock = maxTimeSlotConsideredPreviousBlock
        self.actionSelectedPreviousBlockAvailability = True

        # for reset due to significant decline in action quality
        self.preferredActionIndex = -1
        self.preferredActionGainList = []
        self.rollingAverageWindowSize = rollingAverageWindowSize
        self.percentageDeclineForReset = percentageDeclineForReset
        self.maxTimeSlotConsideredPreferredNetwork =maxTimeSlotConsideredPreferredNetwork
        self.qualityDrop = False
        # print(" rollingAverageWindowSize:", self.rollingAverageWindowSize, ", percentageDeclineForReset:", self.percentageDeclineForReset, ", maxTimeSlotConsideredPreferredNetwork:", self.maxTimeSlotConsideredPreferredNetwork)
        self.log = []
        # np.random.seed(seed)
        # random.seed(seed)
        # end __init__

    ''' ################################################################################################################################################################### '''
    def initializeB(self, problemInstance, numAction, deviceID, optimization, numTimeSlot, numRepeat):
        '''
        description: initilizes the 3d list b to store the "weight" of by playing arm i during time slots assigned label l by partition function f (for each f, l anf i)
                     b is initialized to zero if the network is available at that time slot (with given label), else -1
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
        args:        self, current time slot, whether or not there is a change in action availability, availability status of each action, index of current action selected,
                     ID of device
        return:      None
        '''
        # compute r
        r = [0] * self.numAction
        for i in range(self.numAction):
            for partitionFunctionIndex, partitionFunction in enumerate(self.partitionFunctionList):
                currentLabel = SmartPeriodicEXP4.getLabel(self, t, partitionFunction)  # probability update is done at the beginning of time slot; label might have changed
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

        # set the value of blockLengthForHybrid if the probability distribution is not close to uniform from this time slot
        if SmartPeriodicEXP4.isProbabilityCloseToUniform(self) == False and self.blockLengthForHybrid == 0:
            self.blockLengthForHybrid = max(2, self.blockLengthPerAction[self.probability.index(max(self.probability))])
            # print("@t=", t, ", device", deviceID, ",blockLengthForHybrid:", self.blockLengthForHybrid)

        # current action (action chosen in the previous time slot) is no longer available
        if changeInActionAvailability: SmartPeriodicEXP4.handleChangeInNetworkAvailability(self, t, currentActionAvailabilityStatus)
        else: SmartPeriodicEXP4.reset(self, t, deviceID)  # reset block once every day

        if t != 1: self.actionSelectedPreviousBlockAvailability = (self.actionAvailabilityStatus[self.actionSelectedPreviousBlock] == 1)
        # end updateProbabilityDistribution

    ''' ################################################################################################################################################################### '''
    def reset(self, t, deviceID):
        '''
        description: resets block lengths, hence restarts a new block, after every 24 hours from start of the algorithm, and when there is significant decline in preferred
                     network chosen for a long duration of time
        args:        self, the current time slot
        return:      None
        '''
        if t != 1 and t % self.numTimeSlotPerRepetition == 1: SmartPeriodicEXP4.restartBlock(self)
        #
        # elif (self.qualityDrop == False) and (self.probability[self.preferredActionIndex] >= 0.9) and (len(self.preferredActionGainList) == self.maxTimeSlotConsideredPreferredNetwork) and \
        #     SmartPeriodicEXP4.actionQualityDecline(self, self.preferredActionGainList, self.percentageDeclineForReset):
        #     # significant decline has been detected - need to monitor the situation to see if the quality drop is for a long duration
        #     self.qualityDrop = True
        #     self.preferredActionGainList = []
        #     print(colored("@t= " + str(t) + " - " + str(deviceID) + "- quality drop - need to monitor quality for duration of drop", "blue"))
        # elif (self.qualityDrop == True) and (self.probability[self.preferredActionIndex] >= 0.9) and (len(self.preferredActionGainList)  == self.maxTimeSlotConsideredPreferredNetwork) and \
        #         (SmartPeriodicEXP4.actionQualityDecline(self, self.preferredActionGainList, 5, False) == False):
        #     print(colored("@t= " + str(t) + " - " + str(deviceID) + " reset", "cyan"))
        #     SmartPeriodicEXP4.resetBlockLength(self)
        # end mustReset

    ''' ################################################################################################################################################################### '''
    def actionQualityDecline(self, gainList, percentageDecline, explicitDecline=True):
        '''
        decsription: determines whether there is a substantial decline in quality of the preferred action (the one selected during the highest number of time slots)
        args:           self, the list of gains from which to determine is there has been a significant decline, percentage decline to consider
        return:         percentage decline from first gain in list
        True or False depending on whether there is a substantial decline in quality of the preferred action
        '''
        gainList = [x for x in gainList if x != -1]
        if len(gainList) < (self.maxTimeSlotConsideredPreferredNetwork//2): return False
        gainList = pandas.Series.rolling(pandas.Series(gainList), self.rollingAverageWindowSize).mean()    # rolling average of gain
        gainList = list(gainList[~np.isnan(gainList)])                                                                                   # remove nan from list
        changeInGain = 0; prevGain = gainList[0]
        for gain in gainList[1:]: changeInGain += (gain - prevGain); prevGain = gain
        if explicitDecline:
            return True if ((changeInGain < 0) and (abs(changeInGain) >= ((percentageDecline * gainList[0]) / 100))) else False
        return True if  abs(changeInGain) >= ((percentageDecline * gainList[0]) / 100) else False
        return changeInGain
        # end actionQualityDecline

    ''' ################################################################################################################################################################### '''
    def restartBlock(self):
        '''
        description: start a new block
        args:           self
        return:        None
        '''
        if self.blockLength != 0:   # if blockLength is already zero, then the following has already been done when updating the weight in the previous time slot
            self.blockLength = 0; self.blockIndex += 1                                                       # restart a new block and increment the block index
            self.gamma = SmartPeriodicEXP4.computeGamma(self, self.blockIndex)                               # update gamma based on the new block index
            self.gainPerTimeSlotPreviousBlock = deepcopy(self.gainPerTimeSlotCurrentBlock)                   # save the gain per time slot of the current block ending
        # self.blockLengthPerAction = [1] * self.numAction # reset count of number of times each network has been selected - maybe can remove!!!!!
        # end reset
    ''' ################################################################################################################################################################### '''
    # def resetBlockLength(self):
    #     '''
    #     description: reset the block lengths for each network
    #     args:           self
    #     return:        None
    #     '''
    #     SmartPeriodicEXP4.restartBlock(self)
    #     self.actionToExplore = [i for i in range(len(self.actionToExplore)) if self.actionToExplore[i] == 1]
    #     self.blockLengthPerAction = [1] * self.numAction
    #     self.numBlockActionSelected = [0] * self.numAction
    #     self.cumulativeGainPerAction = [0] * self.numAction
    #     self.numTimeSlotActionSelected = [0] * self.numAction
    #     self.preferredActionIndex = -1
    #     self.preferredActionGainList = []
    #     self.blockLengthForHybrid = 0
    #     self.qualityDrop = False
    #     # end reset

    ''' ################################################################################################################################################################### '''
    def handleChangeInNetworkAvailability(self, t, currentActionAvailabilityStatus):
        '''
        description: handles changes in status of action availability; set action to explore accordingly
        args:        self, current time slot, current availability status of each action
        return:      None
        '''
        # new actions detected? if the action has never been selected - add it to the list of actions to explore
        newActionAvailable = [x - y for x, y in zip(currentActionAvailabilityStatus, self.actionAvailabilityStatus)]
        for i in range(len(newActionAvailable)):
            # need to explore newly discovered networks not explored at all since the last reset/start of algorithm
            if newActionAvailable[i] == 1 and self.numBlockActionSelected[i] == 0: self.actionToExplore.append(i); self.blockLengthForHybrid = 0
            # is any action no longer available was still in the list of networks to be explored, remove it from that list
            elif newActionAvailable[i] == -1 and (i in self.actionToExplore): self.actionToExplore.remove(i)
        self.actionAvailabilityStatus = deepcopy(currentActionAvailabilityStatus)   # update action availability status

        # reset block lengths, hence restart the current block, and update the value of maxProbDiff used to control the use of greedy policy
        if t != 1: SmartPeriodicEXP4.restartBlock(self); self.maxProbabilityDifference = 1 / (sum(currentActionAvailabilityStatus) - 1); self.numBlockActionSelected = [0] * self.numAction;
        # end handleChangeInNetworkAvailability

    ''' ################################################################################################################################################################### '''
    def chooseAction(self, currentTimeSlot, numAgent, currentActionIndex, deviceID):
        '''
        description: selects an action at random from the probability distribution
        args:        self, current time slot, number of agents, the index of the current action selected
        return:      index of the chosen action (index according to the list of weight/probability)
        '''
        coinFlipped = chooseGreedily = greedySameAction = switchBack = exploration = False; numMaxAvgerageGainAction = 0; previousActionIndex = currentActionIndex
        tmpActionSelectedPreviousBlock = self.actionSelectedPreviousBlock
        if self.blockLength == 0:
            if self.actionToExplore != []:
                ''' exploration '''
                self.actionSelectedPreviousBlock = currentActionIndex; currentActionIndex = random.choice(self.actionToExplore); self.actionToExplore.remove(currentActionIndex)
                exploration = True
            elif self.switchBack and self.actionSelectedPreviousBlockAvailability:
                ''' switch back '''
                currentActionIndex = self.actionSelectedPreviousBlock; self.actionSelectedPreviousBlock = previousActionIndex; switchBack = True; self.log =["switch_back"]
            else:
                ''' hybrid '''
                self.actionSelectedPreviousBlock = currentActionIndex
                if (0 not in [self.numBlockActionSelected[i] for i in range(self.numAction) if self.actionAvailabilityStatus[i] == 1]) \
                        and (SmartPeriodicEXP4.isProbabilityCloseToUniform(self) or self.blockLengthPerAction[self.probability.index(max(self.probability))] <= self.blockLengthForHybrid):
                    # if (SmartPeriodicEXP4.isProbabilityCloseToUniform(self) == False) and (self.blockLengthPerAction[self.probability.index(max(self.probability))] <= self.blockLengthForHybrid): print("@t =", currentTimeSlot, ", device", deviceID, ", hybrid because block length < blockLengthForHybrid", self.blockLengthForHybrid);
                    coinFlip = random.randint(1, 2); coinFlipped = True
                    if coinFlip == 1:
                        numMaxAvgerageGainAction, greedySameAction, currentActionIndex = SmartPeriodicEXP4.chooseActionGreedy(self, currentActionIndex); chooseGreedily = 1
                    else: currentActionIndex = np.random.choice(list(range(self.numAction)), p=self.probability)

                else: currentActionIndex = np.random.choice(list(range(self.numAction)), p=self.probability)

            self.blockLengthPerAction[currentActionIndex] = self.blockLength = SmartPeriodicEXP4.updateBlockLength(self, self.numBlockActionSelected[currentActionIndex])
            self.numBlockActionSelected[currentActionIndex] += 1
            self.gainPerTimeSlotCurrentBlock = []
            self.probabilityCurrentBlock = self.probability[currentActionIndex]

            # set the probability with which the chosen action for the current block was selected
            self.log = []
            if exploration: self.probabilityCurrentBlock = 1 / (len(self.actionToExplore) + 1); self.log =["exploration"]
            elif switchBack and self.actionSelectedPreviousBlockAvailability: self.probabilityCurrentBlock = 1; self.log =["switch_back"]
            elif chooseGreedily:
                self.probabilityCurrentBlock = 1 / 2 if greedySameAction else (1 / 2) * (1 / numMaxAvgerageGainAction);
                if self.switchBack: self.log = ["greedy_could_not_switch_back_" + str(tmpActionSelectedPreviousBlock)]
                else: self.log =["greedy"]
            else:
                self.probabilityCurrentBlock = (1 / 2) * self.probabilityCurrentBlock if coinFlipped else self.probabilityCurrentBlock;
                if self.switchBack: self.log = [str(coinFlipped) + "-random_could_not_switch_back_" + str(tmpActionSelectedPreviousBlock)]
                else: self.log = [str(coinFlipped) + "-random"]
        return currentActionIndex
        # end chooseAction

    ''' ################################################################################################################################################################### '''
    def chooseActionGreedy(self, currentActionIndex):
        '''
        description: selects the (or one of the) action(s) with the highest average gain
        args:        self, index of action chosen in previous slot
        return:      number of actions with the same maximum average gain, whether the algorithm decides to choose the same action (in case more than one action has the same
                     maximum average gain, the index of the action with the highest average gain
        '''
        averageGainPerAction = list(totalGain / numTimeSlot if numTimeSlot > 0 else 0 for totalGain, numTimeSlot in zip(self.cumulativeGainPerAction, self.numTimeSlotActionSelected))
        averageGainPerAction = [x * y for x, y in zip(averageGainPerAction, self.actionAvailabilityStatus)] # consider only available actions
        maxAvgerageGain = max(averageGainPerAction); numMaxAvgerageGainAction = averageGainPerAction.count(maxAvgerageGain)

        if numMaxAvgerageGainAction == 1:
            return numMaxAvgerageGainAction, False, averageGainPerAction.index(maxAvgerageGain)  # a single action with the highest max average gain
        else:  # several actions with the same highest average gain; choose one at random
            indices = [i for i, x in enumerate(averageGainPerAction) if x == maxAvgerageGain]
            if currentActionIndex in indices: return numMaxAvgerageGainAction, True, currentActionIndex
            else: return numMaxAvgerageGainAction, False, random.choice(indices)
        # end chooseActionGreedy

    ''' ################################################################################################################################################################### '''
    def updateWeight(self, t, chosenActionIndex, gain, maxGain, previousActionIndex, deviceID):
        '''
        description: updates the "weight" for selecting an action in time slots  during which a particular action is selected
        args:        self, current time slot - used to compute gain, index at which the weight for the chosen action is stored, the gain observed, maximum gain available
        return:      None
        '''
        self.gamma = SmartPeriodicEXP4.computeGamma(self, self.blockIndex + 1) # update gamma
        scaledGain, estimatedGain = SmartPeriodicEXP4.computeEstimatedGain(self, gain, maxGain)  # compute scaled gain and estimated gain
        self.cumulativeGainPerAction[chosenActionIndex] += scaledGain; self.numTimeSlotActionSelected[chosenActionIndex] += 1;
        self.gainPerTimeSlotCurrentBlock.append(scaledGain)

        # need to switch back?
        SmartPeriodicEXP4.mustSwitchBack(self, chosenActionIndex, previousActionIndex, scaledGain)

        # update b
        for partitionFunctionIndex, partitionFunction in enumerate(self.partitionFunctionList):
            currentLabel = SmartPeriodicEXP4.getLabel(self, t, partitionFunction)
            self.b[partitionFunctionIndex][chosenActionIndex][currentLabel - 1] += ((self.gamma * estimatedGain) / self.numAction)

        if self.blockLength == 1:
            self.blockIndex += 1; self.gamma = SmartPeriodicEXP4.computeGamma(self, self.blockIndex)
            self.gainPerTimeSlotPreviousBlock = deepcopy(self.gainPerTimeSlotCurrentBlock)
        self.blockLength -= 1   # decrement block lengthor

        SmartPeriodicEXP4.updatePreferredActionDetail(self, scaledGain, chosenActionIndex)
        # end updateWeight

    ''' ################################################################################################################################################################### '''
    def updatePreferredActionDetail(self, currentGain, currentActionIndex):
        '''
        description:   updates details pertaining to the current preferred action (the one selected during the highest number of time slots)
        args:          self
        return:        None
        '''
        highestProbability = max(self.probability)
        currentPreferredActionIndex = self.probability.index(highestProbability)

        if self.probability.count(highestProbability) > 1:                                  # no preferred action - multiple actions have same highest count of time slots
            self.preferredActionIndex = -1
            self.preferredActionGainList = []
            self.qualityDrop = False
        elif self.preferredActionIndex != currentPreferredActionIndex:              # single network with highest count of time slots - change in preference
            self.preferredActionIndex = currentPreferredActionIndex
            self.preferredActionGainList = [currentGain]
            self.qualityDrop = False
        elif currentActionIndex == self.preferredActionIndex:                          # no change in preferred action, and current action chosen is the preferred action
            self.preferredActionGainList.append(currentGain)
        else:                                                                                                   # no change in preferred action, and current action chosen is not the preferred action
            self.preferredActionGainList.append(-1)                                          # append -1 as gain of the preferred network if it's not chosen for the current time slot
        if len(self.preferredActionGainList) > self.maxTimeSlotConsideredPreferredNetwork: self.preferredActionGainList = self.preferredActionGainList[len(self.preferredActionGainList) - self.maxTimeSlotConsideredPreferredNetwork:]
        # end updatePreferredNetworkDetail

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
    def computeGamma(self, blockIndex):
        '''
        description: computes the value of gamma based on t, without the need to know the horizon
        args:        self, current time step t
        returns:     value of gamma for the current time step
        '''
        return blockIndex ** (-1 / 10)
        # end computeGamma

    ''' ################################################################################################################################################################### '''
    def computeEstimatedGain(self, gain, maxGain):
        '''
        description: computes the scaled gain and estimated gain for the action selected
        args:        self, gain observed from chosen action, maximum gain available, index at which probability of chosen action is stored
        return:      estimated gain of chosen action
        '''
        scaledGain = gain/maxGain
        return scaledGain, scaledGain/self.probabilityCurrentBlock #self.probability[index]
        # end computeEstimatedGain

    ''' ################################################################################################################################################################### '''
    def updateBlockLength(self, numBlockActionSelected):
        '''
        description: computes the number of consecutive time slots (block length) for which a particular action must be chosen; limits the length to 20 for faster adaptation
        args:        self, the number of times the action has been selected prior to its choice in this time slot
        return:      the number of consecutive time slots (block length) for which a particular action must be chosen
        '''
        try: blockLength = min(ceil((1 + self.beta) ** numBlockActionSelected), 20); return blockLength
        except OverflowError: return 20
        # end updateBlockLength

    ''' ################################################################################################################################################################### '''
    def isProbabilityCloseToUniform(self):
        '''
        description: checks if differences among the probability values are less than or equal to 1/(k-1)
        args:        self
        return:      True or False depending on the differences among the probability values
        '''
        probability = [self.probability[i] for i in range(self.numAction) if self.probability[i] > 0]
        if (max(probability) != min(probability)) and ((max(probability) - min(probability)) > self.maxProbabilityDifference): return False
        return True
        # end isProbNearUniform

    ''' ################################################################################################################################################################### '''
    def mustSwitchBack(self, currentActionIndex, previousActionIndex, scaledGain):
        '''
        description: determines if there is a need to switch back, and set the value of the attribute switchBack accordingly
        args:        self, index of the current action selected,
        return:      None
        '''
        if not self.switchBack:
            if self.blockLength == self.blockLengthPerAction[currentActionIndex] and SmartPeriodicEXP4.isCurrentActionWorse(self, currentActionIndex, previousActionIndex, scaledGain):
                self.switchBack = True; self.blockLength = 1
        else: self.switchBack = False
        # end mustSwitchBack

    ''' ################################################################################################################################################################### '''
    def isCurrentActionWorse(self, currentActionIndex, previousActionIndex, scaledGain):
        '''
        description: determines if the current action chosen is worse than the one chosen in the previous block (considering scaled gain that excludes switching cost)
        args:        self
        return:     True if the current action chosen is worse than the one chosen in the previous block, and False otherwise
        '''
        if (0 not in [self.numBlockActionSelected[i] for i in range(self.numAction) if self.actionAvailabilityStatus[i] == 1]) \
                and (currentActionIndex != self.actionSelectedPreviousBlock):
            if len(self.gainPerTimeSlotPreviousBlock) < self.maxTimeSlotConsideredPreviousBlock: gainPerTimeSlot = self.gainPerTimeSlotPreviousBlock
            else: gainPerTimeSlot = self.gainPerTimeSlotPreviousBlock[len(self.gainPerTimeSlotPreviousBlock) - self.maxTimeSlotConsideredPreviousBlock:]

            if ((sum(gainPerTimeSlot) / len(gainPerTimeSlot) > scaledGain) or (SmartPeriodicEXP4.percentageGainGreater(self, gainPerTimeSlot, scaledGain) > 50) \
                    or (gainPerTimeSlot[-1] > scaledGain)):
                return True
            return False
        # end isCurrentActionWorse

    ''' ################################################################################################################################################################### '''
    def percentageGainGreater(self, alist, val):
        '''
        description: determines the percentage gain from a list which is greater than a certain value
        args:        alist - a list of gains, val - a certain value
        return:      percentage of gains from alist which is greater than val
        '''
        count = 0
        for num in alist:
            if num > val: count += 1
        return count * 100 / len(alist)
        # end percentageGainGreaterOrEqual

    ''' ################################################################################################################################################################### '''
    def getAttributeName(self):
        '''
        decsription: returns the list of attributes (of this class) used by EXP3 algorithm
        args:        self
        return:      list of attribute names (strings)
        '''
        attributeList = ["gamma"]
        for i in range(1, self.numAction + 1): attributeList.append("probability %d" % (i))
        attributeList += ['blockIndex', 'blockLength', 'log', 'probCurrentBlock', 'actionToExplore', 'numBlockActionSelected']
        return attributeList
        # end getAttributeList

    ''' ################################################################################################################################################################### '''
    def getAttributeValue(self):
        '''
        description: returns a list of values of all the attributes (of this class) used by EXP3 algorithm
        args:        self
        return:      list of attribute values
        '''
        return [self.gamma] + self.probability + [self.blockIndex, self.blockLength] + self.log + [self.probabilityCurrentBlock] + [deepcopy(self.actionToExplore)] \
               + [deepcopy(self.numBlockActionSelected)]
        # end getAttributeValue
    # end class SmartPeriodicEXP4
