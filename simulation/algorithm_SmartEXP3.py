'''
description: Implementation of the SmartEXP3 algorithm
'''

import numpy as np
from math import exp
from sys import float_info
from math import ceil
from copy import deepcopy
import random
import pandas

class SmartEXP3:
    def __init__(self, numAction, beta = 0.1, maxTimeSlotConsideredPreviousBlock = 8, convergedProbability = 0.75, numConsecutiveSlotForReset = 4,
                 rollingAverageWindowSize = 12, percentageDeclineForReset = 15, minBlockLengthForReset = 40, seed = 0):
        ''' initializes all attributes '''
        self.numAction = numAction
        self.beta = beta
        self.gamma = 1
        self.weight = [1] * numAction
        self.probability = [0] * numAction

        # for block concept
        self.blockIndex = 1
        self.blockLength = 0
        self.numBlockActionSelected = [0] * self.numAction     # keeps track of number of blocks in which each action has been selected
        self.probabilityCurrentBlock = 0

        # for hybrid
        self.maxProbabilityDifference = 1/(self.numAction - 1)
        self.blockLengthPerAction = [1] * self.numAction        # block length initially chosen for each action
        self.blockLengthForHybrid = 0
        self.cumulativeGainPerAction = [0] * self.numAction     # cumulative scaled gain for each action
        self.numTimeSlotActionSelected = [0] * self.numAction   # keeps track of number of time slots in which each action has been selected

        # for switch back mechanism
        self.actionToExplore = [] #list(range(self.numAction))      # the indices of the actions
        self.switchBack = False
        self.actionSelectedPreviousBlock = -1
        self.gainPerTimeSlotCurrentBlock = []
        self.gainPerTimeSlotPreviousBlock = []
        self.maxTimeSlotConsideredPreviousBlock = maxTimeSlotConsideredPreviousBlock

        # for periodic reset
        self.numConsecutiveSlotPreferredActionChosen = 0
        self.preferredActionGainList = []
        self.preferredActionIndex = -1
        self.convergedProbability = convergedProbability
        self.numConsecutiveSlotForReset = numConsecutiveSlotForReset
        self.rollingAverageWindowSize = rollingAverageWindowSize
        self.percentageDeclineForReset = percentageDeclineForReset
        self.minBlockLengthForReset = minBlockLengthForReset

        # to handle changes in action availability
        self.actionAvailabilityStatus = [1] * self.numAction
        self.actionSelectedPreviousBlockAvailability = True

        # np.random.seed(seed)
        # random.seed(seed)
        # end __init__

    ''' ################################################################################################################################################################### '''
    def updateProbabilityDistribution(self, t, changeInActionAvailability, currentActionAvailabilityStatus, currentActionIndex, deviceID):
        '''
        description: updates the probability distribution over available actions
        args:        self
        return:      None
        '''
        # handle reset due to change in action availability; comment it to allow the algorithm to be oblivious to availability of actions
        resetDueToChangeInActionAvailability = SmartEXP3.handleChangeInActionAvailability(self, t, currentActionAvailabilityStatus, currentActionIndex) \
            if changeInActionAvailability == True else False
        if t != 1 and changeInActionAvailability == True: self.actionSelectedPreviousBlockAvailability = (self.actionAvailabilityStatus[self.actionSelectedPreviousBlock] == 1)

        # compute probability distribution
        gamma = self.gamma if self.blockLength == 0 else SmartEXP3.computeGamma(self, self.blockIndex + 1)
        self.probability = list(((1 - gamma) * (weight/sum(self.weight))) + (gamma/sum(self.actionAvailabilityStatus)) if weight != 0 else 0 for weight in self.weight)

        # update the value of self.blockLengthForHybrid if the distribution is not close to uniform as from this time slot
        if SmartEXP3.isProbabilityCloseToUniform(self) == False and self.blockLengthForHybrid == 0:
            self.blockLengthForHybrid = max(2, self.blockLengthPerAction[self.probability.index(max(self.probability))])

        # need to reset the algorithm (by resetting the block length)?
        if t != 1 and not resetDueToChangeInActionAvailability: SmartEXP3.mustReset(self)
        # end updateProbabilityDistribution

    ''' ################################################################################################################################################################### '''
    def chooseAction(self, currentTimeSlot, numAgent, currentActionIndex, deviceID):
        '''
        description: selects an action at random from the probability distribution
        args:        self, current time slot, number of agents
        return:      index of the chosen action (index according to the list of weight/probability)
        '''
        coinFlipped = chooseGreedily = greedySameAction = switchBack = exploration = False; numMaxAvgerageGainAction = 0; previousActionIndex = currentActionIndex
        if self.blockLength == 0:
            if self.actionToExplore != []:
                ''' exploration '''
                self.actionSelectedPreviousBlock = currentActionIndex
                currentActionIndex = random.choice(self.actionToExplore)
                self.actionToExplore.remove(currentActionIndex)
                exploration = True
            elif self.switchBack:
                ''' switch back '''
                currentActionIndex = self.actionSelectedPreviousBlock
                self.actionSelectedPreviousBlock = previousActionIndex
                switchBack = True
            else:
                ''' hybrid '''
                self.actionSelectedPreviousBlock = currentActionIndex
                if (0 not in [self.numBlockActionSelected[i] for i in range(self.numAction) if self.actionAvailabilityStatus[i] == 1]) \
                        and (SmartEXP3.isProbabilityCloseToUniform(self) or self.blockLengthPerAction[self.probability.index(max(self.probability))] <= self.blockLengthForHybrid):
                    coinFlip = random.randint(1, 2); coinFlipped = True
                    if coinFlip == 1:
                        numMaxAvgerageGainAction, greedySameAction, currentActionIndex = SmartEXP3.chooseActionGreedy(self, currentActionIndex)
                        chooseGreedily = 1
                    else: currentActionIndex = np.random.choice(list(range(self.numAction)), p=self.probability)
                else: currentActionIndex = np.random.choice(list(range(self.numAction)), p=self.probability)

            self.blockLengthPerAction[currentActionIndex] = self.blockLength = SmartEXP3.updateBlockLength(self, self.numBlockActionSelected[currentActionIndex])
            self.numBlockActionSelected[currentActionIndex] += 1
            self.gainPerTimeSlotCurrentBlock = []
            self.probabilityCurrentBlock = self.probability[currentActionIndex]

            # set the probability with which the chosen action for the current block was selected
            if exploration: self.probabilityCurrentBlock = 1/(len(self.actionToExplore) + 1)
            elif switchBack: self.probabilityCurrentBlock = 1
            elif chooseGreedily: self.probabilityCurrentBlock = 1/2 if greedySameAction else (1/2) * (1/numMaxAvgerageGainAction)
            else: self.probabilityCurrentBlock = (1/2) * self.probabilityCurrentBlock if coinFlipped else self.probabilityCurrentBlock

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
        averageGainPerAction = list(totalGain/numTimeSlot if numTimeSlot > 0 else 0 for totalGain, numTimeSlot in zip(self.cumulativeGainPerAction, self.numTimeSlotActionSelected))
        maxAvgerageGain = max(averageGainPerAction); numMaxAvgerageGainAction = averageGainPerAction.count(maxAvgerageGain)

        if numMaxAvgerageGainAction == 1: return numMaxAvgerageGainAction, False, averageGainPerAction.index(maxAvgerageGain) # a single action with the highest max average gain
        else: # several actions with the same highest average gain; choose one at random
            indices = [i for i, x in enumerate(averageGainPerAction) if x == maxAvgerageGain]
            if currentActionIndex in indices: return numMaxAvgerageGainAction, True, currentActionIndex
            else: return numMaxAvgerageGainAction, False, random.choice(indices)
        # end chooseActionGreedy

    ''' ################################################################################################################################################################### '''
    def updateWeight(self, t, chosenActionIndex, gain, maxGain, previousActionIndex, deviceID):
        '''
        description: updates the weight of the action selected
        args:        self, current time slot - used to compute gain, index at which the weight for the chosen action is stored, the gain observed, maximum gain available
        return:      None
        '''
        gamma = SmartEXP3.computeGamma(self, self.blockIndex + 1)
        scaledGain, estimatedGain = SmartEXP3.computeEstimatedGain(self, gain, maxGain) # compute scaled gain and estimated gain
        self.cumulativeGainPerAction[chosenActionIndex] += scaledGain; self.numTimeSlotActionSelected[chosenActionIndex] += 1;
        self.gainPerTimeSlotCurrentBlock.append(scaledGain)

        # need to switch back?
        SmartEXP3.mustSwitchBack(self, chosenActionIndex, previousActionIndex, scaledGain)

        # try: self.weight[chosenActionIndex] *= exp((gamma * estimatedGain) / self.numAction)    # update weight of chosen action
        try: self.weight[chosenActionIndex] *= exp((gamma * estimatedGain) / sum(self.actionAvailabilityStatus))    # update weight of chosen action
        except OverflowError: self.weight[chosenActionIndex] = 1                        # in case of overflow, set to 1
        # self.weight = [w / max(self.weight) if w / max(self.weight) > 0 else (float_info.min * float_info.epsilon) for w in self.weight]  # normalize the weights
        self.weight = [self.weight[i] / max(self.weight) if self.weight[i] / max(self.weight) > 0 else (float_info.min * float_info.epsilon) if self.weight[i] > 0 else 0 for i in range(len(self.weight))]  # normalize the weights

        if self.blockLength == 1:
            self.blockIndex += 1; self.gamma = SmartEXP3.computeGamma(self, self.blockIndex); self.gainPerTimeSlotPreviousBlock = deepcopy(self.gainPerTimeSlotCurrentBlock)
        self.blockLength -= 1   # decrement block length

        SmartEXP3.updatePreferredActionDetail(self, scaledGain, chosenActionIndex)
        # end updateWeight

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
        return:      estimated gain of chosen network
        '''
        scaledGain = gain / maxGain

        return scaledGain, scaledGain / self.probabilityCurrentBlock #self.probability[index]
        # end computeEstimatedGain

    ''' ################################################################################################################################################################### '''
    def updateBlockLength(self, numBlockNetworkSelected):
        return ceil((1 + self.beta) ** numBlockNetworkSelected)
        # end updateBlockLength

    ''' ################################################################################################################################################################### '''
    def isProbabilityCloseToUniform(self):
        '''
        description: checks if differences among the probability values are less than or equal to 1/(k-1)
        args:        self
        return:      True or False depending on the differences among the probability values
        '''
        probability = [self.probability[i] for i in range(self.numAction) if self.probability[i] > 0]
        if (max(probability) - min(probability)) > self.maxProbabilityDifference: return False
        # if (max(self.probability) - min(self.probability)) > self.maxProbabilityDifference: return False
        return True
        # end isProbNearUniform

    ''' ################################################################################################################################################################### '''
    def isCurrentActionWorse(self, currentActionIndex, previousActionIndex, scaledGain):
        '''
        description: determines if the current action chosen is worse than the one chosen in the previous block (considering scaled gain that excludes switching cost)
        args:        self
        return:     True if the current action chosen is worse than the one chosen in the previous block, and False otherwise
        '''
        # if (0 not in self.numBlockActionSelected) and (currentActionIndex != self.actionSelectedPreviousBlock):
        if (0 not in [self.numBlockActionSelected[i] for i in range(self.numAction) if self.actionAvailabilityStatus[i] == 1]) \
                and (currentActionIndex != self.actionSelectedPreviousBlock):
            if len(self.gainPerTimeSlotPreviousBlock) < self.maxTimeSlotConsideredPreviousBlock: gainPerTimeSlot = self.gainPerTimeSlotPreviousBlock
            else: gainPerTimeSlot = self.gainPerTimeSlotPreviousBlock[len(self.gainPerTimeSlotPreviousBlock) - self.maxTimeSlotConsideredPreviousBlock:]

            if ((sum(gainPerTimeSlot)/len(gainPerTimeSlot) > scaledGain) or (SmartEXP3.percentageGainGreater(self, gainPerTimeSlot, scaledGain) > 50) or
                    (gainPerTimeSlot[-1] > scaledGain)):
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
    def mustSwitchBack(self, currentActionIndex, previousActionIndex, scaledGain):
        '''
        description: determines if there is a need to switch back, and set the value of the attribute switchBack accordingly
        args:
        return: None
        '''
        if not self.switchBack and self.actionSelectedPreviousBlockAvailability:
            if self.blockLength == self.blockLengthPerAction[currentActionIndex] and SmartEXP3.isCurrentActionWorse(self, currentActionIndex, previousActionIndex, scaledGain):
                self.switchBack = True; self.blockLength = 1
        else: self.switchBack = False
        # end mustSwitchBack

    ''' ################################################################################################################################################################### '''
    def mustReset(self):
        '''
        description: determines whether there is a need to reset the algorithm (resetting the block length)
        args:        self
        return:      None
        '''
        if (max(self.probability) >= self.convergedProbability and (self.blockLengthPerAction[self.probability.index(max(self.probability))] >= self.minBlockLengthForReset)) \
            or (self.numConsecutiveSlotPreferredActionChosen > self.numConsecutiveSlotForReset and len(self.preferredActionGainList) >= (self.rollingAverageWindowSize + 1)
                and SmartEXP3.actionQualityDecline(self)):
            if self.blockLength != 0:
                self.blockIndex += 1
                self.gamma = SmartEXP3.computeGamma(self, self.blockIndex)
            self.gainPerTimeSlotPreviousBlock = deepcopy(self.gainPerTimeSlotCurrentBlock)

            SmartEXP3.resetActionBlockLength(self)
            self.blockLength = 0
        # end mustReset

    ''' ################################################################################################################################################################### '''
    def actionQualityDecline(self):
        '''
        decsription: determines whether there is a substantial decline in quality of the preferred action (the one selected during the highest number of time slots)
        args:        self
        return:      True or False depending on whether there is a substantial decline in quality of the preferred action
        '''
        # gainList = pandas.rolling_mean(np.array(self.preferredActionGainList), self.rollingAverageWindowSize) # rolling average of gain
        gainList = pandas.Series.rolling(pandas.Series(self.preferredActionGainList), self.rollingAverageWindowSize).mean()  # rolling average of gain
        gainList = list(gainList[~np.isnan(gainList)])  # remove nan from list

        changeInGain = SmartEXP3.computeActionQualityChange(self, gainList)
        return True if ((changeInGain < 0) and (abs(changeInGain) >= (self.percentageDeclineForReset * gainList[0]) / 100)) else False
        # end actionQualityDecline

    ''' ################################################################################################################################################################### '''
    def resetActionBlockLength(self):
        '''
        description: resets the block length and attributes used for greedy selection
        args:        self
        return:      None
        '''
        self.blockLengthPerAction = [1] * self.numAction
        self.numBlockActionSelected = [0] * self.numAction
        self.cumulativeGainPerAction = [0] * self.numAction
        self.numTimeSlotActionSelected = [0] * self.numAction
        # self.actionToExplore = list(range(self.numAction))
        self.actionToExplore = [i for i in range(self.numAction) if self.actionAvailabilityStatus[i] == 1]

        self.preferredActionIndex = -1
        self.numConsecutiveSlotPreferredActionChosen = 0
        self.preferredActionGainList = []
        # end resetActionBlockLength

    ''' ################################################################################################################################################################### '''
    def computeActionQualityChange(self, gainList):
        '''
        description: computes the change in quality of a particular action given a list of (scaled) gains observed from that action
        args:        self, list of (scaled) gains observed from an action
        return:      change in gain
        '''
        changeInGain = 0

        prevGain = gainList[0]
        for gain in gainList[1:]: changeInGain += (gain - prevGain); prevGain = gain
        return changeInGain
        # end computeActionQualityChange

    ''' ################################################################################################################################################################### '''
    def updatePreferredActionDetail(self, currentGain, currentActionIndex):
        '''
        description:   updates details pertaining to the current preferred action (the one selected during the highest number of time slots)
        args:          self
        return:        None
        '''
        highestCountTimeSlot = max(self.numTimeSlotActionSelected)
        currentPreferredActionIndex = self.numTimeSlotActionSelected.index(highestCountTimeSlot)

        if self.numTimeSlotActionSelected.count(highestCountTimeSlot) > 1:  # no preferred action - multiple actions have same highest count of time slots
            self.preferredActionIndex = -1
            self.numConsecutiveSlotPreferredActionChosen = 0
            self.preferredActionGainList = []
        elif self.preferredActionIndex != currentPreferredActionIndex:      # single network with highest count of time slots - change in preference
            self.preferredActionIndex = currentPreferredActionIndex
            self.numConsecutiveSlotPreferredActionChosen = 1
            self.preferredActionGainList = [currentGain]
        elif currentActionIndex == self.preferredActionIndex:               # no change in preferred action
            self.numConsecutiveSlotPreferredActionChosen += 1
            self.preferredActionGainList.append(currentGain)
        else: self.numConsecutiveSlotPreferredActionChosen = 0
        # end updatePreferredNetworkDetail

    ''' ################################################################################################################################################################### '''
    def handleChangeInActionAvailability(self, t, currentActionAvailabilityStatus, currentActionIndex):
        resetDueToChangeInActionAvailability = False

        highestProbability = max(self.probability); actionWithHighestProbability = self.probability.index(highestProbability)

        if t == 1: self.weight = currentActionAvailabilityStatus
        else:
            self.weight = [x * y for x, y in zip(self.weight, currentActionAvailabilityStatus)]   # set weight of actions no longer available to zero
            # set weight of newly discovered actions to maxWeight
            maxWeight = max(self.weight) if max(self.weight) > 0 else 1
            self.weight = [self.weight[i] if currentActionAvailabilityStatus[i] == 1 and self.weight[i] > 0 else maxWeight if currentActionAvailabilityStatus[i] == 1 and self.weight[i] == 0 else 0 for i in range(self.numAction)]
            self.weight = [self.weight[i] / max(self.weight) if self.weight[i] / max(self.weight) > 0 else (float_info.min * float_info.epsilon) if self.weight[i] > 0 else 0 for i in range(len(self.weight))]  # normalize the weights
        self.numBlockActionSelected = [x * y for x, y in zip(self.numBlockActionSelected, currentActionAvailabilityStatus)]
        self.blockLengthPerAction = [x * y for x, y in zip(self.blockLengthPerAction, currentActionAvailabilityStatus)]
        self.cumulativeGainPerAction = [x * y for x, y in zip(self.cumulativeGainPerAction, currentActionAvailabilityStatus)]
        self.numTimeSlotActionSelected = [x * y for x, y in zip(self.numTimeSlotActionSelected, currentActionAvailabilityStatus)]
        self.maxProbabilityDifference = 1/(sum(currentActionAvailabilityStatus) - 1)

        convergedActionUnavailability = highestProbability >= self.convergedProbability and currentActionAvailabilityStatus[actionWithHighestProbability] == 0
        newActionDiscovered = 1 in [x - y for x, y in zip(currentActionAvailabilityStatus, self.actionAvailabilityStatus)]
        currentActionUnavailability = currentActionAvailabilityStatus[currentActionIndex] == 0

        self.actionAvailabilityStatus = deepcopy(currentActionAvailabilityStatus)
        self.actionToExplore = [i for i in range(self.numAction) if currentActionAvailabilityStatus[i] == 1]  # also in resetActionBlockLength; must be called after setting self.actionAvailabilityStatus

        if t != 1 and (convergedActionUnavailability or newActionDiscovered or currentActionUnavailability):
            if convergedActionUnavailability or newActionDiscovered:
                SmartEXP3.resetActionBlockLength(self)
                self.blockIndex = 1
                self.blockLengthForHybrid = 0
                resetDueToChangeInActionAvailability = True
            else: self.blockIndex += 1
            self.blockLength = 0
            self.gamma = SmartEXP3.computeGamma(self, self.blockIndex)
            self.gainPerTimeSlotPreviousBlock = deepcopy(self.gainPerTimeSlotCurrentBlock)

        return resetDueToChangeInActionAvailability
        # end handleChangeInActionAvailability

    ''' ################################################################################################################################################################### '''
    def getAttributeName(self):
        '''
        decsription: returns the list of attributes (of this class) used by EXP3 algorithm
        args:        self
        return:      list of attribute names (strings)
        '''
        attributeList = ["gamma", "blockIndex", "blockLength", "#blockActionSelected"]
        for i in range(1, self.numAction + 1): attributeList.append("weight %d" % (i))
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
        return [self.gamma, self.blockIndex, self.blockLength, [deepcopy(self.numBlockActionSelected)]] + self.weight + self.probability
        # end getAttributeValue
    # end class EXP3