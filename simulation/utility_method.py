'''
@description:   Defines utility methods used by other programs (python files)
'''

import csv
from itertools import permutations, product
import numpy as np

''' ___________________________________________________ class definition - save device and network details to CSV file ___________________________________________________ '''
class CSVdata():
    def __init__(self, filepath, headers):
        self.filepath = filepath
        self.headers = headers
        self.rows = []

    def addRow(self, row):
        self.rows.append(tuple(row))

    def saveToFile(self, numRows):
        if len(self.rows) == numRows: saveToCSVfile(self.filepath, [self.headers] + self.rows, "w")
        else: saveToCSVfile(self.filepath, self.rows[len(self.rows) - numRows:], "a")
# end CSVdata class

''' ___________________________________________________________________ compute moving average of a list _________________________________________________________________ '''
def computeMovingAverage(values, window):
    ''' source: https://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/ '''
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma
    # end computeMovingAverage

''' __________________________________________________ computes number of devices per network at Nash equilibrium state __________________________________________________ '''
def isNashEquilibrium(combination, numNetwork, bandwidthPerNetwork):
    for i in range(numNetwork):
        for j in range(numNetwork):
            if i != j and combination[i] != 0 and bandwidthPerNetwork[i]/combination[i] < bandwidthPerNetwork[j]/(combination[j] + 1): return False
    return True

def isEpsilonEquilibrium(combination, numNetwork, bandwidthPerNetwork):
    for i in range(numNetwork):
        for j in range(numNetwork):
            if i != j and combination[i] != 0 and\
             (bandwidthPerNetwork[j]/(combination[j] + 1) - bandwidthPerNetwork[i]/combination[i]) > (EPSILON * (bandwidthPerNetwork[i]/combination[i])/100):
                return False
    return True

def computeNashEquilibriumState(numDevice, numNetwork, bandwidthPerNetwork):
    NashEquilibriumElist = []
    # epsilonEquilibriumList = []

    for item in product(range(numDevice + 1), repeat = numNetwork):
        if sum(item) == numDevice:
            if isNashEquilibrium(item, numNetwork, bandwidthPerNetwork):

                NashEquilibriumElist.append(list(item))#; print(item)
            # elif isEpsilonEquilibrium(item, numNetwork, bandwidthPerNetwork): epsilonEquilibriumList.append(item); print(item)

    return NashEquilibriumElist

''' ___________________________________________________________________ get the index of object in list __________________________________________________________________ '''
def getListIndex(networkList, searchID):
    '''
    description: returns the index in a list (e.g. networkList or self.weight) at which details of the network with ID searchID is stored
    args:        self, ID of the network whose details is being sought
    returns:     index of array at which details of the network is stored
    assumption:  the list contains a network object with the given network ID searchID
    '''
    index = 0
    while index < len(networkList) and networkList[index].networkID != searchID:
        index += 1
    return index
    # end getListIndex

''' ________________________________________________________ compute percentage element in list greater than a value _____________________________________________________ '''
def percentageElemGreaterOrEqual(alist, val):
    '''
    description: computes the percentage of elements in a list that are greater than a particular value
    args:        a list of elements alist, a value val
    returns:     percentage of elements in alist that are greater than val
    '''
    count = 0
    for num in alist:
        if num > val: count += 1
    return count * 100 / len(alist)
    # end percentageElemGreaterOrEqual

''' ____________________________________________________________ check if an observation was already received ____________________________________________________________ '''
def duplicateObservation(observationList, observation):
    # message format: [timeslot, deviceID, networkselected, bitrate, numAssociatedDevice, probabilityDistribution, ttl]
    # get the time slot, deviceID and networkselected from the observation; together they can be used to identify uniqueness of a message/feedback
    observation = observation.split(",")
    observation = ','.join(str(element) for element in observation[:3])
    # print("observation:",observation)

    observationList = observationList.split(";")
    for someObservation in observationList:
        someObservation = someObservation.split(",")
        someObservation = ','.join(str(element) for element in someObservation[:3])
        if observation == someObservation: return True
    return False
    # end duplicateObservation

''' _________________________________________________________________ combine 2 strings of observation(s) ________________________________________________________________ '''
def combineObservation(original, update):
    '''
    decsription: combines observations formatted as strings, dropping duplicate ones
    args:        original string of observations, new observations received to be appended to the original string of observations
    return:      combined string of unique observations
    '''
    if update == "": return original

    updateList = update.split(";")
    for update in updateList:
        if duplicateObservation(original, update) == False:
            if original == "": original = update
            else: original += ";" + update
    return original
    # end combineObservation

''' ____________________________________________________________ decrements the ttl value of each observation ____________________________________________________________ '''
def decrementTTL(observationStr):
    '''
    description: decrements the ttl value of every observation in observationStr and drop stale ones
     args:       string of observation
    return:      string of relevant (in time) observation(s) with the right ttl value
    '''
    # message format: [timeslot, deviceID, networkselected, bitrate, numAssociatedDevice, probabilityDistribution, ttl]
    updatedObservationStr = ""

    if observationStr != "":
        observationList = observationStr.split(";")
        # print("updatedObservationStr:", updatedObservationStr,", observationList:", observationList)
        for i in range(len(observationList)):
            observation = observationList[i].split(",")
            if int(observation[-1]) == 1: continue # skip the rest of the loop; drop the observation/feedback as it's no longer relevant (stale)
            observation[-1] = int(observation[-1]) - 1
            if updatedObservationStr == "": updatedObservationStr = ','.join(str(element) for element in observation)
            else: updatedObservationStr += ";" + ','.join(str(element) for element in observation)
    return updatedObservationStr
    # end decrementTTL

''' ___________________________________________________ computes and returns the time taken in the proper units of time __________________________________________________ '''
def getTimeTaken(startTime, endTime):
    timeTaken = endTime - startTime; unit = "seconds"
    if timeTaken > (60 ** 2): timeTaken /= (60 ** 2); unit = "hours"
    elif timeTaken > 60: timeTaken /= 60; unit = "minutes"
    return timeTaken, unit

''' ________________________________________________________________________ save data to csv file _______________________________________________________________________ '''
def saveToCSVfile(outputCSVfile, data, fileOpenMode):
    # print("saving to file", outputCSVfile)
    myfile = open(outputCSVfile, fileOpenMode)
    out = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    # print("data:",data)
    for row in data:
        try: out.writerow(row)
        except: print("error saving data:", data)
    myfile.close()
''' _____________________________________________________________________________ end of file ____________________________________________________________________________ '''