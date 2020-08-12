#!/usr/bin/python3
'''
@description:   Simulate a collaborative version of Exponential Weighted Average for wireless network selection which allows devices to share their observations with their
                neighbors, with the aim of improving the rate of convergence to Nash equilibrium.
@assumptions:   (1) all devices are in the service area throughout the simulation, (2) all devices see the same wireless networks, (3) mobile devices are time-synchronized,
                (4) mobile devices have enough battery to collaborate with neighbors (broadcasting data using BLE and scanning for data), (5) all devices who broadcast data
                at a time slot are able to get their data to all devices who are listening at that time.
@author:        Anuja
@date:          18 January 2018; @update: 8 May 2018, 1-2 October 2018
'''

import simpy
from network import Network
import global_setting
import argparse
import os
from statistics import median, stdev
import numpy as np
from copy import deepcopy
from utility_method import getTimeTaken, computeNashEquilibriumState, isNashEquilibrium, saveToCSVfile, computeMovingAverage
from computeDistanceToNashEquilibrium import computeDistanceToNashEquilibrium
import time
from algorithm_EXP3 import EXP3
from algorithm_FullInformation import FullInformation
from algorithm_CoBandit import CoBandit
from algorithm_SmartEXP3 import SmartEXP3
from algorithm_ContextualSmartEXP3 import ContextualSmartEXP3
from algorithm_PeriodicEXP4 import PeriodicEXP4
from algorithm_SmartPeriodicEXP4 import SmartPeriodicEXP4
import problem_instance

''' ______________________________________________________________________________ constants ______________________________________________________________________________ '''
# set from values passed as arguments when the program is executed
boolstr = lambda s : (s.lower() == 'true')
parser = argparse.ArgumentParser(description='Simulates the wireless network selection by a number of wireless devices in the service area.')
parser.add_argument('-n', dest="num_device", required=True, help='number of active devices in the service area')
parser.add_argument('-t', dest="num_time_slot", required=True, help='number of time slots in the simulation run')
parser.add_argument('-r', dest="run_index", required=True, help='current run index')
parser.add_argument('-a', dest="algorithm_name", required=True, help='name of selection algorithm used by the devices')
parser.add_argument('-dir', dest="directory", required=True, help='root directory containing the simulation files')
parser.add_argument('-st', dest="num_sub_time_slot", required=True, help='number of sub-time slots in one time slot')
parser.add_argument('-d', dest="delay", required=True, help='maximum delayed feedback considered')
parser.add_argument('-l', dest="learning_rate", required=True, help='learning rate')
parser.add_argument('-pt', dest="transmit_probability", required=True, help='probability with which to transmit')
parser.add_argument('-pl', dest="listen_probability", required=True, help='probability with which to listen')
parser.add_argument('-max', dest="max_time_unheard_acceptable", required=True, help='maximum time a network can be unheard of')
parser.add_argument('-f', dest="save_to_file_frequency", required=True, help='frequency of saving to file')
parser.add_argument('-sd', dest="time_slot_duration", required=True, help='duration of a time slot (in seconds)')
parser.add_argument('-p', dest="problem_instance", required=True, help='the problem instance (setting) chosen')
parser.add_argument('-rep', dest="num_repeat", required=True, help='number of times the setting is repeated over the time horizon')
parser.add_argument('-opt', dest="optimization", required=True, type=boolstr, help='whether the optimized version of the chosen algorithm must be used')
parser.add_argument('-period', dest='period_option', required=True, help='periods considered - partition functions to be used')
parser.add_argument('-roll', dest='rolling_average_window', required=True, help='rolling average window size for distance to Nash equilibrium')

args = parser.parse_args()
NUM_MOBILE_DEVICE = int(args.num_device); global_setting.constants.update({'num_mobile_device':NUM_MOBILE_DEVICE})
NUM_TIME_SLOT = int(args.num_time_slot); global_setting.constants.update({'num_time_slot':NUM_TIME_SLOT})
global_setting.constants.update({'num_sub_time_slot':int(args.num_sub_time_slot)})  # per time slot
global_setting.constants.update({'delay':int(args.delay)})
global_setting.constants.update({'run_num':int(args.run_index)})
ALGORITHM_NAME = args.algorithm_name; global_setting.constants.update({'algorithm_name':ALGORITHM_NAME})
DIR = args.directory; global_setting.constants.update({'output_dir':DIR})
PROBLEM_INSTANCE = args.problem_instance; global_setting.constants.update({'problem_instance':PROBLEM_INSTANCE})
LEARNING_RATE = float(args.learning_rate); global_setting.constants.update({'learning_rate':LEARNING_RATE})
TRANSMIT_PROBABILITY = float(args.transmit_probability); global_setting.constants.update({'p_t':TRANSMIT_PROBABILITY})
LISTEN_PROBABILITY = float(args.listen_probability); global_setting.constants.update({'p_l':LISTEN_PROBABILITY})
MAX_TIME_UNHEARD_ACCEPTABLE = int(args.max_time_unheard_acceptable); global_setting.constants.update({'max_time_unheard_acceptable':MAX_TIME_UNHEARD_ACCEPTABLE})
SAVE_TO_FILE_FREQUENCY = int(args.save_to_file_frequency); global_setting.constants.update({'save_to_file_frequency':SAVE_TO_FILE_FREQUENCY})
global_setting.constants.update({'time_slot_duration':int(args.time_slot_duration)})
NUM_REPEAT = int(args.num_repeat); global_setting.constants.update({'num_repeat':NUM_REPEAT})
OPTIMIZATION = args.optimization; global_setting.constants.update({'optimization':OPTIMIZATION})
PERIOD_OPTION = int(args.period_option); global_setting.constants.update({'period_option':PERIOD_OPTION})
ROLLING_AVERAGE_WINDOW = int(args.rolling_average_window); global_setting.constants.update({'rolling_average_window':ROLLING_AVERAGE_WINDOW})
problem_instance.initialize()    # retrieve the global variables

''' ____________________________________________________________________ setup and start the simulation ___________________________________________________________________ '''
startTime = time.time()

env = simpy.Environment()

if not os.path.exists(DIR): os.makedirs(DIR)                                     # create output directory if it doesn't exist

# get the number of networks and create the Network objects
networkDataRate = problem_instance.getNetworkDataRate(PROBLEM_INSTANCE, 1)  # print(t, currentDataRate)
NUM_NETWORK = len(networkDataRate)
networkList = [Network(0) for i in range(NUM_NETWORK)]                       # create network objects and store in networkList
global_setting.constants.update({'network_list':networkList})
from mobile_device import MobileDevice

print("going to start simulation...")
# create mobile device objects and store in mobileDeviceList
mobileDeviceList = [MobileDevice(networkList) for i in range(NUM_MOBILE_DEVICE)]

# each mobile device object executes the appropriate algorithm
for i in range(NUM_MOBILE_DEVICE):
    if ALGORITHM_NAME == "EXP3":
        algorithm = EXP3(NUM_NETWORK, i + 1)
    elif ALGORITHM_NAME == "FullInformation":
        algorithm = FullInformation(NUM_NETWORK, LEARNING_RATE, i + 1)
    elif ALGORITHM_NAME == "SmartEXP3":
        algorithm = SmartEXP3(NUM_NETWORK, seed=i + 1)
    elif ALGORITHM_NAME == "CoBandit":
        algorithm = CoBandit(NUM_NETWORK, LEARNING_RATE, MAX_TIME_UNHEARD_ACCEPTABLE, TRANSMIT_PROBABILITY, LISTEN_PROBABILITY, i + 1)
    elif ALGORITHM_NAME == "PeriodicEXP4":
        algorithm = PeriodicEXP4(NUM_NETWORK, NUM_TIME_SLOT, NUM_REPEAT, PROBLEM_INSTANCE, PERIOD_OPTION, mobileDeviceList[i].deviceID, OPTIMIZATION, i + 1)
    elif ALGORITHM_NAME == "SmartPeriodicEXP4":
        algorithm = SmartPeriodicEXP4(NUM_NETWORK, NUM_TIME_SLOT, NUM_REPEAT, PROBLEM_INSTANCE, PERIOD_OPTION, mobileDeviceList[i].deviceID, 0.1, OPTIMIZATION, 8, i + 1)
    elif ALGORITHM_NAME == "ContextualSmartEXP3":
        algorithm = ContextualSmartEXP3(NUM_NETWORK, seed=i + 1)
    proc = env.process(mobileDeviceList[i].performWirelessNetworkSelection(env, algorithm))

env.run(until=proc)  # SIM_TIME)

numTimeSlotPerRepetition = NUM_TIME_SLOT//NUM_REPEAT
print("----- going to compute distance to Nash equilibrium -----")
distanceToNE = computeDistanceToNashEquilibrium(PROBLEM_INSTANCE, NUM_TIME_SLOT, mobileDeviceList[0].networkCSVdata.rows, NUM_MOBILE_DEVICE)
saveToCSVfile(DIR + "distanceToNashEquilibrium.csv", [["timeslot", "distance"]] + [[timeIndex + 1, distanceToNE[timeIndex]] for timeIndex in range(len(distanceToNE))], "w")

print("----- going to compute average distance to Nash equilibrium per repetition -----")
meanDistancetoNEperRepetition = []
for repetition in range(1, NUM_REPEAT + 1): meanDistancetoNEperRepetition.append(sum(distanceToNE[(repetition - 1) * numTimeSlotPerRepetition:repetition * numTimeSlotPerRepetition])/numTimeSlotPerRepetition)
saveToCSVfile(DIR + "meanDistanceToNashEquilibriumPerRepetition.csv", [["repetition", "mean_distance"]] + [[repetition + 1, meanDistancetoNEperRepetition[repetition]] for repetition in range(NUM_REPEAT)], "w")

medianDistancetoNEperRepetition = []
for repetition in range(1, NUM_REPEAT + 1): medianDistancetoNEperRepetition.append(median(distanceToNE[(repetition - 1) * numTimeSlotPerRepetition:repetition * numTimeSlotPerRepetition]))
saveToCSVfile(DIR + "medianDistanceToNashEquilibriumPerRepetition.csv", [["repetition", "median_distance"]] + [[repetition + 1, medianDistancetoNEperRepetition[repetition]] for repetition in range(NUM_REPEAT)], "w")

print("----- going to save cumulative gain per device -----")
cumulativeDownloadPerDevice = []; cumulativeDownloadPerRepetitionPerDevice =[]
for i in range(NUM_REPEAT + 1): cumulativeDownloadPerRepetitionPerDevice.append([])  # last element stores list of cumulative download per device over the run
for i in range(NUM_MOBILE_DEVICE):  # for each device
    if ALGORITHM_NAME == "SmartPeriodicEXP4": index =13 + NUM_NETWORK
    elif ALGORITHM_NAME == "PeriodicEXP4": index = 7 + NUM_NETWORK
    elif ALGORITHM_NAME == "EXP3": index = 7 + (2 *NUM_NETWORK)
    elif ALGORITHM_NAME == "SmartEXP3": index = 10 + (2 * NUM_NETWORK)
    downloadPerDevice = [mobileDeviceList[i].deviceCSVdata.rows[j][index] for j in range(NUM_TIME_SLOT)] # get column download per time slot from csv file
    # compute cumulative download per repetition for the device
    cumulativeDownloadPerRepetition = []
    for repetition in range(1, NUM_REPEAT + 1): cumulativeDownloadPerRepetition.append(sum(downloadPerDevice[(repetition - 1) * numTimeSlotPerRepetition:repetition * numTimeSlotPerRepetition]))
    for repetition in range(NUM_REPEAT): cumulativeDownloadPerRepetitionPerDevice[repetition].append(cumulativeDownloadPerRepetition[repetition])
    cumulativeDownload = sum(downloadPerDevice); cumulativeDownloadPerDevice.append(cumulativeDownload)
    cumulativeDownloadPerRepetitionPerDevice[-1].append(cumulativeDownload) # last element stores list of cumulative download per device over the whole run
# compute and store the min, max, median, mean, stdev cumulative download for each repetition, and the aggregate per device for the whole run
saveToCSVfile(DIR + "cumulativeGainPerDevicePerRepetition.csv", [["repetition"] + ["device " + str(x) for x in range(1, NUM_MOBILE_DEVICE + 1)] + ["min", "max", "median", "mean", "std"]], "w")
for repetition in range(NUM_REPEAT + 1): # last list stores aggregate cumulative download per device for the whole run
    cumulativeDownloadPerRepetition = deepcopy(cumulativeDownloadPerRepetitionPerDevice[repetition])
    cumulativeDownloadPerRepetitionPerDevice[repetition].append(min(cumulativeDownloadPerRepetition))
    cumulativeDownloadPerRepetitionPerDevice[repetition].append(max(cumulativeDownloadPerRepetition))
    cumulativeDownloadPerRepetitionPerDevice[repetition].append(median(cumulativeDownloadPerRepetition))
    cumulativeDownloadPerRepetitionPerDevice[repetition].append(sum(cumulativeDownloadPerRepetition)/NUM_MOBILE_DEVICE)
    cumulativeDownloadPerRepetitionPerDevice[repetition].append(np.std(cumulativeDownloadPerRepetition, ddof=1))
    rep = str(repetition + 1) if repetition < NUM_REPEAT else "total"
    saveToCSVfile(DIR + "cumulativeGainPerDevicePerRepetition.csv", [[rep] + [x for x in cumulativeDownloadPerRepetitionPerDevice[repetition]]], "a")
# print("cumulativeDownloadPerDevice:", cumulativeDownloadPerDevice)
medianCumulativeDownload = median(cumulativeDownloadPerDevice); standardDeviationCumulativeDownload = np.std(cumulativeDownloadPerDevice, ddof=1)
saveToCSVfile(DIR + "cumulativeGainPerDevice.csv", [[x] for x in cumulativeDownloadPerDevice], "w")
saveToCSVfile(DIR + "cumulativeGainPerDevice_median_std.csv", [['median', medianCumulativeDownload], ['std', standardDeviationCumulativeDownload]], "w")

endTime = time.time()
timeTaken, unit = getTimeTaken(startTime, endTime)

print("----- simulation completed in %s %s -----" % (timeTaken, unit))

''' _____________________________________________________________________________ end of file _____________________________________________________________________________ '''