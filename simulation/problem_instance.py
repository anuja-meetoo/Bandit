'''
description: defines problem instances for simulation
'''

import global_setting
from copy import deepcopy
from math import ceil
from utility_method import computeNashEquilibriumState
import pickle

# this file is imported in wns.py before the values of the global parameters are set; their correct values are set in initialize()
NUM_MOBILE_DEVICE = NUM_TIME_SLOT = NUM_REPEAT = 0; PROBLEM_INSTANCES = {}; PERIOD_OPTIONS = {}
NETWORK_DATA_RATE = []

''' __________________________________________________________ definition of problem instances and period options _________________________________________________________ '''
def initialize():
    '''
    description: retrieves the values of the global parameters - they're are set after this file is imported; defines the problem instances and the period options (that will
                 define the partition functions); these problem instances may be repeated several times over the time horizon
    args:        none
    return:      None
    '''
    global NUM_MOBILE_DEVICE, NUM_TIME_SLOT, NUM_REPEAT, PROBLEM_INSTANCES, PERIOD_OPTIONS, NETWORK_DATA_RATE, NOISY_DATA_RATE

    NUM_MOBILE_DEVICE = global_setting.constants['num_mobile_device']
    NUM_TIME_SLOT = global_setting.constants['num_time_slot']
    NUM_REPEAT = global_setting.constants['num_repeat']

    ''' definition of problem instances '''

    ''' definition of period options '''
    PROBLEM_INSTANCES = {
        # number of devices, number of networks and data rates of networks remain unchanged
        'test': {
            0: {
                'data_rate': [4, 7, 22],
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi'],
                'device_list': [[1, 2], [1, 2], [1, 2]],
                'NEstate_list': [[0, 0, 2]]
            # [[0,1,4]]# #computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [4, 7, 22])
            }
            ,
            1.0 / 2: {
                'data_rate': [4, 7, 22],
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi'],
                'device_list': [[1], [1, 2], [2]],
                'NEstate_list': [[0, 1, 1]]
            }
        },  # end 'test'

        'static':{
            0: {
                'data_rate': [4, 7, 22],
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': [[2, 4, 14]] # [[0,1,4]]# #computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [4, 7, 22])
            }
        }, # end 'static'

        'change_in_data_rates': {
            0: {
                'data_rate': [7, 14, 44],
                'wireless_technology': ['WiFi', 'WiFi', 'Cellular'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [7, 14, 44])
            },
            1.0 / 2: {
                'data_rate': [36, 7, 22],
                'wireless_technology': ['WiFi', 'WiFi', 'Cellular'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [36, 7, 22])
            }
        }, # end 'change_in_data_rates'

        'change_in_data_rates_no_mobility': {
            0: {
                'data_rate': [7, 14, 44],
                'wireless_technology': ['WiFi', 'WiFi', 'Cellular'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': [[2, 4, 14]] #computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [7, 14, 44])
            },
            1.0 / 4: {
                'data_rate': [36, 7, 22],
                'wireless_technology': ['WiFi', 'WiFi', 'Cellular'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': [[11, 2, 7]] #computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [7, 14, 44])
            },
            2.0 / 4: {
                'data_rate': [9, 16, 40],
                'wireless_technology': ['WiFi', 'WiFi', 'Cellular'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': [[2, 5, 13]] #computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [7, 14, 44])
            },
            3.0 / 4: {
                'data_rate': [40, 4, 21],
                'wireless_technology': ['WiFi', 'WiFi', 'Cellular'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': [[13, 1, 6]] #computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [36, 7, 22])
            }
        },  # end 'change_in_data_rates_no_mobility'

        'noisy_change_in_data_rates_no_mobility': {
            0: {
                'data_rate': 'noisy_change_in_data_rates_no_mobility.pkl',
                'wireless_technology': ['WiFi', 'WiFi', 'Cellular'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)),
                                list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': [[2, 4, 14]]  # computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [7, 14, 44])
            },
            1.0 / 4: {
                'data_rate': 'noisy_change_in_data_rates_no_mobility.pkl',
                'wireless_technology': ['WiFi', 'WiFi', 'Cellular'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)),
                                list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': [[11, 2, 7]]  # computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [7, 14, 44])
            },
            2.0 / 4: {
                'data_rate': 'noisy_change_in_data_rates_no_mobility.pkl',
                'wireless_technology': ['WiFi', 'WiFi', 'Cellular'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)),
                                list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': [[2, 5, 13]]  # computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [7, 14, 44])
            },
            3.0 / 4: {
                'data_rate': 'noisy_change_in_data_rates_no_mobility.pkl',
                'wireless_technology': ['WiFi', 'WiFi', 'Cellular'],
                'device_list': [list(range(1, NUM_MOBILE_DEVICE + 1)), list(range(1, NUM_MOBILE_DEVICE + 1)),
                                list(range(1, NUM_MOBILE_DEVICE + 1))],
                'NEstate_list': [[13, 1, 6]]  # computeNashEquilibriumState(NUM_MOBILE_DEVICE, 3, [36, 7, 22])
            }
        },  # end 'noisy_change_in_data_rates_no_mobility'

        'foodcourt_studyarea_busstop': {
            0: {
                'data_rate': [16, 14, 22, 7, 4],
                'wireless_technology': ['Cellular', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, 21)), list(range(1, 11)), list(range(1, 16)), list(range(11, 21)), list(range(11, 21))],
                'NEstate_list': [[5, 5, 7, 2, 1]]
            },
            1.0 / 3: {
                'data_rate': [16, 14, 22, 7, 4],
                'wireless_technology': ['Cellular', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, 21)), [9, 10], list(range(1, 16)), list(range(1, 9)) + list(range(11, 21)), list(range(1, 9)) + list(range(11, 21))],
                'NEstate_list': [[6, 2, 9, 2, 1]]
            },
            2.0 / 3: {
                'data_rate': [16, 14, 22, 7, 4],
                'wireless_technology': ['Cellular', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, 21)), [9, 10], list(range(9, 16)), list(range(1, 9)) + list(range(11, 21)), list(range(1, 9)) + list(range(11, 21))],
                'NEstate_list': [[8, 2, 5, 3, 2]]
            }
        },

        # number of devices and number of networks remain unchanged, but data rates of networks change
        # 20 mobile users staying in a hostel; 15 work in the same office and travel together to work, in a bus that picks them from the hostel; 5 stay at home
        'mobility_setup_home_office': {
            0: { # at home - 13 hrs
                'data_rate': [16, 7, 44, 40, 14, 22, 7, 36, 18],
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, 21)), list(range(1, 21)), list(range(1, 11)), list(), list(), list(), list(), list(), list()],
                'NEstate_list': [[7, 3, 10, 0, 0, 0, 0, 0, 0]]
            },
            13/24: { # travelling to work - 1 hr
                'data_rate': [16, 7, 44, 40, 14, 22, 7, 36, 18],
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1,6)), list(range(1,6)), list(range(1,6)), list(range(6, 21)), list(range(6, 21)), list(), list(), list(), list()],
                'NEstate_list': [[1, 0, 4, 11, 4, 0, 0, 0, 0]]
            },
            14/24: { # in office - 3 hrs
                'data_rate': [16, 7, 44, 40, 14, 22, 7, 36, 18],
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1,6)), list(range(1,6)), list(range(1,6)), list(), list(), list(range(6, 21)), list(range(6, 21)), list(range(6, 21)), list()],
                'NEstate_list': [[1, 0, 4, 0, 0, 5, 1, 9, 0]]
            },
            17/24: { # lunch time - 1 hr
                'data_rate': [16, 7, 44, 40, 14, 22, 7, 36, 18],
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1,6)), list(range(1,6)), list(range(1,6)), list(), list(), list(range(6, 11)), list(range(6, 11)), list(range(6, 21)), list(range(11, 21))],
                'NEstate_list': [[1, 0, 4, 0, 0, 4, 1, 7, 3]]
            },
            18/24: { # in office - 5 hrs
                'data_rate': [16, 7, 44, 40, 14, 22, 7, 36, 18],
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1,6)), list(range(1,6)), list(range(1,6)), list(), list(), list(range(6, 21)), list(range(6, 21)), list(range(6, 21)), list()],
                'NEstate_list': [[1, 0, 4, 0, 0, 5, 1, 9, 0]]
            },
            23/24: { # travelling home - 1 hr
                'data_rate': [16, 7, 44, 40, 14, 22, 7, 36, 18],
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1,6)), list(range(1,6)), list(range(1,6)), list(range(6, 21)), list(range(6, 21)), list(), list(), list(), list()],
                'NEstate_list': [[1, 0, 4, 11, 4, 0, 0, 0, 0]]
            }
        }, # end 'mobility_setup_home_office'

        'noisy_mobility_setup_home_office': {
            0: {  # at home - 13 hrs
                'data_rate': 'noisy_mobility_setup_home_office.pkl',
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, 21)), list(range(1, 21)), list(range(1, 11)), list(), list(), list(), list(), list(), list()],
                'NEstate_list': [[7, 3, 10, 0, 0, 0, 0, 0, 0]]
            },
            13 / 24: {  # travelling to work - 1 hr
                'data_rate': 'noisy_mobility_setup_home_office.pkl',
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, 6)), list(range(1, 6)), list(range(1, 6)), list(range(6, 21)), list(range(6, 21)), list(), list(), list(), list()],
                'NEstate_list': [[1, 0, 4, 11, 4, 0, 0, 0, 0]]
            },
            14 / 24: {  # in office - 3 hrs
                'data_rate': 'noisy_mobility_setup_home_office.pkl',
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, 6)), list(range(1, 6)), list(range(1, 6)), list(), list(), list(range(6, 21)), list(range(6, 21)), list(range(6, 21)), list()],
                'NEstate_list': [[1, 0, 4, 0, 0, 5, 1, 9, 0]]
            },
            17 / 24: {  # lunch time - 1 hr
                'data_rate': 'noisy_mobility_setup_home_office.pkl',
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, 6)), list(range(1, 6)), list(range(1, 6)), list(), list(), list(range(6, 11)), list(range(6, 11)), list(range(6, 21)), list(range(11, 21))],
                'NEstate_list': [[1, 0, 4, 0, 0, 4, 1, 7, 3]]
            },
            18 / 24: {  # in office - 5 hrs
                'data_rate': 'noisy_mobility_setup_home_office.pkl',
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, 6)), list(range(1, 6)), list(range(1, 6)), list(), list(), list(range(6, 21)), list(range(6, 21)), list(range(6, 21)), list()],
                'NEstate_list': [[1, 0, 4, 0, 0, 5, 1, 9, 0]]
            },
            23 / 24: {  # travelling home - 1 hr
                'data_rate': 'noisy_mobility_setup_home_office.pkl',
                'wireless_technology': ['WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi', 'WiFi'],
                'device_list': [list(range(1, 6)), list(range(1, 6)), list(range(1, 6)), list(range(6, 21)), list(range(6, 21)), list(), list(), list(), list()],
                'NEstate_list': [[1, 0, 4, 11, 4, 0, 0, 0, 0]]
            }
        },  # end 'noisy_mobility_setup_home_office'
    } # end of PROBLEM_INSTANCES

    PERIOD_OPTIONS = {
        1: [1],                                             # EXP3
        2: [NUM_REPEAT],                                    # EXP3, but reset every day
        3: list(range(1, 16)),                              # 1 - 15
        4: list(range(1, 40)),                              # 1 - 39
        5: [1, 2, 3, 5, 7, 11 ,13],                         # primes below 15
        6: [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37], # primes below 40
        7: [1, 2, 4, 8, 16],                                # powers of 2 up to 16
        8: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],   # powers of 2 up to 1024
        9: list(range(100)),                                # all numbers up to 100
        10: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97],# primes below 100
        11: [53, 59, 61, 67, 71, 73, 79, 83, 89, 97],       # large primes only
        12: [24],
        13: list(range(1,25)),
        14: [96],
        15: list(range(1,5)),
        16: [1, 2],
        17: [4]
    } # end PERIOD_OPTIONS

    with open('noisy_mobility_setup_home_office.pkl', 'rb') as f: noisy_mobility_setup_home_office = pickle.load(f)
    with open('noisy_change_in_data_rates_no_mobility.pkl', 'rb') as f: noisy_change_in_data_rates_no_mobility = pickle.load(f)

    NOISY_DATA_RATE = {
        'noisy_mobility_setup_home_office': { 'data_rate': noisy_mobility_setup_home_office },
        'noisy_change_in_data_rates_no_mobility': {'data_rate': noisy_change_in_data_rates_no_mobility }
        # 'change_in_data_rates_no_mobility_drop':
        #     { 'data_rate': change_in_data_rates_no_mobility_drop, 'NEstate_list': change_in_data_rates_no_mobility_drop_NE }
    }

    # end initialize

''' ___________________________________________________________ functions to extract details of problem instance __________________________________________________________ '''
def mapKeyToTimeSlot(key):
    '''
    description: maps a key (that represents time of a change in the environment) from the problem instance definition to an actual time slot (in the first repetition)
    args:        a key in a problem instance definition
    return:      a time slot
    '''
    global NUM_TIME_SLOT, NUM_REPEAT

    return key*(NUM_TIME_SLOT//NUM_REPEAT) + 1
    # end mapKeyToTimeSlot

def getTimeOfChangeInEnvironment(problem_instance):
    '''
    description: returns the time slots at which there is a change in the environment (considering repetitions - list of time slots till end of simulation run)
    args:        name of problem instance being considered
    return:      the time slots at which there are changes
    '''
    global PROBLEM_INSTANCES, NUM_REPEAT, NUM_TIME_SLOT

    timeOfChange = []

    # get the list of keys in ascending order for the problem instance considered
    key_list = PROBLEM_INSTANCES[problem_instance].keys(); key_list = sorted(key_list)

    for key in key_list: timeOfChange += [int(mapKeyToTimeSlot(key)) + (repetition * (NUM_TIME_SLOT//NUM_REPEAT)) for repetition in range(NUM_REPEAT)]
    return sorted(timeOfChange)
    # end getTimesOfChange


def mapTimeSlotToKey(problem_instance, timeSlot):
    '''
    description: maps an actual time slot to a key (that represents time of a change in the environment) in the problem instance definition
    args:        a time slot
    return:      a key in a problem instance definition that defines the current state of the environment
    '''
    global NUM_TIME_SLOT, NUM_REPEAT

    # get the list of keys in ascending order for the problem instance considered, and map them to a time slot in the first repetition
    key_list = PROBLEM_INSTANCES[problem_instance].keys(); key_list = sorted(key_list)
    timeOfChange = getTimeOfChangeInEnvironment(problem_instance)

    # map the current time slot to a time slot in the first repetition - faster to search for the key
    equivalentTimeSlotInFirstRepetition = NUM_TIME_SLOT//NUM_REPEAT if timeSlot%(NUM_TIME_SLOT//NUM_REPEAT) == 0 else timeSlot%(NUM_TIME_SLOT//NUM_REPEAT)
    # print("timeOfChange:", timeOfChange, ", time:", timeSlot, ", equivalentTimeSlotInFirstRepetition:", equivalentTimeSlotInFirstRepetition)
    # print("timeOfChange:", timeOfChange, ", t:", timeSlot, ", equivalentTimeSlotInFirstRepetition:", equivalentTimeSlotInFirstRepetition)

    # print(key_list, timeOfChange, equivalentTimeSlotInFirstRepetition)
    if len(timeOfChange) == 1: return key_list[0]
    if equivalentTimeSlotInFirstRepetition >= timeOfChange[-1]: return key_list[-1]
    for index, time in enumerate(timeOfChange):
        if time == equivalentTimeSlotInFirstRepetition: return key_list[index]
        if time > equivalentTimeSlotInFirstRepetition: return key_list[index - 1]
    # end mapTimeSlotToKey

def changeInEnvironment(problem_instance, timeSlot):
    '''
    description: determines if there is a change in the current time slot
    args:        the name of the problem instance being considered, the current time slot
    return:      True or False depending on whether there is a change in the current time slot
    '''
    global PROBLEM_INSTANCES

    return timeSlot in getTimeOfChangeInEnvironment(problem_instance)
    # end changeInEnvironment

def changeInNetworkAvailability(problem_instance, timeSlot, deviceID):
    '''
    decsription: determines if there is a change in network availability in the current time slot for a particular device
    args:        the name of the problem instance being considered, the current time slot, ID of a mobile device
    return:      True or False depending on whether there is a change in availability of networks in the current time slot for device deviceID, set of networks
                 available in the current and previous time slot
    '''
    global PROBLEM_INSTANCES

    equivalentTimeSlotInFirstRepetition = NUM_TIME_SLOT // NUM_REPEAT if timeSlot % (NUM_TIME_SLOT // NUM_REPEAT) == 0 else timeSlot % (NUM_TIME_SLOT // NUM_REPEAT)

    timeOfChangeInNetworkAvailabilityList = getTimeOfChangeInNetworkAvailability(problem_instance, deviceID)
    change = False if timeSlot > 1 and len(timeOfChangeInNetworkAvailabilityList) == 1 else equivalentTimeSlotInFirstRepetition in timeOfChangeInNetworkAvailabilityList
    return change, getAvailableNetwork(problem_instance, equivalentTimeSlotInFirstRepetition, deviceID)
    # end changeInNetworkAvailability

def getNetworkDataRate(problem_instance, timeSlot):
    '''
    description: gets the data rate of each network for the current time slot, based on the definition of the problem instance being considered
    args:        the name of the problem instance being considered, the current time slot
    return:      data rates for each network for the current time slot
    '''
    global PROBLEM_INSTANCES

    key = mapTimeSlotToKey(problem_instance, timeSlot)
    dataRate = PROBLEM_INSTANCES[problem_instance][key]['data_rate']
    if '.pkl' in dataRate: dataRate = NOISY_DATA_RATE[problem_instance]['data_rate'][timeSlot - 1]
    return dataRate
    # end getNetworkDataRate

def getWirelessTechnology(problem_instance, timeSlot):
    '''
    description: gets the wireless technology of each network available during the current time slot, based on the definition of the problem instance being considered
    args:        the name of the problem instance being considered, the current time slot
    return:      wireless technology of each network available during the current time slot
    '''
    global PROBLEM_INSTANCES

    key = mapTimeSlotToKey(problem_instance, timeSlot)
    return PROBLEM_INSTANCES[problem_instance][key]['wireless_technology']
    # end getWirelessTechnology

def getDeviceListPerNetwork(problem_instance, timeSlot):
    '''
    description: gets the list of devices that have access to each network
    args:        the name of the problem instance being considered, the current time slot
    return:      list of devices that have access to each network
    '''
    global PROBLEM_INSTANCES

    key = mapTimeSlotToKey(problem_instance, timeSlot)
    return PROBLEM_INSTANCES[problem_instance][key]['device_list']
    # end getDeviceListPerNetwork

def getNashEquilibriumState(problem_instance, timeSlot):
    '''
    description: gets the list Nash equilibrium state(s) for the current time slot, based on the definition of the problem instance being considered
    args:        the name of the problem instance being considered, the current time slot
    return:      the list Nash equilibrium state(s) for the current time slot
    '''
    global PROBLEM_INSTANCES

    key = mapTimeSlotToKey(problem_instance, timeSlot)
    NEstate_list = PROBLEM_INSTANCES[problem_instance][key]['NEstate_list']
    if '.pkl' in NEstate_list: NEstate_list = NOISY_DATA_RATE[problem_instance]['NEstate_list'][timeSlot - 1]
    # print(NEstate_list)
    return NEstate_list
    #PROBLEM_INSTANCES[problem_instance][key]['NEstate_list']
    # end getNashEquilibriumState

def isActive(problem_instance, timeSlot, deviceID):
    '''
    description: determines whether the device is active in the current time slot; i.e., selecting and associating with a network
    args:        the name of the problem instance being considered, the current time slot, ID of a mobile device
    return:      True or False depending on whether the device is active in the current time slot
    '''
    global PROBLEM_INSTANCES

    key = mapTimeSlotToKey(problem_instance, timeSlot)
    deviceList = PROBLEM_INSTANCES[problem_instance][key]['device_list']
    for list in deviceList:
        if deviceID in list: return True
    return False
    # end isActive

def getUnavailableNetwork(problem_instance, timeSlot, deviceID):
    '''
    description: returns the list of networks currently unavailable to a device
    args:        name of the problem instance being considered, the current time slot, ID of a mobile device
    return:      list of networks currently not available (their IDs)
    '''
    global PROBLEM_INSTANCES

    unavailableNetworkList = []
    key = mapTimeSlotToKey(problem_instance, timeSlot)
    deviceList = PROBLEM_INSTANCES[problem_instance][key]['device_list']
    for index, list in enumerate(deviceList):
        if deviceID not in list: unavailableNetworkList.append(index + 1)
    return unavailableNetworkList

def getAvailableNetwork(problem_instance, timeSlot, deviceID):
    '''
    description: returns the list of networks currently available to a device
    args:        name of the problem instance being considered, the current time slot, ID of a mobile device
    return:      list of networks currently not available (their IDs)
    '''
    global PROBLEM_INSTANCES

    availableNetworkList = []
    key = mapTimeSlotToKey(problem_instance, timeSlot)
    deviceList = PROBLEM_INSTANCES[problem_instance][key]['device_list']
    for index, list in enumerate(deviceList):
        if deviceID in list: availableNetworkList.append(index + 1)
    return availableNetworkList
    # end getAvailableNetwork

def isNetworkAccessible(problem_instance, timeSlot, networkID, deviceID):
    '''
    description: determines whether a network is available to a device
    args:        the name of the problem instance being considered, the current time slot, ID of a mobile device
    return:      True or False depending on whether the network is available to the device
    '''
    global PROBLEM_INSTANCES

    key = mapTimeSlotToKey(problem_instance, timeSlot)
    # print('timeSlot:', timeSlot, 'key', key)
    deviceList = PROBLEM_INSTANCES[problem_instance][key]['device_list']
    if deviceID in deviceList[networkID - 1]: return True
    return False
    # end isNetworkAccessible

def getTimeOfChangeInNetworkAvailability(problem_instance, deviceID):
    '''
    description: identifies the time slots at which there is a change in the set of networks available to a particular device - considers repetition; till end of simulatiton run
    args:        the name of the problem instance being considered, the current time slot, ID of a mobile device
    return:      the time slots at which there are is a change in network availability
    '''
    global NUM_TIME_SLOT, NUM_REPEAT

    timeOfChangeInNetworkAvailability = []
    previousAvailableNetwork = []

    # get the list of keys in ascending order for the problem instance considered
    timeOfChangeInEnvironment = getTimeOfChangeInEnvironment(problem_instance)
    timeOfChangeInEnvironment = sorted(i for i in timeOfChangeInEnvironment if i <= NUM_TIME_SLOT)

    for time in timeOfChangeInEnvironment:
        availableNetwork = getAvailableNetwork(problem_instance, time, deviceID)
        if sorted(availableNetwork) != sorted(previousAvailableNetwork): timeOfChangeInNetworkAvailability.append((time))
        previousAvailableNetwork = deepcopy(availableNetwork)

    # print("timeOfChangeInNetworkAvailability:", timeOfChangeInNetworkAvailability)
    return timeOfChangeInNetworkAvailability

    # end getTimeOfChangeInNetworkAvailability

''' _______________________________________________________________ main program to test the above functions ______________________________________________________________ '''
def main():
    problem_instance = "foodcourt_studyarea_busstop"#'mobility_setup_home_office'
    global NUM_MOBILE_DEVICE, NUM_TIME_SLOT, NUM_REPEAT, PROBLEM_INSTANCES, PERIOD_OPTIONS

    global_setting.constants.update({'num_mobile_device':6})
    global_setting.constants.update({'num_time_slot':48})
    global_setting.constants.update({'num_repeat': 2})

    initialize()
    networkID = 2
    deviceID = 6
    # print(getTimeOfChangeInEnvironment(problem_instance))
    # # for k in PROBLEM_INSTANCES[problem_instance]: print(k, mapKeyToTimeSlot(k))
    # for t in range(1, NUM_TIME_SLOT + 1):
    #     key = mapTimeSlotToKey(problem_instance, t)
    #     print("t = ",t,", key:", key)
    # print(42, 1, isActive(problem_instance, 42, 1))
    # print(getUnavailableNetwork(problem_instance, 47, 10))
    # print(isNetworkAccessible(problem_instance, 47, 6, 10))
    print("TimeOfChangeInEnvironment:", getTimeOfChangeInEnvironment(problem_instance))
    print("TimeOfChangeInNetworkAvailability: ", getTimeOfChangeInNetworkAvailability(problem_instance, deviceID))
    for time in getTimeOfChangeInNetworkAvailability(problem_instance, deviceID):
        print("time: ", time)
        print("available network:", getAvailableNetwork(problem_instance, time, deviceID))
        print("unavailable network:", getUnavailableNetwork(problem_instance, time, deviceID))

if __name__ == '__main__': main()
''' _____________________________________________________________________________ end of file _____________________________________________________________________________ '''