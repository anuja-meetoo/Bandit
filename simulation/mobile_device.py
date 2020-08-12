'''
@description:   Defines a class that models a mobile device
'''

''' ______________________________________________________________________ import external libraries ______________________________________________________________________ '''
from scipy.stats import t, johnsonsu
from copy import deepcopy
from utility_method import saveToCSVfile, getListIndex, CSVdata
import global_setting
import problem_instance

''' _____________________________________________________________________________ for logging _____________________________________________________________________________ '''
import logging
from colorlog import ColoredFormatter   # install using sudo pip3 install colorlog
import numpy as np

LOG_LEVEL = logging.DEBUG
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(
    "  %(log_color)s%(levelname)-8s%(reset)s %(log_color)s%(message)s%(reset)s",
	datefmt=None,
	reset=True,
	log_colors={
		'DEBUG':    'cyan',
		'INFO':     'green',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'white,bg_red',
	},
	secondary_log_colors={},
	style='%'
)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logging = logging.getLogger('pythonConfig')
logging.setLevel(LOG_LEVEL)
logging.addHandler(stream)

''' ______________________________________________________________________________ constants ______________________________________________________________________________ '''
TIME_SLOT_DURATION = global_setting.constants['time_slot_duration'] # duration of a time step in seconds
NUM_MOBILE_DEVICE = global_setting.constants['num_mobile_device']
# NETWORK_BANDWIDTH = global_setting.constants['network_bandwidth'] # in Mbps
NUM_TIME_SLOT = global_setting.constants['num_time_slot']
RUN_NUM = global_setting.constants['run_num']
ALGORITHM = global_setting.constants['algorithm_name']
ORIGINAL_OUTPUT_DIR = OUTPUT_DIR = global_setting.constants['output_dir']
networkList = global_setting.constants['network_list']
SAVE_TO_FILE_FREQUENCY = global_setting.constants['save_to_file_frequency']   # 1 means every time slot, 10 means every 10 time slots...
PROBLEM_INSTANCE = global_setting.constants['problem_instance']
NUM_REPEAT = global_setting.constants['num_repeat']

''' ____________________________________________________________________ MobileDevice class definition ____________________________________________________________________ '''
class MobileDevice(object):
    numMobileDevice = 0                                     # keeps track of number of mobile devices to automatically assign an ID to device upon creation
    sharedObservation = {}                                  # observations about networks shared among devices; there may be more than one service area
    resetTimeSlotPerDevice = {}

    def __init__(self, networks):
        MobileDevice.numMobileDevice = MobileDevice.numMobileDevice + 1
        self.deviceID = MobileDevice.numMobileDevice        # ID of device
        self.availableNetwork = [networks[i].networkID for i in range(len(networks))]  # networkIDs of set of available networks
        self.currentNetwork = -1                            # network to which the device is currently associated
        self.gain = 0                                       # bit rate observed (ignores switching cost)
        self.download = 0                                   # amount of data downloaded in Mbits (takes into account switching cost)
        self.maxGain = 0#max([NETWORK_BANDWIDTH[i - 1] for i in self.availableNetwork])
        self.delay = 0                                      # delay incurred while switching network in seconds

        # attribute for log
        self.deviceCSVdata = None
        self.networkCSVdata = None
        # end __init__

    ''' ################################################################################################################################################################### '''
    def performWirelessNetworkSelection(self, env, algorithm):
        '''
        description: repeatedly performs a wireless network selection, following the a particular bandit-style algorithm
        args:        self, env
        returns:     None
        '''
        global NUM_TIME_SLOT, ORIGINAL_OUTPUT_DIR, RUN_NUM, NUM_MOBILE_DEVICE, PROBLEM_INSTANCE, NUM_TIME_SLOT, NUM_REPEAT
        collaboration = True if ALGORITHM == "CoBandit" else False

        # create csv files to store per time slot details
        MobileDevice.createDeviceCSVfile(self, algorithm)
        if self.deviceID == 1: MobileDevice.createNetworkCSVfile(self)
        currentNetworkAvailabilityStatus = [1] * len(self.availableNetwork)

        for t in range(1, NUM_TIME_SLOT + 1):
            # update changes in network data rate
            changeInNetworkAvailability = False

            if 'noisy' in PROBLEM_INSTANCE:
                currentDataRate = problem_instance.getNetworkDataRate(PROBLEM_INSTANCE, t); self.maxGain = max(currentDataRate); wirelessTechnology = problem_instance.getWirelessTechnology(PROBLEM_INSTANCE, t)
                if self.deviceID == 1:
                    for i in range(len(networkList)): networkList[i].setDataRate(currentDataRate[i]); networkList[i].setWirelessTechnology(wirelessTechnology[i])

            if problem_instance.changeInEnvironment(PROBLEM_INSTANCE, t):
                if self.deviceID == 1: print("@t = ", t, " - change in network data rate"); #input()

                if 'noisy' not in PROBLEM_INSTANCE:
                    currentDataRate = problem_instance.getNetworkDataRate(PROBLEM_INSTANCE, t); self.maxGain = max(currentDataRate); wirelessTechnology = problem_instance.getWirelessTechnology(PROBLEM_INSTANCE, t)
                    if self.deviceID == 1:
                        for i in range(len(networkList)): networkList[i].setDataRate(currentDataRate[i]); networkList[i].setWirelessTechnology(wirelessTechnology[i])

                changeInNetworkAvailability, currentAvailableNetwork = problem_instance.changeInNetworkAvailability(PROBLEM_INSTANCE, t, self.deviceID)
                currentNetworkAvailabilityStatus = [0] * len(self.availableNetwork)
                for availableNetwork in currentAvailableNetwork: currentNetworkAvailabilityStatus[self.availableNetwork.index(availableNetwork)] = 1
                if changeInNetworkAvailability: self.maxGain = max([network.getDataRate() for network in networkList if network.getID() in currentAvailableNetwork])

            yield env.timeout(1)

            # update probability distribution
            if ALGORITHM == "SmartEXP3" or ALGORITHM == "EXP3": algorithm.updateProbabilityDistribution(t, changeInNetworkAvailability, currentNetworkAvailabilityStatus, self.availableNetwork.index(self.currentNetwork) if self.currentNetwork != -1 else -1, self.deviceID)
            elif ALGORITHM == "ContextualSmartEXP3": algorithm.updateProbabilityDistribution(t, changeInNetworkAvailability, currentNetworkAvailabilityStatus, self.availableNetwork.index(self.currentNetwork) if self.currentNetwork != -1 else -1, NUM_TIME_SLOT, NUM_REPEAT, self.deviceID)
            elif ALGORITHM == "PeriodicEXP4" or ALGORITHM == "SmartPeriodicEXP4": algorithm.updateProbabilityDistribution(t, changeInNetworkAvailability, currentNetworkAvailabilityStatus, self.availableNetwork.index(self.currentNetwork) if self.currentNetwork != -1 else -1, self.deviceID)
            else: algorithm.updateProbabilityDistribution(t, self.deviceID)

            # select wireless network
            prevNetworkSelected = self.currentNetwork; self.currentNetwork = self.availableNetwork[algorithm.chooseAction(t, NUM_MOBILE_DEVICE, self.availableNetwork.index(self.currentNetwork) if self.currentNetwork != -1 else -1, self.deviceID)]

            # associate with wireless network
            MobileDevice.associateWithWirelessNetwork(self, prevNetworkSelected)

            yield env.timeout(1)

            # observe gain
            MobileDevice.observeGain(self, t)

            # collaborate
            # if collaboration == True: transmit = algorithm.transmit(); yield env.timeout(1); algorithm.listen(transmit)
            # else: yield env.timeout(1)

            # save details of the run
            MobileDevice.saveDeviceDetail(self, t, algorithm, currentNetworkAvailabilityStatus)
            if self.deviceID == 1: MobileDevice.saveNetworkDetail(self, t)
            MobileDevice.writeCSVfile(self, t) # save details to csv file

            # update weight
            if ALGORITHM == "FullInformation": scaledGainPerNetwork = MobileDevice.computeGainPerNetwork(self, t); algorithm.updateWeight(scaledGainPerNetwork)
            else: algorithm.updateWeight(t, self.availableNetwork.index(self.currentNetwork), self.gain, self.maxGain, prevNetworkSelected, self.deviceID)

            yield env.timeout(1)
        # end performWirelessNetworkSelection

    ''' ################################################################################################################################################################### '''
    def associateWithWirelessNetwork(self, prevNetworkSelected):
        if prevNetworkSelected != self.currentNetwork:
            if prevNetworkSelected != -1: MobileDevice.leaveNetwork(self, prevNetworkSelected)
            MobileDevice.joinNetwork(self, self.currentNetwork);
            self.delay = MobileDevice.computeDelay(self)
        else: self.delay = 0
        # end associateWithWirelessNetwork

    ''' ################################################################################################################################################################### '''
    def joinNetwork(self, networkSelected):
        '''
        description: adds a particular device to a specified network, by incrementing the number of devices in that network by 1
        arg:         self, ID of network to join
        returns:     None
        '''
        global networkList

        networkIndex = getListIndex(networkList, networkSelected)
        networkList[networkIndex].associateDevice(self.deviceID)
        # end joinNetwork

    ''' ################################################################################################################################################################### '''
    def leaveNetwork(self, prevNetworkSelected):
        '''
        description: removes a particular device from a specified network, by decrementing the number of devices in that network by 1
        arg:         self, ID of network to leave
        returns:   None
        '''
        global networkList

        networkIndex = getListIndex(networkList, prevNetworkSelected)
        networkList[networkIndex].disassociateDevice(self.deviceID)
        # end leaveNetwork

    ''' ################################################################################################################################################################### '''
    def computeDelay(self):
        '''
        description: generates a delay for switching between WiFi networks, which is modeled using Johnson’s SU distribution (identified as a best fit to 500 delay values),
                     and delay for switching between WiFi and cellular networks, modeled using Student's t-distribution (identified as best fit to 500 delay values)
        args:        self
        returns:     a delay value
        '''
        wifiDelay = [3.0659475327, 14.6918344498]       # min and max delay observed for wifi in some real experiments; used as caps for the delay generated
        cellularDelay = [4.2531193161, 14.3172883892]   # min and max delay observed for 3G in some real experiments; used as caps for the delay generated

        if networkList[getListIndex(networkList, self.currentNetwork)].getWirelessTechnology() == 'WiFi':
            delay = min(max(johnsonsu.rvs(0.29822254217554717, 0.71688524931466857, loc=6.6093350624107909, scale=0.5595970482712973), wifiDelay[0]), wifiDelay[1])
        else:
            delay = min(max(t.rvs(0.43925241212097499, loc=4.4877772816533934, scale=0.024357324434644639), cellularDelay[0]), cellularDelay[1])
        return delay
        # end computeDelay

    ''' ################################################################################################################################################################### '''
    def observeGain(self, currentTimeSlot):
        '''
        description: determines the bit rate observed by the device from the wireless network selected and scale the gain to the range [0, 1]
        args:        self
        returns:     amount of bandwidth observed by the device
        '''
        global networkList, TIME_SLOT_DURATION, PROBLEM_INSTANCE

        networkIndex = getListIndex(networkList, self.currentNetwork)   # get the index in lists where details of the specific network is saved
        self.gain = networkList[networkIndex].getPerDeviceBitRate(PROBLEM_INSTANCE, currentTimeSlot, self.deviceID)     # in Mbps
        self.download = networkList[networkIndex].getPerDeviceDownload(PROBLEM_INSTANCE, currentTimeSlot, self.deviceID, TIME_SLOT_DURATION, self.delay)  # Mbits
        # end observeGain

    ''' ################################################################################################################################################################### '''
    def computeGainPerNetwork(self, currentTimeSlot):
        '''
        description: computes the gain of each network; assuming availability of full information
        args:        self
        re'urn:      gain of each network
        '''
        global networkList, PROBLEM_INSTANCE

        scaledGainPerNetwork = [0] * len(self.availableNetwork)

        for i in range(len(self.availableNetwork)):
            networkIndex = getListIndex(networkList, self.availableNetwork[i])
            if not problem_instance.isNetworkAccessible(PROBLEM_INSTANCE, currentTimeSlot, self.availableNetwork[i], self.deviceID): scaledGainPerNetwork[i] = 0
            elif networkList[networkIndex].networkID == self.currentNetwork: scaledGainPerNetwork[i] = self.gain / self.maxGain
            else: scaledGainPerNetwork[i] = (networkList[networkIndex].dataRate / (networkList[networkIndex].getNumAssociatedDevice() + 1)) / self.maxGain
        return scaledGainPerNetwork
        # end computeGainPerNetwork

    ''' ################################################################################################################################################################### '''
    def createDeviceCSVfile(self, algorithm):
        '''
        description: creates CSV file(s) to save per time slot details about the algorithm run by the device
        args:        self, algorithm being used (an object)
        return:      None
        '''
        global ORIGINAL_OUTPUT_DIR

        # create csv to save details of the device
        algorithmHeader = ["run", "timeslot", "deviceID"] + algorithm.getAttributeName() + ["current network", "gain (Mbps)", "delay (secs)", "download (Mbits)"]
        for i in range(1, len(self.availableNetwork) + 1): algorithmHeader.append("download %d (Mbits)" %(i))
        self.deviceCSVdata = CSVdata(ORIGINAL_OUTPUT_DIR + "device%d.csv" %(self.deviceID), algorithmHeader)
        # end createDeviceCSVfile

    ''' ################################################################################################################################################################### '''
    def createNetworkCSVfile(self):
        '''
        description: creates CSV file(s) to save per time slot details about devices associated to each network
        args:        self
        return:      None
        '''
        global ORIGINAL_OUTPUT_DIR

        # create csv file to save details for network
        networkHeader = ["run", "timeslot"]
        for i in range(1, len(self.availableNetwork) + 1): networkHeader.append("data rate %d" % (i))
        for i in range(1, len(self.availableNetwork) + 1): networkHeader.append("technology %d" % (i))
        for i in range(1, len(self.availableNetwork) + 1): networkHeader.append("#devices %d" % (i))
        for i in range(1, len(self.availableNetwork) + 1): networkHeader.append("device list %d" % (i))
        self.networkCSVdata = CSVdata(ORIGINAL_OUTPUT_DIR + "network.csv", networkHeader)
        # end createNetworkCSVfile

    ''' ################################################################################################################################################################### '''
    def saveDeviceDetail(self, t, algorithm, currentNetworkAvailabilityStatus):
        '''
        description: save run time details about the algorithm run by the device
        args:        self, algorithm being used (an object), and current time slot
        return:      None
        '''
        global ORIGINAL_OUTPUT_DIR, RUN_NUM, SAVE_TO_FILE_FREQUENCY, TIME_SLOT_DURATION, NUM_TIME_SLOT

        algorithmData = [RUN_NUM, t, self.deviceID] + algorithm.getAttributeValue() + [self.currentNetwork, self.gain, self.delay, self.download]
        # build the download from each network (had the device chosen it)
        for i in range(len(self.availableNetwork)):
            networkIndexInNetworkList = getListIndex(networkList, self.availableNetwork[i])
            if self.availableNetwork[i] == self.currentNetwork:
                algorithmData.append((networkList[networkIndexInNetworkList].getDataRate()/networkList[networkIndexInNetworkList].getNumAssociatedDevice()) * TIME_SLOT_DURATION)
            elif currentNetworkAvailabilityStatus[i] == 1:
                algorithmData.append((networkList[networkIndexInNetworkList].getDataRate()/(networkList[networkIndexInNetworkList].getNumAssociatedDevice()+1)) * TIME_SLOT_DURATION)
            else: algorithmData.append(0)   # the network is not currently available to the device
            # else: algorithmData.append((networkList[networkIndexInNetworkList].getDataRate()/(networkList[networkIndexInNetworkList].getNumAssociatedDevice()+1)) * TIME_SLOT_DURATION)
        self.deviceCSVdata.addRow(algorithmData)
        # end saveDeviceDetail

    ''' ################################################################################################################################################################### '''
    def saveNetworkDetail(self, t):
        '''
        description: save run time details about devices associated to each network
        args:        self, current time slot
        return:      None
        '''
        global ORIGINAL_OUTPUT_DIR, RUN_NUM, SAVE_TO_FILE_FREQUENCY, TIME_SLOT_DURATION, NUM_TIME_SLOT

        networkData = [RUN_NUM, t]
        for i in range(len(networkList)): networkData.append(networkList[i].getDataRate())
        for i in range(len(networkList)): networkData.append(networkList[i].getWirelessTechnology())
        for i in range(len(networkList)): networkData.append(networkList[i].getNumAssociatedDevice())
        for i in range(len(networkList)): networkData.append(deepcopy(networkList[i].getAssociatedDevice()))
        self.networkCSVdata.addRow(networkData)
        # end saveNetworkDetail

    ''' ################################################################################################################################################################### '''
    def writeCSVfile(self, timeSlot):
        '''
        description: saves details to csv files
        args:        self, the current times slot
        return:      None
        '''
        global SAVE_TO_FILE_FREQUENCY

        # print("t", timeSlot, ", freq:", SAVE_TO_FILE_FREQUENCY)
        if timeSlot % SAVE_TO_FILE_FREQUENCY == 0:
            # print("----- going to write CSV files -----")
            if self.deviceCSVdata != None: self.deviceCSVdata.saveToFile(SAVE_TO_FILE_FREQUENCY)
            if self.deviceID == 1 and self.networkCSVdata != None: self.networkCSVdata.saveToFile(SAVE_TO_FILE_FREQUENCY)
    # end MobileDevice class