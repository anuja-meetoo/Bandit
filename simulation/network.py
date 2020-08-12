'''
@description:   Defines a class that models a wireless network
'''

import problem_instance

''' _______________________________________________________________________ Network class definition ______________________________________________________________________ '''
class Network(object):
    ''' class to represent network objects '''
    numNetwork = 0  # keeps track of number of networks to automatically assign an ID to network upon creation

    def __init__(self, dataRate):
        Network.numNetwork = Network.numNetwork + 1     # increment number of network objects created
        self.networkID = Network.numNetwork             # ID of network
        self.dataRate = dataRate                        # date rate of network (in Mbps)
        self.wirelessTechnology = None                  # e.g. WiFi or 3G
        self.associatedDevice = set()                   # set of associated devices
        # end __init__

    ''' ################################################################################################################################################################### '''
    def associateDevice(self, deviceID):
        '''
        description: increments the number of devices connected to the network
        arg:         self
        returns:     None
        '''
        self.associatedDevice.add(deviceID)
        # end associateDevice

    ''' ################################################################################################################################################################### '''
    def disassociateDevice(self, deviceID):
        '''
        description: decrements the number of devices connected to the network
        arg:         self
        returns:     None
        '''
        self.associatedDevice.remove(deviceID)
        # end disassociateDevice

    ''' ################################################################################################################################################################### '''
    def getID(self):
        '''
        description: returns the ID of the network
        args:        self
        return:      ID of the network
        '''
        return self.networkID

    ''' ################################################################################################################################################################### '''
    def getDataRate(self):
        '''
        description: returns the data rate of the network
        args:        self
        return:      data rate of the network
        '''
        return self.dataRate

    ''' ################################################################################################################################################################### '''
    def setDataRate(self, rate):
        '''
        description: updates the data rate of the network
        args:        self, the data rate
        return:      None
        '''
        self.dataRate = rate
        # end setDataRate

    ''' ################################################################################################################################################################### '''
    def getWirelessTechnology(self):
        '''
        description: returns the wireless technology of the networks
        args:        self
        return:      wireless technology of the network
        '''
        return self.wirelessTechnology
        # end getWirelessTechnology

    ''' ################################################################################################################################################################### '''
    def setWirelessTechnology(self, wirelessTechnology):
        '''
        description: updates the wireless technology of the network
        args:        self, the data rate
        return:      None
        '''
        self.wirelessTechnology = wirelessTechnology

    ''' ################################################################################################################################################################### '''
    def getPerDeviceBitRate(self, problemInstance, currentTimeSlot, deviceID):
        '''
        description: computes the bit rate observed by one device in Mbps, assuming network bandwidth is equally shared among its clients
        args:        self
        returns:     bit rate observed by a mobile device of the network (in Mbps)
        '''
        # return self.dataRate / len(self.associatedDevice) #self.numDevice
        relevantAssociatedDevice = Network.getRelevantAssociatedDevice(self, problemInstance, currentTimeSlot)
        if problem_instance.isNetworkAccessible(problemInstance, currentTimeSlot, self.networkID, deviceID):
            # return self.dataRate / len(self.associatedDevice) # self.numDevice
            return self.dataRate / len(relevantAssociatedDevice)
        return 0

    ''' ################################################################################################################################################################### '''
    def getPerDeviceDownload(self, problemInstance, currentTimeSlot, deviceID, timeSlotDuration, delay=0):
        '''
        description: computes the total download of a mobile device during a time slot in Mbits (when switching cost is considered)
        args:        self, delay (the delay incurred while diassociating from the previous network and associating with this one and resuming, e.g., download)
        returns:     total download of a device during one time slot (in Mbits), considering switching cost
        '''
        relevantAssociatedDevice = Network.getRelevantAssociatedDevice(self, problemInstance, currentTimeSlot)
        if problem_instance.isNetworkAccessible(problemInstance, currentTimeSlot, self.networkID, deviceID):
            return (self.dataRate / len(relevantAssociatedDevice)) * (timeSlotDuration - delay)
        return 0
        # return (self.dataRate / len(self.associatedDevice)) * (timeSlotDuration - delay)

    ''' ################################################################################################################################################################### '''
    def getNumAssociatedDevice(self):
        '''
        description: returns the number of devices associated to the network
        args:        self
        return:      count of associated device
        '''
        return len(self.associatedDevice)

    ''' ################################################################################################################################################################### '''
    def getAssociatedDevice(self):
        '''
        description: returns the set of devices associated to the network
        args:        self
        return:      set of associated device
        '''
        return self.associatedDevice

    ''' ################################################################################################################################################################### '''
    def getRelevantAssociatedDevice(self, problemInstance, currentTimeSlot):
        '''
        description: among the devices that chose the network, determine the number of them which actually have access to the network
        args:        self, the problem instance being considered, the current time slot
        return:      list of devices that selected the network which actually have access to it
        '''
        relevantAssociatedDevice = []
        for deviceID in self.associatedDevice:
            if problem_instance.isNetworkAccessible(problemInstance, currentTimeSlot, self.networkID, deviceID):
                relevantAssociatedDevice.append(deviceID)
        return relevantAssociatedDevice
        # end getRelevantAssociatedDevice
# end class Network