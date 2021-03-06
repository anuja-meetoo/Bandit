'''
@description: Type *#0011* on the tethered mobile phone's keypad to enter service menu where the cellular load is displayed on the screen. Then run this program to save the cellular load.
'''
#!/usr/bin/python3
import sys
from subprocess import Popen, PIPE, STDOUT
import traceback
import argparse
import os
import csv
from datetime import datetime
from time import time

networkTypeList = ["3G", "LTE"]
loadParameter = ["EcIo", "RSRQ"] # 0 - 3G/H+; 1 - 4G/LTE
fileHeader = [["EcIo", "RSCP"], ["RSRP", "RSRQ", "SNR"]]

##### command line arguments
parser = argparse.ArgumentParser(description='Gets the cell load from adb logcat.')
parser.add_argument('-a', dest = "algorithm_name", required = True, help = 'name of wireless network selection algorithm (smartEXP3/greedy/wifi)')
parser.add_argument('-n', dest = "network_type", required = True, help = 'type of cellular network (3G/LTE)')
# read command-line arguments
args = parser.parse_args()
algorithmName = args.algorithm_name	    # algorithm name
networkType = int(networkTypeList.index(args.network_type))	# cellular network type

print("network type:", networkType)
outputCSVfile = os.getcwd() + "/cellLoad_" + algorithmName + "_" + (datetime.now().strftime("%H:%M:%S_%d%B%Y")) + ".csv"   # output csv file url
outfile = open(outputCSVfile, "w")
out = csv.writer(outfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
out.writerow(["timestamp", "date time"] + fileHeader[networkType])


p = Popen("adb logcat -c", stdout = PIPE, stderr = STDOUT, shell = True)
p = Popen("adb logcat -v time", stdout = PIPE, stderr = STDOUT, shell = True)
cellID = ""

try:
    while(1):
        # data = p.stdout.readline()
        # print(data)
        # data= str(data, 'ISO-8859-1').rstrip()
        data = str(p.stdout.readline(), 'ISO-8859-1').rstrip()
        if "CID" in data: cellID = data[:-1].split(":")[-1];
        if loadParameter[networkType] in data:
            timestamp, date = time(), datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(data)
            # save to csv file
            if networkType == 1: data = data.split("  ")[-1][:-1].split(" ")
            else: data = data.split("  ")[-1][:-1].split(",")
            # print(data)
            output = [timestamp, date, "" + str(cellID) + ""]
            for element in data: output.append(element.split(":")[1])
            out.writerow(output)
except KeyboardInterrupt:
    # close csv file
    outfile.close()
