Evaluates the performance of Smart EXP3, greedy and epsilon-greedy based on real network traces. Traces of a Starbucks public WiFi network and SingTel's 3G network were collected by downloading a file from a remote server on both networks simultaneously and measuring their bit rates.

A brief description of the files is as follows:
* simulationTrace_resetMinimal.py: Evaluates Smart EXP3, greedy and epsilon-greedy algorithms based on real network traces.
* extractDataFromNetworkTrace.py: Extracts the per time slot data rate over WiFi and cellular network, and per time slot cellular load (average over a time slot as more than one value might be available for one time slot).
* scenario*.csv: Network trace file.
* scenario_details.txt: Details of how each of the network traces were collected.
