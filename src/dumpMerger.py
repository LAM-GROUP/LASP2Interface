import os
import numpy as np
from sys import exit
import re

# Function used to merge dump files of the different iterations
def merge(numIterations, fileNames='dump*.lammpstrj', outputFile='dumpComplete.lammpstrj'): #Reads starting from the last file
    dumpFile = fileNames.replace('*', str(numIterations))
    # Check if latest file exists
    if not os.path.isfile(dumpFile):
        print('Dump file of last iteration could not be found: '+dumpFile)
        exit()
    f = open(dumpFile, 'r')
    outputDump = f.readlines()
    f.close()
    earlier = 0
    # Obtain the earliest simulation step in the last file
    for i in range(len(outputDump)):
        if "ITEM: TIMESTEP" in outputDump[i]:
            earlier = int(outputDump[i+1])
            break
    # Iterate backwards to read the rest of the files
    for i in reversed(range(1, numIterations)):
        dumpFile = fileNames.replace('*', str(i))
        # Get the earliest step of the next file
        steps = os.popen('grep --no-group-separator -A1 "ITEM: TIMESTEP" '+dumpFile+' | grep -v "ITEM: TIMESTEP"').read()
        steps = steps.splitlines()
        first = int(steps[0])
        # If next file has an earlier step, this can be appended to the output dumps file
        if first < earlier:
            f = open(dumpFile, 'r')
            earlierDump = f.readlines()
            f.close()
            positionCut = 0
            for b in range(len(earlierDump)):
                if "ITEM: TIMESTEP" in earlierDump[b]:
                    timeStep = int(earlierDump[b+1])
                    if timeStep >= earlier:
                        positionCut = b
                        break
            outputDump = earlierDump[:positionCut] + outputDump
            earlier = first
        # If the beginning of the simulation is reached, stop checking
        if earlier == 0:
            break
    # Write the output dumps file
    print('Writing output file: '+outputFile)
    f = open(outputFile, 'w')
    f.writelines(outputDump)
    f.close()