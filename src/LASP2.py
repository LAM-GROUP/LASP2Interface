import os
from os.path import isdir, isfile
import sys
from sys import exit
import subprocess
from subprocess import Popen
import re
import configparser
import sectionsParser
import ase
import time
import interfaceN2P2
import interfaceVASP
import shutil
from dumpMerger import merge

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
# In order to make python output be immediately written to the slurm output file
sys.stdout = Unbuffered(sys.stdout)

def readLASP2():
    global lasp2
    vars = config['LASP2']
    lasp2['simname'] = 'LASP2 Simulation'
    lasp2['procslammps'] = 0
    lasp2['procsn2p2'] = 0
    lasp2['procsvasp'] = 0
    for key in vars:
        if key == 'simname':
            try:
                lasp2[key] = str(vars[key])
            except:
                print('Invalid value for variable: ' +key)
                exit(1)
        elif key == 'numseeds':
            try:
                lasp2[key] = int(vars[key])
            except:
                print('Invalid value for variable: ' +key)
                exit(1)
        elif key == 'numprocs':
            try:
                lasp2[key] = int(vars[key])
            except:
                print('Invalid value for variable: ' + key)
                exit(1)
        elif key == 'procslammps':
            try:
                lasp2[key] = int(vars[key])
            except:
                print('Invalid value for variable: ' + key)
                exit(1)
        elif key == 'procsn2p2':
            try:
                lasp2[key] = int(vars[key])
            except:
                print('Invalid value for variable: ' + key)
                exit(1)
        elif key == 'procsvasp':
            try:
                lasp2[key] = int(vars[key])
            except:
                print('Invalid value for variable: ' + key)
                exit(1)
        elif key == 'exec':
            try:
                lasp2[key] = str(vars[key])
            except:
                print('Invalid value for variable: ' + key)
                exit(1)
        elif key == 'dirpotentials':
            try:
                lasp2[key] = str(vars[key])
                if not os.path.isdir(lasp2[key]):
                    print('Directory of potentials was not found')
                    raise Exception('Directory not found')
                numSeeds = lasp2['numseeds']
                for i in range(1, numSeeds+1):
                    if not os.path.isdir(os.path.join(lasp2[key], 'Seed'+str(i))):
                        print('Seed'+str(i)+' not found')
                        raise Exception('Seed not found')
            except:
                print('Invalid value for variable: ' +key)
        elif key == 'dirdatabase':
            try:
                lasp2[key] = str(vars[key])
                if not os.path.isfile(lasp2[key]):
                    print('n2p2 database file could not be found: '+lasp2[key])
                    raise Exception('File error')
            except:
                print('Invalid value for variable: ' + key)
        else:
            print('Invalid variable: '+key)
            exit(1)
    #If number of processors for lammps, n2p2 or vasp was not overwritten use the general number
    if lasp2['procslammps'] == 0:
        lasp2['procslammps'] = lasp2['numprocs']
    if lasp2['procsn2p2'] == 0:
        lasp2['procsn2p2'] = lasp2['numprocs']
    if lasp2['procsvasp'] == 0:
        lasp2['procsvasp'] = lasp2['numprocs']

# Read input from lasp2.ini or another file indicated by the user
inputFile = 'lasp2.ini'
dirInterface = '###INTERFACE###'
restart = False
if not os.path.isfile(dirInterface):
    print('Binary file for LAMMPS interface could not be found: '+dirInterface)
    raise Exception('File error')
for i in range(len(sys.argv)):
    if sys.argv[i] == '-i':
        try:
            inputFile = sys.argv[i+1]
            if not os.path.isfile(inputFile):
                print('Input file could not be found: '+inputFile)
                raise Exception('File error')
        except:
            print('No valid input parameter after option -i')
            exit(1)
    if sys.argv[i] == '--merge':
        numDumps = 0
        nameDump = ''
        outputDump = ''
        if len(sys.argv) > i+4:
            print('Error: more parameters than expected')
            print('Expected input:')
            print('--merge (number of dumps) (name of dumps) (name of output)')
            print("--merge 10 'dump*.lammpstrj' dumpComplete.lammpstrj")
            print('Name of dump files has to be given inside quotes, as to avoid using regular expressions')
        try:
            numDumps = int(sys.argv[i+1])
            nameDump = str(sys.argv[i+2])
            outputDump = str(sys.argv[i+3])
        except:
            print('Wrong input for merging command')
            print('Expected input:')
            print('--merge (number of dumps) (name of dumps) (name of output)')
            print("--merge 10 'dump*.lammpstrj' dumpComplete.lammpstrj")
            exit(1)
        merge(numDumps, nameDump, outputDump)
        exit()
    if sys.argv[i] == '--restart':
        restart = True
    if sys.argv[i] == '--plot':
        print('Plotting sections')
        lammpsPlot = Popen(dirInterface+' --plot', shell=True, stderr=subprocess.PIPE)
        lammpsPlot.wait()
        exit()

# Dictionary where configuration variables will be stored
lasp2 = dict()
# Read input data for the interface
config = configparser.ConfigParser()
config.read(inputFile)
for section in config:
    if section == 'LASP2':
        readLASP2()
    elif section == 'LAMMPS':
        continue
    elif section == 'N2P2':
        interfaceN2P2.readN2P2(inputFile)
    elif section == 'VASP':
        interfaceVASP.readVASP(inputFile)
    elif section == 'DEFAULT':
        if len(config[section]) > 0:
            print('Undefined section found: DEFAULT')
            exit(1)
    else:
        print('Undefined section found: ' + section)
        exit(1)

potInitial = lasp2['dirpotentials']

if restart:
    print('Simulation is being restarted')
    if not os.path.isfile('Restart/sections.out'): #If file does not exist print error message and exit
        print('Restarting Error: "sections.out" file could not be found in the "Restart" directory. Restarting point could not be read.')
        exit(1)
    # Else, read it and find starting point
    sections = sectionsParser.load('Restart/sections.out')
    trainings = len(sections)
    if os.path.isdir('Training/nnp'+str(trainings)): #If this directory exists DFT finished successfully
        print('DFT calculation was performed before')
        if os.path.isfile('lammps_'+str(trainings+1)+'.out'):
            print('Training was performed before')
            print('LAMMPS simulation needs to be restarted')
            os.remove('lammps_'+str(trainings+1)+'.out') #Deleting previous LAMMPS simulation that was not finished
            trainings += 1
            print('Performing LAMMPS simulation        Iteration: '+str(trainings))
            lammpsRun = Popen(lasp2['exec']+' -n '+str(lasp2['procslammps'])+' '+dirInterface+' --restart -config '+inputFile+' -iteration '+str(trainings)+' > lammps_'+str(trainings)+'.out', shell=True, stderr=subprocess.PIPE)
            lammpsRun.wait()
            exitErr = lammpsRun.stderr.read().decode()
            print('LAMMPS exited with stderr: '+exitErr)
        else:
            #Restarting from the training
            print('Training needs to be restarted')
            shutil.rmtree('Training/nnp'+str(trainings)) #Deleting previous training that was not finished
            print('Performing NNP training             Iteration: '+str(trainings))
            interfaceN2P2.training(lasp2['exec'], trainings, lasp2['numseeds'], lasp2['procsn2p2'])
            trainings += 1
            print('Performing LAMMPS simulation        Iteration: '+str(trainings))
            lammpsRun = Popen(lasp2['exec']+' -n '+str(lasp2['procslammps'])+' '+dirInterface+' --restart -config '+inputFile+' -iteration '+str(trainings)+' > lammps_'+str(trainings)+'.out', shell=True, stderr=subprocess.PIPE)
            lammpsRun.wait()
            exitErr = lammpsRun.stderr.read().decode()
            print('LAMMPS exited with stderr: '+exitErr)
    else:
        if os.path.isdir('DFT/dft'+str(trainings)): #If DFT directory exists, but training does not, DFT must be restarted.
            #Restarting from DFT
            print('DFT calculation needs to be restarted')
            shutil.rmtree('DFT/dft'+str(trainings)) #Deleting previous DFT calculation that was not finished
            print('Performing DFT calculations         Iteration: '+str(trainings))
            interfaceVASP.compute(lasp2['exec'], trainings, lasp2['procsvasp'])
            print('Performing NNP training             Iteration: '+str(trainings))
            interfaceN2P2.training(lasp2['exec'], trainings, lasp2['numseeds'], lasp2['procsn2p2'])
            trainings += 1
            print('Performing LAMMPS simulation        Iteration: '+str(trainings))
            lammpsRun = Popen(lasp2['exec']+' -n '+str(lasp2['procslammps'])+' '+dirInterface+' --restart -config '+inputFile+' -iteration '+str(trainings)+' > lammps_'+str(trainings)+'.out', shell=True, stderr=subprocess.PIPE)
            lammpsRun.wait()
            exitErr = lammpsRun.stderr.read().decode()
            print('LAMMPS exited with stderr: '+exitErr)
        else: #If DFT directory does not exist, simulation might have finished (COULD ADD A CHECK TO KNOW IF IT FINISHED)
            print('Simulation has been finished. Restart unsuccessful')
            exit(1)
else: # Default starting point
    print('LASP2 simulation starting')
    os.makedirs('Training/', exist_ok=True)
    os.makedirs('Restart', exist_ok=True)
    os.system('cp -r '+potInitial+' Training/Potentials')
    os.system('cp '+lasp2['dirdatabase']+' Training/complete0.data')
    trainings = 1
    # Begin running LAMMPS
    print('Performing LAMMPS simulation        Iteration: '+str(trainings))
    lammpsRun = Popen(lasp2['exec']+' -n '+str(lasp2['procslammps'])+' '+dirInterface+' --start -config '+inputFile+' -iteration '+str(trainings)+' > lammps_'+str(trainings)+'.out', shell=True, stderr=subprocess.PIPE)
    lammpsRun.wait()
    exitErr = lammpsRun.stderr.read().decode()
    print('LAMMPS exited with stderr: '+exitErr)

# LAMMPS, n2p2 and VASP loop. Executed as far as LAMMPS exits with stderr 50
while True:
    if re.match('^50', exitErr): #Exit code returned when the flag for training is activated
        print('Performing DFT calculations         Iteration: '+str(trainings))
        interfaceVASP.compute(lasp2['exec'], trainings, lasp2['procsvasp'])
        print('Performing NNP training             Iteration: '+str(trainings))
        interfaceN2P2.training(lasp2['exec'], trainings, lasp2['numseeds'], lasp2['procsn2p2'])
        trainings += 1
        print('Performing LAMMPS simulation        Iteration: '+str(trainings))
        lammpsRun = Popen(lasp2['exec']+' -n '+str(lasp2['procslammps'])+' '+dirInterface+' --restart -config '+inputFile+' -iteration '+str(trainings)+' > lammps_'+str(trainings)+'.out', shell=True, stderr=subprocess.PIPE)
        lammpsRun.wait()
        exitErr = lammpsRun.stderr.read().decode()
        print('LAMMPS exited with stderr: '+exitErr)
    else: #If stderr is different then exit the loop
        break

# Merge internal LAMMPS dump files
if not re.match('^50', exitErr):
    print('LAMMPS interface exited successfully')
    if trainings != 1: #If there is more than one then merge them
        merge(trainings, 'Restart/dump*.lammpstrj', 'Restart/dumpComplete.lammpstrj')
    else: #Else, just copy the only file and name it dumpComplete.lammpstrj
        os.system('cp Restart/dump1.lammpstrj Restart/dumpComplete.lammpstrj')