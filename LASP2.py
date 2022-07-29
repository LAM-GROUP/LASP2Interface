import os
import sys
from sys import exit
import configparser
import ase
import time
from interfaceN2P2 import training
from interfaceVASP import compute
from dumpMerger import merge

def readLASP2():
    global lasp2
    vars = config['LASP2']
    for key in vars:
        if key == 'numseeds':
            try:
                lasp2[key] = int(vars[key])
            except:
                print('Invalid value for variable: ' +key)
                exit(1)
        else:
            print('Invalid variable: '+key)
            exit(1)

def readN2P2():
    global n2p2
    vars = config['N2P2']
    for key in vars:
        if key == 'dirpotentials':
            try:
                n2p2[key] = str(vars[key])
                if not os.path.isdir(n2p2[key]):
                    print('Directory of potentials was not found')
                    raise Exception('Directory not found')
                numSeeds = lasp2['numseeds']
                for i in range(1, numSeeds+1):
                    if not os.path.isdir(os.path.join(n2p2[key], 'Seed'+str(i))):
                        print('Seed'+str(i)+' not found')
                        raise Exception('Seed not found')
            except:
                print('Invalid value for variable: ' +key)
        elif key == 'epochslong':
            try:
                n2p2[key] = int(vars[key])
            except:
                print('Invalid value for variable: ' +key)
                exit(1)

def readVASP():
    global vasp
    vars = config['VASP']
    for key in vars:
        if key == 'numProcesses':
            try:
                vasp[key] = int(vars[key])
            except:
                print('Invalid value for variable: ' +key)

# Read input from lasp2.ini or another file indicated by the user
inputFile = 'lasp2.ini'
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

# Dictionaries where configuration variables will be stored
lasp2 = dict()
n2p2 = dict()
vasp = dict()

# Read input data for the interface
config = configparser.ConfigParser()
config.read(inputFile)
for section in config:
    if section == 'LASP2':
        readLASP2()
    elif section == 'LAMMPS':
        continue
    elif section == 'N2P2':
        readN2P2()
    elif section == 'VASP':
        print('Found variables for VASP')
    elif section == 'DEFAULT':
        if len(config[section]) > 0:
            print('Undefined section found: DEFAULT')
            exit(1)
    else:
        print('Undefined section found: ' + section)
        exit(1)

potDirs = 'Training/' #Training files produced during the simulation
os.makedirs(potDirs, exist_ok=True)
potInitial = n2p2['dirpotentials']
os.system('cp -r '+potInitial+' '+potDirs+'Potentials')
os.system('cp completeinput.data Training/complete0.data')
trainings = 1

# Begin simulation
exitCode = os.system('time srun $(placement ${SLURM_NTASKS_PER_NODE} 1 ) python3 /tmpdir/fresseco/install/LASP2Interface/interfaceLAMMPS.py --start -config '+inputFile+' -iteration '+str(trainings)+' > lasp2_'+str(trainings)+'.out')
print('LAMMPS exited with code: '+str(exitCode))
while True:
    if exitCode == 12800: #Exit code returned when the flag for training is activated
        print('Performing DFT calculations         Iteration: '+str(trainings))
        compute(trainings)
        print('Performing NNP training             Iteration: '+str(trainings))
        training(potDirs, trainings, lasp2['numseeds'], n2p2['epochslong'])
        trainings += 1
        exitCode = os.system('time srun $(placement ${SLURM_NTASKS_PER_NODE} 1 ) python3 /tmpdir/fresseco/install/LASP2Interface/interfaceLAMMPS.py --restart -config '+inputFile+' -iteration '+str(trainings)+' > lasp2_'+str(trainings)+'.out')
    else:
        break
if exitCode == 0:
    print('LAMMPS interface exited successfully')
    merge(trainings-1, 'Restart/dump*.lammpstrj', 'Restart/dumpComplete.lammpstrj')