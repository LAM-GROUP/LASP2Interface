import os
import sys
import configparser
import signal
import subprocess
import time
from interfaceN2P2 import training

# # Signal handler to receive flag from MPI to perform training
# def train(signum, frame):
#     print('Signal received')
#     training('PotentialsComplete/')
#     process.stdin.write(b'ready')
#     stdout, stderr = process.communicate()
#     print('sent stdin')
 
# signal.signal(signal.SIGCONT, train)

# process = subprocess.Popen('mpirun -mca mpi_yield_when_idle 1 -n 16 python3 interfaceLAMMPS.py '+str(os.getpid()), shell=True, stdin=subprocess.PIPE)   # pass cmd and args to the function
# print('waiting')
# time.sleep(60)
# process.wait()
# print('waiting')

# Read input from lasp2.ini or another file indicated by the user
inputFile = 'lasp2.ini'
for i in range(len(sys.argv)):
    if sys.argv[i] == '-i':
        try:
            inputFile = sys.argv[i+1]
        except:
            print('No valid inpupt parameter after option -i')
            exit(1)

config = configparser.ConfigParser()
config.read(inputFile)
for section in config:
    if section == 'LASP2':
        print('Found variables for LASP2')
    elif section == 'LAMMPS':
        print('Found variables for LAMMPS')
    elif section == 'N2P2':
        print('Found variables for N2P2')
    elif section == 'VASP':
        print('Found variables for VASP')
    elif section == 'DEFAULT':
        continue
    else:
        print('Undefined section found: ' + section)
        exit()

exit()

potDirs = 'Training/'
os.makedirs(potDirs, exist_ok=True)
potInitial = 'PotentialsComplete/'
os.system('cp -r '+potInitial+' '+potDirs+'Potentials')
trainings = 1

exitCode = os.system('mpirun -n 16 python3 interfaceLAMMPS.py '+str(os.getpid())+' start')
print('LAMMPS exited with code')
print(exitCode)
while True:
    if exitCode == 12800:
        print('Performing training')
        #training(potDirs, trainings)
        trainings += 1
        exitCode = os.system('mpirun -n 16 python3 interfaceLAMMPS.py '+str(os.getpid())+' restart')
    else:
        break