from mpi4py import MPI
import numpy as np
import configparser
import sectionsParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from sys import exit

# Read name of configuration file
inputFile = 'lasp2.ini'
for i in range(len(sys.argv)):
    if sys.argv[i] == '-config':
        try:
            inputFile = sys.argv[i+1]
            if not os.path.isfile(inputFile):
                print('Input file could not be found: '+inputFile)
                raise Exception('File error')
        except:
            print('No valid input parameter after option -i')
            exit(1)

lammpsConf = dict()
# Read input data for the interface
config = configparser.ConfigParser()
config.read(inputFile)

# Read configuration options from lasp2.ini
vars = config['LAMMPS']
for key in vars:
    if key == 'totalsteps':
        try:
            lammpsConf[key] = int(vars[key])
        except:
            print('Invalid value for variable: ' +key)
            exit(1)
    elif key == 'checksteps':
        try:
            lammpsConf[key] = int(vars[key])
        except:
            print('Invalid value for variable: ' +key)
            exit(1)
    elif key == 'threshold':
        try:
            lammpsConf[key] = float(vars[key])
        except:
            print('Invalid value for variable: ' +key)
            exit(1)
    elif key == 'dirpylammps':
        try:
            lammpsConf[key] = str(vars[key])
            if lammpsConf[key] == '':
                continue
            if not os.path.isdir(lammpsConf[key]):
                print('LAMMPS python library could not be found: '+lammpsConf[key])
                raise Exception('Directory not found')
        except:
            print('Invalid value for variable: ' +key)
            exit(1)
    else:
        print('Invalid variable: '+key)
        exit(1)

if 'dirpylammps' in lammpsConf:
    try:
        sys.path.append(lammpsConf['dirpylammps'])
    except:
        print('No valid input parameter for lammps python library location')
        exit(1)
import lammps

numSeeds = 0
simName = 'LASP2 simulation'
vars = config['LASP2']
for key in vars:
    if key == 'numseeds':
        try:
            numSeeds = int(vars[key])
        except:
            print('Invalid value for variable: ' +key)
            exit(1)
    if key == 'simname':
        try:
            simName = str(vars[key])
        except:
            print('Invalid value for variable: ' +key)
            exit(1)

# End import
####################################################################################
# .##..........###....##.....##.##.....##.########...######.
# .##.........##.##...###...###.###...###.##.....##.##....##
# .##........##...##..####.####.####.####.##.....##.##......
# .##.......##.....##.##.###.##.##.###.##.########...######.
# .##.......#########.##.....##.##.....##.##..............##
# .##.......##.....##.##.....##.##.....##.##........##....##
# .########.##.....##.##.....##.##.....##.##.........######.

# .####.##....##.########.########.########..########....###.....######..########
# ..##..###...##....##....##.......##.....##.##.........##.##...##....##.##......
# ..##..####..##....##....##.......##.....##.##........##...##..##.......##......
# ..##..##.##.##....##....######...########..######...##.....##.##.......######..
# ..##..##..####....##....##.......##...##...##.......#########.##.......##......
# ..##..##...###....##....##.......##....##..##.......##.....##.##....##.##......
# .####.##....##....##....########.##.....##.##.......##.....##..######..########

# Python script to perform a simulation with nnp potentials and check agreement

####################################################################################

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

# Function to measure the agreement between the different potentials
def check(iteration):
    """Check dispersion by measuring forces with different potentials"""
    disag = 0.0
    # Dispersion measurement loop
    seeds = [None]*numSeeds
    forces = [None]*numSeeds
    for b in range(numSeeds):
        # Create numSeeds lammps objects with no output
        seeds[b] = lammps.lammps(cmdargs=["-log", "none", "-screen", "none",  "-nocite"])
        # Set seed number
        seeds[b].command('variable seed internal '+str(b+1))

        # Read commands for the simulation to be performed
        seeds[b].file('check.lmp')

        # Perform an empty step to calculate forces
        seeds[b].command('run 0')
        # Extract forces and destroy temporary lammps object
        forces[b] = seeds[b].numpy.extract_atom('f').copy()
        seeds[b].close()
    # Calculate the standard deviation of the forces calculated with the different random seeds
    deviationAtom = np.std(np.ma.masked_invalid(forces), axis=0)
    # Get the maximum standard deviation
    try:
        deviationMax = np.ma.masked_invalid(deviationAtom).max()
    except ValueError: #In case the forces array of this MPI process is empty
        deviationMax = 0.0
    # MPI process communication
    if rank == 0: #If this is the main thread, wait for other threads to send their maximum standard deviation
        forceNodes = []
        forceNodes.append(deviationMax)
        for s in range(1, nprocs):
            message = comm.recv(source=s, tag=s)
            forceNodes.append(message)
        disag = max(forceNodes) #Get the maximum of the standard deviations obtained by each MPI process
    else: #If this is not the main thread send a message to the main thread with the max standard deviation
        comm.send(deviationMax, dest=0, tag=rank)
        # Wait for message to know that it is ok to continue
        message = comm.recv(source = 0, tag = rank)
        if message == 1:
            MPI.Finalize()
            exit()
    
    if rank == 0:
        dispersion[0].append(iteration*checkEvery)
        dispersion[1].append(disag)
        print('############ LASP2 #############')
        print('step                  dispersion')
        print(str(iteration*checkEvery)+'        '+str(disag))
        # Flag activation
        message = 0
        if disag > threshold:
            print('DISPERSION IS LARGER THAN THRESHOLD!')
            print('Threshold value: '+str(threshold))
            print('SIMULATION WILL BE INTERRUPTED AND TRAINING WILL TAKE PLACE')
            sections.append(dispersion.copy())
            message = 1
            # Dispersion is too high
            # DFT calculations will be performed and potentials will be trained
        print('################################')
        for i in range(1, nprocs):
            comm.send(message, dest=i, tag=i)
        if disag > threshold:
            sectionsParser.save(sections, 'Restart/sections.out', threshold=str(threshold), totalSteps=str(totalSteps), checkEvery=str(checkEvery))
            print('50', file=sys.stderr)
            MPI.Finalize()
            exit()
    return disag

def initialize():
    """Initialize a new LAMMPS simulation"""
    # Read commands for the simulation to be performed
    lmp.file('input.lmp')

    # Setup internal dump
    lmp.command('dump              dumpInternalLASP2 all custom '+str(checkEvery)+' Restart/dump${iteration}.lammpstrj id type x y z fx fy fz')
    lmp.command('dump_modify    	  dumpInternalLASP2 sort id')

def restart():
    """Restart LAMMPS simulation after training"""
    global sections
    global startPoint
    
    # Read file containing the dispersion at each step and get the number of step
    if rank == 0: #Only read with one process
        sections = sectionsParser.load('Restart/sections.out')
        startPoint = int(sections[-1][0][-1] / checkEvery) - 1
        if startPoint < 0:
            startPoint = 0
        # Send starting steps to the other processes
        for i in range(1, nprocs):
            comm.send(startPoint, dest=i, tag=i)
    else: #Other processes wait to receive the starting step
        startPoint = comm.recv(source = 0, tag = rank)

    lmp.command('variable restartStep internal '+str(startPoint*checkEvery))

    # Read commands for the simulation to be performed
    lmp.file('restart.lmp')

    # Setup internal dump
    lmp.command('dump              dumpInternalLASP2 all custom '+str(checkEvery)+' Restart/dump${iteration}.lammpstrj id type x y z fx fy fz')
    lmp.command('dump_modify    	  dumpInternalLASP2 sort id')

def plot():
    def hex_to_RGB(hex_str):
        """ #FFFFFF -> [255,255,255]"""
        #Pass 16 to the integer function for change of base
        return [int(hex_str[i:i+2], 16) for i in range(1, 6, 2)]
    def get_color_gradient(c1, c2, n):
        """
        Given two hex colors, returns a color gradient
        with n colors.
        """
        if n > 1:
            c1_rgb = np.array(hex_to_RGB(c1))/255
            c2_rgb = np.array(hex_to_RGB(c2))/255
            mix_pcts = [x/(n-1) for x in range(n)]
            rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
            return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]
        elif n == 1:
            return [c1]

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Dark2.colors)
    fig, ax1 = plt.subplots()

    ax1.set_yscale('log')
    ax1.set_title(simName)
    ax1.set_ylabel('Dispersion')
    ax1.set_xlabel('Timestep')

    colors = get_color_gradient("#FF0000", "#0000FF", len(sections))
    for i in range(len(sections)):
        ax1.plot(sections[i][0], sections[i][1], c=colors[i], marker='o', ls=':')
    ax1.axhline(threshold, ls='--', color='black')

    fig.savefig('simulationDispersion.png')
    print('Image saved... "simulationDispersion.png"')

for i in range(len(sys.argv)):
    if sys.argv[i] == '--plot':
        print('...')
        threshold = lammpsConf['threshold']
        try:
            print('Reading "Restart/sections.out"')
            sections = sectionsParser.load('Restart/sections.out')
        except:
            print('Sections file not found under Restart directory')
            print('Trying to read in  working directory...')
        try:
            sections = sectionsParser.load('sections.out')
        except:
            print('Sections file not found')
            print('Exiting ...')
            exit()
        plot()
        exit()

# MPI variables and set COMM_WORLD as communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

lmp = lammps.lammps(cmdargs=["-log", "none"])

# Training variables
dispersion = [[],[]]
sections = []
threshold = lammpsConf['threshold'] #Dispersion value that will activate the training flag
totalSteps = lammpsConf['totalsteps'] #Total number of steps to be simulated
checkEvery = lammpsConf['checksteps'] #Number of steps after which the agreement will be measured
checkSteps = int(totalSteps / checkEvery) #Number of times the agreement will be measured
startPoint = 0

# Read the number of iteration from the input parameters
for i in range(len(sys.argv)):
    if sys.argv[i] == '-iteration':
        try:
            numIteration = int(sys.argv[i+1])
            lmp.command('variable iteration internal '+str(numIteration))
        except:
            print('No valid number of iteration')
            exit(1)

# Check if the simulation is being restarted after training
for i in range(len(sys.argv)):
    if sys.argv[i] == '--start':
        initialize()
        break
    elif sys.argv[i] == '--restart':
        restart()
        break

# Simulation loop
for a in range(startPoint, checkSteps):
    # Save the structure
    lmp.command('write_data Restart/check.data')
    # Write file to restart
    lmp.command('write_restart Restart/tmp*.restart')
    # Measure agreement
    disag = check(a)

    # Advance the simulation
    lmp.command('run '+str(checkEvery)+' start 0 stop '+str(totalSteps)+' pre no post yes')

# Measure agreement of last structure in the simulation
lmp.command('write_data Restart/check.data')
# Write file to restart
lmp.command('write_restart Restart/tmp*.restart')
# Measure agreement
disag = check(a+1)
if rank == 0:
    sections.append(dispersion.copy())
    sectionsParser.save(sections, 'Restart/sections.out', threshold=str(threshold), totalSteps=str(totalSteps), checkEvery=str(checkEvery))

# Plotting dispersion over time (Only on rank 0)
if rank == 0:
    plot()
    print('Simulation Completed')

# End of the program
MPI.Finalize()