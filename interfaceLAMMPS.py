from mpi4py import MPI
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
for i in range(len(sys.argv)):
    if sys.argv[i] == '-pylammps':
        try:
            dirLibrary = sys.argv[i+1]
            if not os.path.isdir(dirLibrary):
                print('LAMMPS python library could not be found: '+dirLibrary)
                raise Exception('File error')
            sys.path.append(dirLibrary)
        except:
            print('No valid input parameter for lammps python library location')
            exit(1)
import lammps
import signal
import time

# Python script to perform a simulation with nnp potentials and check agreement

# Function to measure the agreement between the different potentials
def check():
    disag = 0.0
    # Disagreement measurement loop
    seeds = [None]*5
    forces = [None]*5
    for b in range(5):
        # Create 5 lammps objects with no output
        seeds[b] = lammps.lammps(cmdargs=["-log", "none", "-screen", os.devnull,  "-nocite"])
        seeds[b].command('units metal')
        seeds[b].command('boundary p p p')
        # Read the last structure of the main simulation
        seeds[b].command('read_data Restart/check.data')

        # Setup potentials with different random seeds for each lammps object
        seeds[b].command('pair_style hdnnp 6.0 dir '+potDir+'/Seed'+str(b+1)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0')
        seeds[b].command('pair_coeff * * Au')
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
        disagreement.append(disag)
        # Flag activation
        message = 0
        if disag > threshold:
            sections.append(disagreement.copy())
            disagreement.clear()
            message = 1
            # Disagreement is too high
            # DFT calculations will be performed and potentials will be trained
        for i in range(1, nprocs):
            comm.send(message, dest=i, tag=i)
        if disag > threshold:
            np.save('Restart/sections.npy', np.array(sections, dtype=object))
            MPI.Finalize()
            exit(50)
    return disag

def initialize():
    # Main LAMMPS object to carry on the simulation
    lmp.command('units metal')
    lmp.command('boundary p p p')
    lmp.command('read_data '+dataDir)

    # Setup n2p2 potential
    lmp.command('pair_style hdnnp 6.0 dir '+potDir+'/Seed'+str(potentialSeed)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0')
    lmp.command('pair_coeff * * Au')

    # Setup internal dump
    lmp.command('dump              dumpInternalLASP2 all custom 10  Restart/dump${iteration}.lammpstrj id type x y z fx fy fz')
    lmp.command('dump_modify    	  dumpInternalLASP2 sort id')

    # Read commands for the simulation to be performed
    lmp.file('input.lmp')
    lmp.command('run 0')

def restart():
    global sections
    global startPoint
    global threshold
    #threshold = 0.2 ################################################## TEST #################################
    if rank == 0:
        sections = np.load('Restart/sections.npy', allow_pickle=True)
        sections = list(sections)
        for i in range(len(sections)):
            startPoint += len(sections[i]) - 1
        startPoint -= (len(sections)-1)
        if startPoint < 0:
            startPoint = 0
        for i in range(1, nprocs):
            comm.send(startPoint, dest=i, tag=i)
    else:
        startPoint = comm.recv(source = 0, tag = rank)
    lmp.command('read_restart Restart/tmp'+str(startPoint*checkEvery)+'.restart')
    
    # Setup n2p2 potential
    lmp.command('pair_style hdnnp 6.0 dir '+potDir+'/Seed'+str(potentialSeed)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0')
    lmp.command('pair_coeff * * Au')

    # Setup internal dump
    lmp.command('dump              dumpInternalLASP2 all custom 10  Restart/dump${iteration}.lammpstrj id type x y z fx fy fz')
    lmp.command('dump_modify    	  dumpInternalLASP2 sort id')

    # Read commands for the simulation to be performed
    lmp.file('restart.lmp')
    lmp.command('run 0')

# MPI variables and set COMM_WORLD as communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

potentialSeed = 1 #Potential used for main simulation
dataDir = 'Bulk.data' #Location of the structure being simulated
#potDir = 'PotentialsComplete' ############################### TEST ####################################
potDir = 'Training/Potentials'
lmp = lammps.lammps()
os.makedirs('Restart', exist_ok=True)

# Training variables
disagreement = []
sections = []
threshold = 0.02001143776 * 4 #Disagreement value that will activate the training flag
totalSteps = 300 #Total number of steps to be simulated
checkEvery = 100 #Number of steps after which the agreement will be measured
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
if sys.argv[2] == 'start':
    initialize()
elif sys.argv[2] == 'restart':
    restart()

# Simulation loop
for a in range(startPoint, checkSteps):
    # Save the structure
    lmp.command('write_data Restart/check.data')
    # Write file to restart
    lmp.command('write_restart Restart/tmp*.restart')
    # Measure agreement
    disag = check()

    # Advance the simulation
    lmp.command('run '+str(checkEvery))

# Measure agreement of last structure in the simulation
lmp.command('write_data Restart/check.data')
# Write file to restart
lmp.command('write_restart Restart/tmp*.restart')
# Measure agreement
disag = check()
sections.append(disagreement.copy())
np.save('Restart/sections.npy', np.array(sections, dtype=object))

# Plotting disagreement over time (Only on rank 0)
if rank == 0:
    def hex_to_RGB(hex_str):
        """ #FFFFFF -> [255,255,255]"""
        #Pass 16 to the integer function for change of base
        return [int(hex_str[i:i+2], 16) for i in range(1, 6, 2)]
    def get_color_gradient(c1, c2, n):
        """
        Given two hex colors, returns a color gradient
        with n colors.
        """
        assert n > 1
        c1_rgb = np.array(hex_to_RGB(c1))/255
        c2_rgb = np.array(hex_to_RGB(c2))/255
        mix_pcts = [x/(n-1) for x in range(n)]
        rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
        return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Dark2.colors)
    fig, ax1 = plt.subplots()

    ax1.set_title('Evolution (100) Au (T = 1000)')
    ax1.set_ylabel('Disagreement')
    ax1.set_xlabel('Timestep')

    x = []
    colors = get_color_gradient("#FF0000", "#0000FF", len(sections))
    for i in range(len(sections)):
        startLine = 0
        if i > 0:
            for b in range(i):
                startLine += len(sections[b]) - 1
            startLine -= (i-1)
        x.append((np.arange(len(sections[i]))+startLine)*checkEvery)
        ax1.plot(x[i], sections[i], c=colors[i], marker='o', ls=':')
    ax1.axhline(threshold, ls='--', color='black')

    fig.savefig('timeAgreement.png')
    print('Simulation Completed')

# End of the program
MPI.Finalize()