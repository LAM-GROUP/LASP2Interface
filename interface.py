from mpi4py import MPI
import lammps
import numpy as np
import matplotlib.pyplot as plt
import os

# Python script to perform a simulation with nnp potentials and check agreement

# MPI variables and set COMM_WORLD as communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# Training variables
potentialSeed = 1 #Potential used for main simulation
dataDir = 'PhysicalPropertiesTemplate/Configurations/Au100Messy.data' #Location of the structure being simulated
disagreement = []

threshold = 0.02001143776 #Disagreement value that will activate the training flag

totalSteps = 1000 #Total number of steps to be simulated
checkEvery = 100 #Number of steps after which the agreement will be measured
checkSteps = int(totalSteps / checkEvery) #Number of times the agreement will be measured

# Main LAMMPS object to carry on the simulation
lmp = lammps.lammps()
lmp.command('units metal')
lmp.command('boundary p p p')
lmp.command('read_data '+dataDir)

# Setup n2p2 potential
lmp.command('pair_style hdnnp 6.0 dir PotentialsComplete/Seed'+str(potentialSeed)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0')
lmp.command('pair_coeff * * Au')

# Read commands for the simulation to be performed
lmp.file('sim.melt')

# Simulation loop
for a in range(checkSteps):
    # Advance the simulation
    lmp.command('run '+str(checkEvery))

    # Save the structure
    lmp.command('write_data check.data')

    # Disagreement measurement loop
    seeds = [None]*5
    forces = [None]*5
    for b in range(5):
        # Create 5 lammps objects with no output
        seeds[b] = lammps.lammps(cmdargs=["-log", "none", "-screen", os.devnull,  "-nocite"])
        seeds[b].command('units metal')
        seeds[b].command('boundary p p p')
        # Read the last structure of the main simulation
        seeds[b].command('read_data check.data')

        # Setup potentials with different random seeds for each lammps object
        seeds[b].command('pair_style hdnnp 6.0 dir PotentialsComplete/Seed'+str(b+1)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0')
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
            buff = np.empty(1, dtype=np.float64)
            comm.Recv(buff, source=s, tag=s)
            forceNodes.append(buff)
        disag = max(forceNodes) #Get the maximum of the standard deviations obtained by each MPI process
        disagreement.append(max(forceNodes))
        # Flag activation
        if disag > threshold:
            # Disagreement is too high
            # DFT calculations will be performed and potentials will be trained
            print('Training is needed')
    else: #If this is not the main thread send a message to the main thread with the max standard deviation
        comm.Send(np.array([deviationMax], dtype=np.float64), dest=0, tag=rank)  
        

# Plotting disagreement over time
if rank == 0:
    disagreement = np.array(disagreement)

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Dark2.colors)
    fig, ax1 = plt.subplots()

    ax1.set_title('Evolution (100) Au (T = 1000)')
    ax1.set_ylabel('Disagreement')
    ax1.set_xlabel('Timestep')

    x = (np.arange(len(disagreement))+1)*checkEvery

    ax1.errorbar(x, disagreement, label='Disagreement', color='C0')

    ax1.legend()
    plt.savefig('timeAgreement.png')
    print('Finished')

# End of the program
MPI.Finalize()