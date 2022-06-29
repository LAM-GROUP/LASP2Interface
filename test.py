from mpi4py import MPI
import lammps
import numpy as np
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Python script to perform a simulation with nnp potentials and check agreement
potentialSeed = 1
dataDir = 'PhysicalPropertiesTemplate/Configurations/Au100Messy.data'
disagreement = []

threshold = 0.02001143776

totalSteps = 200
checkEvery = 100
checkSteps = int(totalSteps / checkEvery)

randNum = np.zeros(1)

if rank == 1:
    lmp = lammps.lammps()
    lmp.command('units metal')
    lmp.command('boundary p p p')
    lmp.command('read_data '+dataDir)

    lmp.command('pair_style nnp dir PotentialsComplete/Seed'+str(potentialSeed)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"')
    lmp.command('pair_coeff * * 6.01')

    lmp.file('sim.melt')

    for a in range(checkSteps):
        lmp.command('run '+str(checkEvery))

        lmp.command('write_data check.data')

        seeds = [None]*5
        forces = [None]*5
        for b in range(5):
            seeds[b] = lammps.lammps()
            seeds[b].command('units metal')
            seeds[b].command('boundary p p p')
            seeds[b].command('read_data check.data')

            seeds[b].command('pair_style nnp dir PotentialsComplete/Seed'+str(b+1)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"')
            seeds[b].command('pair_coeff * * 6.01')
            seeds[b].command('run 0')
            forces[b] = seeds[b].numpy.extract_atom('f').copy()
            seeds[b].close()
        deviationAtom = np.std(np.ma.masked_invalid(forces), axis=0)
        deviationMax = np.ma.masked_invalid(deviationAtom).max()
        disagreement.append(deviationMax)

    randNum = np.random.random_sample(1)
    print("Process", rank, "drew the number", randNum[0])
    req = comm.Isend(randNum, dest=0)
    req.Wait()
        
if rank == 0:
    print("Process", rank, "before receiving has the number", randNum[0])
    req = comm.Irecv(randNum, source=1)
    req.Wait()
    print("Process", rank, "received the number", randNum[0])
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
    print('Finished fawjepoifjapoweihfosidhfih')