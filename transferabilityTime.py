import sys
import lammps
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

disagreement = []

totalSteps = 1000
checkEvery = 100
checkSteps = int(totalSteps / checkEvery)

# Python script to perform a simulation with nnp potentials and check agreement
potentialSeed = 1
dataDir = 'PhysicalPropertiesTemplate/Configurations/Au100Messy.data'
locationSeeds = ['PotentialsComplete', 'PotentialsNo100', 'PotentialsNo111', 'PotentialsNoSurface']

print(sys.argv)
print('')

# me = MPI.COMM_WORLD.Get_rank()
# nprocs = MPI.COMM_WORLD.Get_size()

for a in range(len(locationSeeds)):
    lmp = lammps.lammps()
    lmp.command('units metal')
    lmp.command('boundary p p p')
    lmp.command('read_data '+dataDir)

    lmp.command('pair_style nnp dir '+locationSeeds[0]+'/Seed'+str(potentialSeed)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"')
    lmp.command('pair_coeff * * 6.01')

    lmp.file('sim.melt')

    #lmp.command('run 0')

    lmp.command('write_data check.data')
    for s in range(checkSteps):
        lmp.command('run '+str(checkEvery))

        lmp.command('write_data check.data')

        disagreement.append([])

        seeds = [None]*5
        forces = [None]*5
        for b in range(len(seeds)):
            seeds[b] = lammps.lammps()
            seeds[b].command('units metal')
            seeds[b].command('boundary p p p')
            seeds[b].command('read_data check.data')

            seeds[b].command('pair_style nnp dir '+locationSeeds[a]+'/Seed'+str(b+1)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"')
            seeds[b].command('pair_coeff * * 6.01')
            seeds[b].command('run 0')
            forces[b] = seeds[b].numpy.extract_atom('f').copy()
            seeds[b].close()
        deviationAtom = np.std(np.ma.masked_invalid(forces), axis=0)
        deviationMax = np.ma.masked_invalid(deviationAtom).max()

        disagreement[a].append(deviationMax)
    lmp.close()

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Dark2.colors)

fig, ax1 = plt.subplots()

ax1.set_title('Evolution (100) Au (T = 1000)')
ax1.set_ylabel('Disagreement')
ax1.set_xlabel('Timestep')

x = (np.arange(len(disagreement[0]))+1)*checkEvery

ax1.plot(x, disagreement[0], label='Complete', color='C0')
ax1.plot(x, disagreement[1], label='No100', color='C1')
ax1.plot(x, disagreement[2], label='No111', color='C2')
#ax1.plot(x, disagreement[3], label='No Surface', color='C3')

ax1.legend()
plt.savefig('transferabilityTime.png')
print('Finished fawjepoifjapoweihfosidhfih')

#MPI.Finalize()