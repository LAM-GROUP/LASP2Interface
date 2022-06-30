import sys
import lammps
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

disagreement = []

# Python script to perform a simulation with nnp potentials and check agreement
potentialSeed = 1
dataDir = 'PhysicalPropertiesTemplate/Configurations/Au100Messy.data'
locationSeeds = ['PotentialsComplete', 'PotentialsNo100', 'PotentialsNo111', 'PotentialsNoSurface']

print(sys.argv)
print('')
lmp = lammps.lammps()
lmp.command('units metal')
lmp.command('boundary p p p')
lmp.command('read_data '+dataDir)

lmp.command('pair_style hdnnp dir '+locationSeeds[0]+'/Seed'+str(potentialSeed)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"')
lmp.command('pair_coeff * * 6.01')

lmp.file('sim.melt')
lmp.command('run 0')

#lmp.command('run 0')
# me = MPI.COMM_WORLD.Get_rank()
# nprocs = MPI.COMM_WORLD.Get_size()
# MPI.Finalize()

lmp.command('write_data check.data')

for a in range(len(locationSeeds)):
    disagreement.append([])

    seeds = [None]*5
    forces = [None]*5
    for b in range(len(seeds)):
        seeds[b] = lammps.lammps()
        seeds[b].command('units metal')
        seeds[b].command('boundary p p p')
        seeds[b].command('read_data check.data')

        seeds[b].command('pair_style hdnnp dir '+locationSeeds[a]+'/Seed'+str(b+1)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"')
        seeds[b].command('pair_coeff * * 6.01')
        seeds[b].command('run 0')
        forces[b] = seeds[b].numpy.extract_atom('f')
    deviationAtom = np.std(np.ma.masked_invalid(forces), axis=0)
    deviationMax = np.ma.masked_invalid(deviationAtom).max()

    disagreement[a].append(deviationMax)

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Dark2.colors)
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams['axes.axisbelow'] = True

surfaces = ['100']

# set width of bars
barWidth = 0.15
 
# Set position of bar on X axis
r1 = np.arange(1)-(barWidth/2)
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
 
plt.figure()
plt.title('MAX Standard Deviation Au')
plt.ylabel('Standard Deviation')
plt.xlabel('Surface')

# Make the plot
#plt.bar(r1, dft, width=barWidth, label='DFT value')
plt.bar(r1, disagreement[0], ecolor='black', width=barWidth, label='Complete Training')
plt.bar(r2, disagreement[1], ecolor='black', width=barWidth, label='No 100 Training')
plt.bar(r3, disagreement[2], ecolor='black', width=barWidth, label='No 111 Training')
plt.bar(r4, disagreement[3], ecolor='black', width=barWidth, label='No Surface Training')
 
# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(1)], surfaces)

# Create legend & Show graphic
plt.yscale('log')
plt.grid(axis='y', which='major')
plt.grid(axis='y', which='minor', linewidth=0.5)
plt.legend()
plt.savefig('transferability.png')