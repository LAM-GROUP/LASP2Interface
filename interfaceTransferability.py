import sys
import lammps
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def sorting(filename):
    f1 = open(filename)
    infile = f1.readlines()
    startAtoms = 0
    numAtoms = 0
    for i in range(len(infile)):
        if 'atom types' in infile[i]:
            numAtoms = int(infile[i-1].split()[0])
        if 'Atoms # atomic' in infile[i]:
            startAtoms = i+2
    output = sorted(infile[startAtoms:startAtoms+numAtoms], key=lambda x: int(x.split()[0]))
    outfile = open("sorted.data", "w")
    outfile.writelines(infile[:startAtoms] + output)
    outfile.close()

ave = []
errAve = []

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

lmp.command('pair_style nnp dir '+locationSeeds[0]+'/Seed'+str(potentialSeed)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"')
lmp.command('pair_coeff * * 6.01')

lmp.file('sim.melt')
lmp.command('run 10')

#lmp.command('run 0')
# me = MPI.COMM_WORLD.Get_rank()
# nprocs = MPI.COMM_WORLD.Get_size()
# MPI.Finalize()

#lmp.command('write_dump              all custom check.lammpstrj id type x y z fx fy fz modify sort id')
lmp.command('write_data check.data')
sorting('check.data')
# trj = read_lammps_data('check.data', sort_by_id=True, style='atomic')
# write_lammps_data('check copy.data', trj)
# print(trj.get_chemical_symbols())

for a in range(len(locationSeeds)):
    ave.append([])
    errAve.append([])
    devs = []

    for i in range(10):
        seeds = [None]*5
        forces = [None]*5
        for b in range(len(seeds)):
            seeds[b] = lammps.lammps()
            seeds[b].command('units metal')
            seeds[b].command('boundary p p p')
            seeds[b].command('read_data sorted.data')

            seeds[b].command('pair_style nnp dir '+locationSeeds[a]+'/Seed'+str(b+1)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"')
            seeds[b].command('pair_coeff * * 6.01')
            seeds[b].command('run 0')
            forces[b] = seeds[b].numpy.extract_atom('f')
        #print(forces)
        deviationAtom = np.std(np.ma.masked_invalid(forces), axis=0)
        #print(deviationAtom)
        deviationMax = np.ma.masked_invalid(deviationAtom).std()
        devs.append(deviationMax)
        print(deviationMax)
        # for s in range(len(deviationAtom)):
        #     if np.isnan(deviationAtom[s]).any() or np.isinf(deviationAtom[s]).any():
        #         print(s)
        #         for l in range(5):
        #             print(forces[l][s])

    ave[a].append(np.mean(devs))
    errAve[a].append(np.std(devs))
    # ave[a].append(np.ma.masked_invalid(devs).mean())
    # errAve[a].append(np.ma.masked_invalid(devs).std())

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
plt.bar(r1, ave[0], yerr=errAve[0], ecolor='black', capsize=10, width=barWidth, label='Complete Training')
plt.bar(r2, ave[1], yerr=errAve[1], ecolor='black', capsize=10, width=barWidth, label='No 100 Training')
plt.bar(r3, ave[2], yerr=errAve[2], ecolor='black', capsize=10, width=barWidth, label='No 111 Training')
plt.bar(r4, ave[3], yerr=errAve[3], ecolor='black', capsize=10, width=barWidth, label='No Surface Training')
 
# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(1)], surfaces)

 
# Create legend & Show graphic
#plt.yscale('log')
plt.grid(axis='y', which='major')
plt.grid(axis='y', which='minor', linewidth=0.5)
plt.legend()
plt.savefig('transferability.png')