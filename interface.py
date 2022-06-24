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

totalSteps = 1000
checkEvery = 100
checkSteps = int(totalSteps / checkEvery)

print(sys.argv)
print('')
lmp = lammps.lammps()
lmp.command('units metal')
lmp.command('boundary p p p')
lmp.command('read_data '+dataDir)

lmp.command('pair_style nnp dir PotentialsComplete/Seed'+str(potentialSeed)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"')
lmp.command('pair_coeff * * 6.01')

lmp.file('sim.melt')

# me = MPI.COMM_WORLD.Get_rank()
# nprocs = MPI.COMM_WORLD.Get_size()
# print("Proc %d out of %d procs has" % (me,nprocs),lmp)
#MPI.Finalize()

for a in range(checkSteps):
    lmp.command('run '+str(checkEvery))

    lmp.command('write_data check.data')
    sorting('check.data')

    devs = []

    for i in range(10):
        seeds = [None]*5
        forces = [None]*5
        for b in range(5):
            seeds[b] = lammps.lammps()
            seeds[b].command('units metal')
            seeds[b].command('boundary p p p')
            seeds[b].command('read_data sorted.data')

            seeds[b].command('pair_style nnp dir PotentialsComplete/Seed'+str(b+1)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0 emap "2:Au"')
            seeds[b].command('pair_coeff * * 6.01')
            seeds[b].command('run 0')
            forces[b] = seeds[b].numpy.extract_atom('f')
        #print(forces)
        deviationAtom = np.std(np.ma.masked_invalid(forces), axis=0)
        #print(deviationAtom)
        masked = np.ma.masked_greater(np.ma.masked_invalid(deviationAtom), 1000)
        deviationMax = masked.max()
        devs.append(deviationMax)
        print(deviationMax)
        # for s in range(len(deviationAtom)):
        #     if np.isnan(deviationAtom[s]).any() or np.isinf(deviationAtom[s]).any():
        #         print(s)
        #         for l in range(5):
        #             print(forces[l][s])

    ave.append(np.mean(devs))
    errAve.append(np.std(devs))
    # ave[a].append(np.ma.masked_invalid(devs).mean())
    # errAve[a].append(np.ma.masked_invalid(devs).std())

ave = np.array(ave)
errAve = np.array(errAve)

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Dark2.colors)
fig, ax1 = plt.subplots()

ax1.set_title('Evolution (100) Au (T = 1000)')
ax1.set_ylabel('Disagreement')
ax1.set_xlabel('Timestep')

x = (np.arange(len(ave))+1)*checkEvery

ax1.errorbar(x, ave, yerr=errAve, label='Disagreement', fmt=':', alpha=0.75, capsize=5.0, color='C0')
ax1.fill_between(x, ave-errAve, ave+errAve, alpha=0.25, color='C0')

ax1.legend()
plt.savefig('timeAgreement.png')
# MPI.Finalize()