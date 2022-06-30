from mpi4py import MPI
import lammps
import numpy as np
import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()



# Python script to perform a simulation with nnp potentials and check agreement
potentialSeed = 1
dataDir = 'PhysicalPropertiesTemplate/Configurations/Au100Messy.data'
disagreement = []

threshold = 0.02001143776

totalSteps = 1000
checkEvery = 100
checkSteps = int(totalSteps / checkEvery)

lmp = lammps.lammps()
lmp.command('units metal')
lmp.command('boundary p p p')
lmp.command('read_data '+dataDir)

lmp.command('pair_style hdnnp 6.0 dir PotentialsComplete/Seed'+str(potentialSeed)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0')
lmp.command('pair_coeff * * Au')

lmp.file('sim.melt')

lmp.command('run 100')
f = lmp.numpy.extract_atom('f')
print(len(f))
#time.sleep(rank*3)
#print(f)