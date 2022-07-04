from mpi4py import MPI
from lammps import PyLammps

lmp = PyLammps()
# Training variables
potentialSeed = 1 #Potential used for main simulation
dataDir = 'PhysicalPropertiesTemplate/Configurations/Au100Messy.data' #Location of the structure being simulated
disagreement = []

threshold = 0.02001143776 #Disagreement value that will activate the training flag

totalSteps = 1000 #Total number of steps to be simulated
checkEvery = 100 #Number of steps after which the agreement will be measured
checkSteps = int(totalSteps / checkEvery) #Number of times the agreement will be measured

# Main LAMMPS object to carry on the simulation
lmp.command('units metal')
lmp.command('boundary p p p')
lmp.command('read_data '+dataDir)

# Setup n2p2 potential
lmp.command('pair_style hdnnp 6.0 dir PotentialsComplete/Seed'+str(potentialSeed)+' showew no showewsum 100 resetew no maxew 10000000 cflength 1.0 cfenergy 1.0')
lmp.command('pair_coeff * * Au')

# Read commands for the simulation to be performed
lmp.file('sim.melt')
print('file was imported')
lmp.run(0)

if MPI.COMM_WORLD.rank == 0:
    print(len(lmp.atoms))

MPI.Finalize()