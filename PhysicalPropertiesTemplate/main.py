#Python script for computing the energy, lattice spacing and elastic constant
#for different configurations
#Files need to be in .data format inside a Configurations/ directory
#The potentials are obtained from N2P2 and must be located in
# a Potentials/Seed# directory

import numpy as np
from os import listdir
from os.path import isfile
from os.path import exists
from os import makedirs
from os import system
from os import getcwd
from os import chdir
import compliance

def findFirstOcurrence(strMatch, linesArray):
    for b in range(len(linesArray)): # Iterate through the lines and find last ocurrence
        if linesArray[b] != None and strMatch in linesArray[b]:
            return b

def findLastOcurrence(strMatch, linesArray):
    for b in reversed(range(len(linesArray))): # Iterate through the lines and find first ocurrence
        if linesArray[b] != None and strMatch in linesArray[b]:
            return b

###############################################################################################################################
# Minimization for the different systems using potentials with different random seeds
confDir = 'Configurations/'
files = listdir(confDir)

originalSize = 12.513864
changesBox = []

# First, perform the minimization of the bulks to obtain change in lattice parameters
for f in files:
    if not f[-5:] == '.data':
        continue
    for i in range(1, 6):
        folderName = f[:-5] + 'seed' + str(i)
        if not exists(confDir + folderName + '/'):
            if '100' in folderName or '111' in folderName:
                continue
            makedirs(confDir + folderName, exist_ok=True) #Create a folder with the same name of the file to save all files for the physical parameters
            print('Calculating physical properties for ' + f)
            system('cp ' + confDir+f + ' '+confDir+folderName+'/IN.data')
            if 'Bulk' in folderName:
                system('cp in.elastic '+confDir+folderName)
                system('cp displace.mod '+confDir+folderName)
                system('cp init.mod '+confDir+folderName)
                system('cp potential.mod '+confDir+folderName)
            chdir(confDir+folderName)
            system('/home/carlos/Desktop/lammps-stable_29Oct2020/build/lmp -i in.elastic -var seed '+str(i)+' > elastic.out')
            # Obtain change of box in x and y
            dump = open('dump.lammpstrj')
            linesDump = dump.readlines()
            dump.close()
            posLastBox = findLastOcurrence('ITEM: BOX BOUNDS', linesDump)
            latticeB = []
            for b in range(1, 4):
                items = str.split(linesDump[posLastBox+b])
                latticeB.append(float(items[1]) - float(items[0]))
            change = []
            for b in range(len(latticeB)):
                change.append((latticeB[b] - originalSize)/originalSize)
            changesBox.append(change)
            chdir('../..')
        else:
            print('Folder for ' + f + str(i) + ' already exists, skipping')
            if 'Bulk' not in folderName:
                continue
            chdir(confDir+folderName)
            # Obtain change of box in x and y
            dump = open('dump.lammpstrj')
            linesDump = dump.readlines()
            dump.close()
            posLastBox = findLastOcurrence('ITEM: BOX BOUNDS', linesDump)
            latticeB = []
            for b in range(1, 4):
                items = str.split(linesDump[posLastBox+b])
                latticeB.append(float(items[1]) - float(items[0]))
            change = []
            for b in range(len(latticeB)):
                change.append((latticeB[b] - originalSize)/originalSize)
            changesBox.append(change)
            chdir('../..')
            continue

# Perform minimization of surfaces and rescale by change in bulk
for f in files:
    if not f[-5:] == '.data':
        continue
    for i in range(1, 6):
        folderName = f[:-5] + 'seed' + str(i)
        if not exists(confDir + folderName + '/'):
            if 'Bulk' in folderName:
                continue
            makedirs(confDir + folderName, exist_ok=True) #Create a folder with the same name of the file to save all files for the physical parameters
            print('Calculating physical properties for ' + f)
            system('cp ' + confDir+f + ' '+confDir+folderName+'/IN.data')
            varLattice = ''
            if '100' in folderName:
                system('cp in.surface '+confDir+folderName)
                varLattice = ' -var xX ' + str(1.0+changesBox[i-1][0]) + ' -var yY ' + str(1.0+changesBox[i-1][1])  + ' -var zZ ' + str(1.0+changesBox[i-1][2])
            elif '111' in folderName:
                system('cp in.surface '+confDir+folderName)
                varLattice = ' -var xX ' + str(1.0+changesBox[i-1][0]) + ' -var yY ' + str(1.0+changesBox[i-1][1])  + ' -var zZ ' + str(1.0+changesBox[i-1][2])
            chdir(confDir+folderName)
            system('/home/carlos/Desktop/lammps-stable_29Oct2020/build/lmp -i in.surface -var seed '+str(i)+varLattice+' > elastic.out')
            chdir('../..')
        else:
            print('Folder for ' + f + str(i) + ' already exists, skipping')
            continue


#########################################################################################################################33
# Reading the data from the output files of the minimization
energy = dict()
lattice = dict()
c = []
s = []
bulkModulus = []
poissonRatio = []
shearModulus1 = []
shearModulus2 = []
energy['bulk'] = []
energy['100'] = []
energy['111'] = []
lattice['bulk'] = []
lattice['100'] = []
lattice['111'] = []
sType = 'bulk'
for f in files:
    if not f[-5:] == '.data':
        continue
    for i in range(1, 6):
        folderName = f[:-5] + 'seed' + str(i)
        if exists(confDir + folderName + '/'):
            # Check wether surface or bulk is being read
            if '100' in folderName:
                sType = '100'
            elif '111' in folderName:
                sType = '111'
            elif 'Bulk' in folderName:
                sType = 'bulk'
            # Go to minimization folder
            chdir(confDir+folderName)
            # Get compliance and stiffness tensors
            if sType == 'bulk':
                comp, stif = compliance.constant()
                c.append(stif)
                s.append(comp)
                
            # Read output file
            file1 = open('elastic.out', 'r')
            lines = file1.readlines()
            file1.close
            
            # Get poisson ratio, bulk modulus and shear modulus
            if sType == 'bulk':
                for b in reversed(range(len(lines))):
                    if 'Bulk Modulus' in lines[b]:
                        items = str.split(lines[b])
                        bulkModulus.append(float(items[3]))
                        items = str.split(lines[b+1])
                        shearModulus1.append(float(items[4]))
                        items = str.split(lines[b+2])
                        shearModulus2.append(float(items[4]))
                        items = str.split(lines[b+3])
                        poissonRatio.append(float(items[3]))
                        break
                        
            # Get last energy from the minimization process
            posLastEnergy = findLastOcurrence('Energy initial', lines)
            items = str.split(lines[posLastEnergy+1])
            energy[sType].append(float(items[2]))
            
            # Read lattice spacing from dump file
            file1 = open('dump.lammpstrj', 'r')
            lines = file1.readlines()
            file1.close
            posLastBox = findLastOcurrence('ITEM: BOX BOUNDS', lines)
            coords = []
            for b in range(1, 4):
                items = str.split(lines[posLastBox+b])
                coords.append(float(items[1])-float(items[0]))
            lattice[sType].append(coords)
            
            # Go back to parent directory
            chdir('../..')


##################################################################################################################
# Calculate average and standard deviation and save to file
enerMean = dict()
enerStd = dict()
latticeMean = dict()
latticeStd = dict()

file2 = open('physicalParams.out', 'w')
lines = []
for key in energy:
    enerMean[key] = np.mean(energy[key])
    enerStd[key] = np.std(energy[key])
    if key == 'bulk':
        sMean = np.mean(s, axis=0)
        sStd = np.std(s, axis=0)
        cMean = np.mean(c, axis=0)
        cStd = np.std(c, axis=0)
        bulkMMean = np.mean(bulkModulus)
        bulkMStd = np.std(bulkModulus)
        shearM1Mean = np.mean(shearModulus1)
        shearM1Std = np.std(shearModulus1)
        shearM2Mean = np.mean(shearModulus2)
        shearM2Std = np.std(shearModulus2)
        poissonRMean = np.mean(poissonRatio)
        poissonRStd = np.std(poissonRatio)
        lattice[key] = np.array(lattice[key]) / 3.0
    latticeMean[key] = np.mean(lattice[key], axis=0)
    latticeStd[key] = np.std(lattice[key], axis=0)
    
    lines.append('---------------------------------------')
    lines.append('PROPERTIES '+key)
    if key == 'bulk':
        lines.append('Compliance tensor [1/GPa]:')
        for i in range(6):
            for b in range(6):
                lines.append('S['+str(i+1)+']['+str(b+1)+']' + str(sMean[i][b]) + ' +/- ' + str(sStd[i][b]))
        lines.append('')

        lines.append('Stiffness tensor [GPa]:')
        for i in range(6):
            for b in range(6):
                lines.append('C['+str(i+1)+']['+str(b+1)+']' + str(cMean[i][b]) + ' +/- ' + str(cStd[i][b]))
        lines.append('')
        
        lines.append('Bulk modulus [GPa]:')
        lines.append(str(bulkMMean) + ' +/- ' + str(bulkMStd))
        lines.append('')
        
        lines.append('Shear modulus 1 [GPa]:')
        lines.append(str(shearM1Mean) + ' +/- ' + str(shearM1Std))
        lines.append('')
        
        lines.append('Shear modulus 2 [GPa]:')
        lines.append(str(shearM2Mean) + ' +/- ' + str(shearM2Std))
        lines.append('')
        
        lines.append('Poisson ratio []:')
        lines.append(str(poissonRMean) + ' +/- ' + str(poissonRStd))
        lines.append('')

    lines.append('Lattice spacing [Angst]: x y z')
    lines.append(str(latticeMean[key][0])+ ' +/- ' + str(latticeStd[key][0]))
    lines.append(str(latticeMean[key][1])+ ' +/- ' + str(latticeStd[key][1]))
    lines.append(str(latticeMean[key][2])+ ' +/- ' + str(latticeStd[key][2]))
    lines.append('')
    
    lines.append('Energy [eV]:')
    lines.append(str(enerMean[key]) + ' +/- ' + str(enerStd[key]))
    lines.append('---------------------------------------')
    lines.append('')

# Surface energies
lines.append('SURFACE ENERGIES [eV/Angst^2]')
#100
e100 = np.array(energy['100'])
ebulk = np.array(energy['bulk'])/108.
xlat100 = np.array([i[0] for i in lattice['100']])
ylat100 = np.array([i[1] for i in lattice['100']])
ener100 = (e100 - 144.*ebulk) / (2 * xlat100 * ylat100)
lines.append('Surface 100 energy: ' + str(np.mean(ener100)) + ' +/- ' + str(np.std(ener100)))
#111
e111 = np.array(energy['111'])
ebulk = np.array(energy['bulk'])/108.
xlat111 = np.array([i[0] for i in lattice['111']])
ylat111 = np.array([i[1] for i in lattice['111']])
ener111 = (e111 - 180.*ebulk) / (2 * xlat111 * ylat111)
lines.append('Surface 111 energy: ' + str(np.mean(ener111)) + ' +/- ' + str(np.std(ener111)))
lines.append('')

# Surface energies
lines.append('SURFACE ENERGIES [mJ/m^2]')
#100
lines.append('Surface 100 energy: ' + str(16021.8*np.mean(ener100)) + ' +/- ' + str(16021.8*np.std(ener100)))
#111
lines.append('Surface 111 energy: ' + str(16021.8*np.mean(ener111)) + ' +/- ' + str(16021.8*np.std(ener111)))

file2.write('\n'.join(lines))
file2.close()
print('Results have been saved to: physicalParams.out')
