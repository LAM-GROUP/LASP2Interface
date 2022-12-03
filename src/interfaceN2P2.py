import numpy as np
import os
import configparser
import sys

#List of element names to obtain the element numbers
elementsAll = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No"
]
elements = []
elementNums = []

epochs = 100
binScaling = 'nnp-scaling'
binTraining = 'nnp-train'
def readN2P2(inputFile):
    global epochs
    global binScaling
    global binTraining
    # Read input data for the interface
    config = configparser.ConfigParser()
    config.read(inputFile)
    vars = config['N2P2']
    for key in vars:
        if key == 'epochslong':
            try:
                epochs = int(vars[key])
            except:
                print('Invalid value for variable: ' +key)
                sys.exit(1)
        elif key == 'binscaling':
            try:
                binScaling = str(vars[key])
                # if not os.path.isfile(binVasp):
                #     print('Binary file for VASP could not be found: '+binVasp)
                #     raise Exception('File error')
            except:
                print('Invalid value for variable: ' + key)
        elif key == 'bintraining':
            try:
                binTraining = str(vars[key])
                # if not os.path.isfile(binVasp):
                #     print('Binary file for VASP could not be found: '+binVasp)
                #     raise Exception('File error')
            except:
                print('Invalid value for variable: ' +key)
        else:
            print('Invalid variable: '+key)
            sys.exit(1)
    
    global elements
    global elementNums
    vars = config['VASP']
    for key in vars:
        if key == 'elements':
            try:
                elements = vars[key].split()
                for element in elements:
                    elementNums.append(elementsAll.index(element)+1)
            except:
                print('Invalid value for variable: ' + key)
                print('List of elements is not readable or elements do not match with known elements')

def convert(inputFile, outputFile):
    # n2p2 - A neural network potential package
    # Copyright (C) 2018 Andreas Singraber (University of Vienna)
    #
    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.
    #
    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.
    #
    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <https://www.gnu.org/licenses/>.

    ###############################################################################
    # File converter from VASP OUTCAR to input.data format.
    # Works also if OUTCAR contains trajectories.
    # Tested with VASP 5.2.12
    ###############################################################################

    file_name = inputFile
    outfile_name = outputFile

    # Read in the whole file first.
    f = open(file_name, "r")
    lines = [line for line in f]
    f.close()

    # If OUTCAR contains ionic movement run (e.g. from an MD simulation) multiple
    # configurations may be present. Thus, need to prepare empty lists.
    lattices   = []
    energies   = []
    atom_lists = []

    # Loop over all lines.
    elements = []
    for i in range(len(lines)):
        line = lines[i]
        # Collect element type information, expecting VRHFIN lines like this:
        #
        # VRHFIN =Cu: d10 p1
        #
        if "VRHFIN" in line:
            elements.append(line.split()[1].replace("=", "").replace(":", ""))
        # VASP specifies how many atoms of each element are present, e.g.
        #
        # ions per type =              48  96
        #
        if "ions per type" in line:
            atoms_per_element = [int(it) for it in line.split()[4:]]
        # Simulation box may be specified multiple times, I guess this line
        # introduces the final lattice vectors.
        if "VOLUME and BASIS-vectors are now" in line:
            lattices.append([lines[i+j].split()[0:3] for j in range(5, 8)])
        # Total energy is found in the line with "energy  without" (2 spaces) in
        # the column with sigma->0:
        #
        # energy  without entropy=     -526.738461  energy(sigma->0) =     -526.738365
        #
        if "energy  without entropy" in line:
            energies.append(line.split()[6])
        # Atomic coordinates and forces are found in the lines following
        # "POSITION" and "TOTAL-FORCE".
        if "POSITION" in line and "TOTAL-FORCE" in line:
            atom_lists.append([])
            count = 0
            for ei in range(len(atoms_per_element)):
                for j in range(atoms_per_element[ei]):
                    atom_line = lines[i+2+count]
                    atom_lists[-1].append(atom_line.split()[0:6])
                    atom_lists[-1][-1].extend([elements[ei]])
                    count += 1

    # Sanity check: do all lists have the same length. 
    if not (len(lattices) == len(energies) and len(energies) == len(atom_lists)):
        raise RuntimeError("ERROR: Inconsistent OUTCAR file.")

    f = open(outfile_name, "w")
    # Write configurations in "input.data" format.
    for i, (lattice, energy, atoms) in enumerate(zip(lattices, energies, atom_lists)):
        f.write("begin\n")
        f.write("comment source_file_name={0:s} structure_number={1:d}\n".format(file_name, i + 1))
        f.write("lattice {0:s} {1:s} {2:s}\n".format(lattice[0][0], lattice[0][1], lattice[0][2]))
        f.write("lattice {0:s} {1:s} {2:s}\n".format(lattice[1][0], lattice[1][1], lattice[1][2]))
        f.write("lattice {0:s} {1:s} {2:s}\n".format(lattice[2][0], lattice[2][1], lattice[2][2]))
        for a in atoms:
            f.write("atom {0:s} {1:s} {2:s} {3:s} {4:s} {5:s} {6:s} {7:s} {8:s}\n".format(a[0], a[1], a[2], a[6], "0.0", "0.0", a[3], a[4], a[5]))
        f.write("energy {0:s}\n".format(energy))
        f.write("charge {0:s}\n".format("0.0"))
        f.write("end\n")

def training(exec, trainingNum, numSeeds, numprocs):
    # Convert OUTCAR file to n2p2 data
    convert('DFT/dft'+str(trainingNum)+'/OUTCAR', 'Training/train.data')

    n = numprocs
    pathTrain = 'Training/nnp'+str(trainingNum)
    os.makedirs(pathTrain, exist_ok=True)
    os.system('cat Training/complete'+str(trainingNum-1)+'.data Training/train.data > Training/complete'+str(trainingNum)+'.data')
    for i in range(1, numSeeds+1):
        os.makedirs(pathTrain+'/Seed'+str(i), exist_ok=True)
        os.system('cp Training/Potentials/Seed'+str(i)+'/input.nn '+pathTrain+'/Seed'+str(i)+'/')
        os.system('cp Training/complete'+str(trainingNum)+'.data '+pathTrain+'/Seed'+str(i)+'/input.data')
        # for num in elementNums: # Copy previous weights for "use_old_weights_short" (disabled for now)
        #     os.system('cp Training/Potentials/Seed'+str(i)+'/weights.{:03d}.data '.format(num)+pathTrain+'/Seed'+str(i)+'/')
        os.chdir(pathTrain+'/Seed'+str(i))
        # Change number of epochs if it is defined
        epochsLine = os.popen("grep '^epochs' input.nn").read()[:-1]
        numEpochs = epochsLine.split()[1]
        if epochs != 0:
            epochsReplace = epochsLine.replace(str(numEpochs), str(epochs))
            os.system("sed -i 's/"+epochsLine+"/"+epochsReplace+"/g' input.nn")
            numEpochs = epochs
        # Change fraction of test set to 0
        testLine = os.popen("grep '^test_fraction' input.nn").read()[:-1]
        testFraction = testLine.split()[1]
        testReplace = testLine.replace(str(testFraction), str(0.0))
        os.system("sed -i 's/"+testLine+"/"+testReplace+"/g' input.nn")
        # # Add option to use old weights (disabled for now)
        # # Check if it has been added before
        # if os.popen("grep '^use_old_weights_short' input.nn").read() == '':
        #     os.system("sed -i '/^elements/ a use_old_weights_short             # Start the simulation using previous weights' input.nn")
        structures = int(os.popen("grep -c '^begin' input.data").read()[:-1])
        if structures < numprocs:
            n = structures
        os.system(exec+' -n '+str(n)+' '+binScaling+' 5000 > out-scaling.txt')
        os.system(exec+' -n '+str(n)+' '+binTraining+' > out-train.txt')
        os.chdir('../../..')

        # Copy last epochs of the training to the Potentials folder
        for num in elementNums:
            os.system('cp '+pathTrain+'/Seed'+str(i)+'/weights.{:03d}'.format(num)+'.{:06d}'.format(numEpochs)+'.out Training/Potentials/Seed'+str(i)+'/weights.079.data')
        os.system('cp '+pathTrain+'/Seed'+str(i)+'/input.nn Training/Potentials/Seed'+str(i)+'/')
        os.system('cp '+pathTrain+'/Seed'+str(i)+'/scaling.data Training/Potentials/Seed'+str(i)+'/')
        os.system('cp '+pathTrain+'/Seed'+str(i)+'/nnp-train.log.0000 Training/Potentials/Seed'+str(i)+'/')