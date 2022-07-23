import numpy as np
import os

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

def training(potLoc, trainingNum):
    # Convert OUTCAR file to n2p2 data
    convert('DFT/dft'+str(trainingNum)+'/OUTCAR', 'Training/train.data')

    pathTrain = potLoc+'nnp'+str(trainingNum)
    print('Starting n2p2 training procedure')
    os.makedirs(pathTrain, exist_ok=True)
    os.system('cat Training/complete'+str(trainingNum-1)+'.data Training/train.data > Training/complete'+str(trainingNum)+'.data')
    for i in range(1, 6):
        os.makedirs(pathTrain+'/Seed'+str(i), exist_ok=True)
        os.system('cp trainingInput/Seed'+str(i)+'/input.nn '+pathTrain+'/Seed'+str(i)+'/')
        os.system('cp Training/complete'+str(trainingNum)+'.data '+pathTrain+'/Seed'+str(i)+'/input.data')
        os.system('cp PotentialsComplete/Seed'+str(i)+'/weights.079.data '+pathTrain+'/Seed'+str(i)+'/')
        os.chdir(pathTrain+'/Seed'+str(i))
        # Change number of epochs
        epochsLine = os.popen("grep '^epochs' input.nn").read()[:-1]
        numEpochs = epochsLine.split()[1]
        epochsReplace = epochsLine.replace(str(numEpochs), str(50))
        os.system("sed -i 's/"+epochsLine+"/"+epochsReplace+"/g' input.nn")
        # Change fraction of test set to 0
        testLine = os.popen("grep '^test_fraction' input.nn").read()[:-1]
        testFraction = testLine.split()[1]
        testReplace = testLine.replace(str(testFraction), str(0.0))
        os.system("sed -i 's/"+testLine+"/"+testReplace+"/g' input.nn")
        # Add option to use old weights
        # Check if it has been added before
        if os.popen("grep '^use_old_weights_short' input.nn").read() == '':
            os.system("sed -i '/^elements/ a use_old_weights_short             # Start the simulation using previous weights' input.nn")
        print('Starting scaling')
        os.system('time srun $(placement ${SLURM_NTASKS_PER_NODE} 1 ) nnp-scaling 5000 > out-scaling.txt')
        print('Starting training')
        os.system('time srun $(placement ${SLURM_NTASKS_PER_NODE} 1 ) nnp-train > out-train.txt')
        os.chdir('../../..')
    
        os.system('cp '+pathTrain+'/Seed'+str(i)+'/weights.079.000050.out '+potLoc+'Potentials/Seed'+str(i)+'/weights.079.data')
        os.system('cp '+pathTrain+'/Seed'+str(i)+'/input.nn '+potLoc+'Potentials/Seed'+str(i)+'/')
        os.system('cp '+pathTrain+'/Seed'+str(i)+'/scaling.data '+potLoc+'Potentials/Seed'+str(i)+'/')
        os.system('cp '+pathTrain+'/Seed'+str(i)+'/nnp-train.log.0000 '+potLoc+'Potentials/Seed'+str(i)+'/')