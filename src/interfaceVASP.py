import os
from ase.io import vasp
from ase.io import lammpsdata
import configparser
import sys

binVasp = 'vasp_std'
elements = []
def readVASP(inputFile):
    global binVasp
    global elements
    # Read input data for the interface
    config = configparser.ConfigParser()
    config.read(inputFile)
    vars = config['VASP']
    for key in vars:
        if key == 'binvasp':
            try:
                binVasp = str(vars[key])
                # if not os.path.isfile(binVasp):
                #     print('Binary file for VASP could not be found: '+binVasp)
                #     raise Exception('File error')
            except:
                print('Invalid value for variable: ' +key)
        elif key == 'elements':
            try:
                elements = vars[key].split()
                print('elements found are:')
                print(elements)
            except:
                print('Invalid value for variable: ' +key)
        else:
            print('Invalid variable: '+key)
            sys.exit(1)

def compute(exec, training, numprocs):
    # Convert lammps data file to POSCAR file
    dftDir = 'DFT'
    os.makedirs(dftDir, exist_ok=True)
    computeDir = os.path.join(dftDir, 'dft'+str(training))
    os.system('cp -r vaspInput '+computeDir)
    lammps = lammpsdata.read_lammps_data('Restart/check.data',  style='atomic')
    
    #Count lammps types and assign elements
    ids = lammps.get_atomic_numbers()
    ids.sort()
    amounts = []
    t = ids[0]
    count = 0
    for i in range(len(ids)):
        if ids[i] == t:
                count += 1
        else:
                amounts.append(count)
                count = 1
                t = ids[i]
    amounts.append(count)
    if len(amounts) != len(elements):
        print('Number of elements is not the same than number of types in LAMMPS')
        sys.exit(1)
    symbols = ''
    for i in range(len(elements)):
        symbols += elements[i] + str(amounts[i])
    lammps.symbols = symbols

    os.chdir(computeDir)
    vasp.write_vasp('POSCAR', lammps)
    os.system(exec+' -n '+str(numprocs)+' '+binVasp+' > out-dft.txt')
    os.chdir('../..')