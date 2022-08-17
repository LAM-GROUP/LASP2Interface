import os
from ase.io import vasp
from ase.io import lammpsdata
import configparser

binVasp = 'vasp_std'
def readVASP(inputFile):
    global binVasp
    # Read input data for the interface
    config = configparser.ConfigParser()
    config.read(inputFile)
    vars = config['VASP']
    for key in vars:
        if key == 'binvasp':
            try:
                binVasp = str(vars[key])
                if not os.path.isfile(binVasp):
                    print('Binary file for VASP could not be found: '+binVasp)
                    raise Exception('File error')
            except:
                print('Invalid value for variable: ' +key)

def setup_particle_types(frame, data):
    types = data.particles_.particle_types_
    types.type_by_id_(1).name = 'Au'

def compute(exec, training, numprocs):
    # Convert lammps data file to POSCAR file
    dftDir = 'DFT'
    os.makedirs(dftDir, exist_ok=True)
    computeDir = os.path.join(dftDir, 'dft'+str(training))
    os.system('cp -r vaspInput '+computeDir)
    lammps = lammpsdata.read_lammps_data('Restart/check.data',  style='atomic')
    lammps.symbols = 'Au'+str(len(lammps))
    os.chdir(computeDir)
    vasp.write_vasp('POSCAR', lammps)
    os.system(exec+' -n '+str(numprocs)+' '+binVasp+' > out-dft.txt')
    os.chdir('../..')