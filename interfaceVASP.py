import os
from ase.io import vasp
from ase.io import lammpsdata

def setup_particle_types(frame, data):
        types = data.particles_.particle_types_
        types.type_by_id_(1).name = 'Au'

def compute(training):
    # Convert lammps data file to POSCAR file
    dftDir = 'DFT'
    os.makedirs(dftDir, exist_ok=True)
    computeDir = os.path.join(dftDir, 'dft'+str(training))
    os.system('cp -r vaspInput '+computeDir)
    lammps = lammpsdata.read_lammps_data('check.data',  style='atomic')
    lammps.symbols = 'Au144'
    os.chdir(computeDir)
    vasp.write_vasp('POSCAR', lammps)
    os.system('time srun $(placement ${SLURM_NTASKS_PER_NODE} 1 ) $binaire  > output_${SLURM_JOBID}')
    os.chdir('../..')