import os
import ovito

def setup_particle_types(frame, data):
        types = data.particles_.particle_types_
        types.type_by_id_(1).name = 'Au'

def compute(training):
    # Convert lammps data file to POSCAR file
    dftDir = 'DFT'
    os.makedirs(dftDir, exist_ok=True)
    computeDir = os.path.join(dftDir, 'dft'+str(training))
    os.system('cp -r vaspInput '+computeDir)
    os.makedirs(os.path.join(dftDir, 'dft'+str(training)), exist_ok=True)
    pipeline = ovito.io.import_file('check.data')
    pipeline.modifiers.append(setup_particle_types)
    ovito.io.export_file(pipeline, os.path.join(computeDir, 'POSCAR'), 'vasp')
    os.system('time srun $(placement ${SLURM_NTASKS_PER_NODE} 1 ) $binaire  > output_${SLURM_JOBID}')