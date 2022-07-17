# LASP2 Interface

This repository provides an interface for 'on the flight' training of a high dimensional neural network. This software is an interface between three programs ([n2p2](https://github.com/CompPhysVienna/n2p2), [LAMMPS](https://www.lammps.org/), and [VASP](https://www.vasp.at/)). The LAMMPS software is used for the dynamics of the simulation, using a potential created with the n2p2 software for high dimensional neural network potentials. VASP is the reference method with which the potentials are trained.

# Basic usage

LASP2 requires as input configuration files for n2p2, LAMMPS and VASP, as well as a own LASP2 configuration file. The n2p2 input that is required consists of:
- The *n2p2Input* directory containing subdirectories for each of the different seeds used. The subdirectories must contain the *input.nn* and *input.data* files used for training.
- The *PotentialsComplete* directory contanining subdirectories for each of the different seeds used. The subdirectories must contain the files necessary to load an hdnnp potential in LAMMPS.
- A *completeinput.data* file with the complete database used for training.

For LAMMPS, the required input consists of:
- An input file named *input.lmp* containing the commands to run the desired simulation.
- A second input file named *restart.lmp* with instructions that need to be defined again after restarting the simulation. The simulation will stop when training is needed, and it will be restarted after the neural network potential is trained again.
- The structure used as starting point for the simulation.

In order to use VASP, the interface requires the following input files:
- A directory named *vaspInput* containing the *INCAR*, *KPOINTS*, and *POTCAR* files, as well as a copy of the VASP binary.

### Output

When the simulation is finished, the interface will produce a plot of the disagreement between the potentials with different random seeds as a function of the steps of the simulation. The output of the VASP calculations will be stored in a directory named *DFT*, numbered by the number of computations made throughout the computation. Similarly, the output of n2p2 training can be found in the directory named *Training*, containing subdirectories for the short and long trainings numbered also by the number of trainings performed throughout the simulation. The last potentials used in the simulation can be found in a subdirectory named *Potentials* under the *Training* directory.