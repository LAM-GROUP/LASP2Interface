import numpy as np
import os

def training(potLoc, trainingNum):
    # Convert OUTCAR file to n2p2 data
    os.system('python3 convert-VASP_OUTCAR.py DFT/dft'+str(trainingNum)+'/OUTCAR train.data')

    pathShort = potLoc+'short'+str(trainingNum)
    print('Starting n2p2 training procedure')
    os.makedirs(pathShort, exist_ok=True)
    for i in range(1, 6):
        os.makedirs(pathShort+'/Seed'+str(i), exist_ok=True)
        os.system('cp trainingInput/Seed'+str(i)+'/input.nn '+pathShort+'/Seed'+str(i)+'/')
        os.system('cp train.data '+pathShort+'/Seed'+str(i)+'/input.data')
        os.system('cp PotentialsComplete/Seed'+str(i)+'/weights.079.data '+pathShort+'/Seed'+str(i)+'/')
        os.chdir(pathShort+'/Seed'+str(i))
        # Change number of epochs
        epochsLine = os.popen("grep '^epochs' input.nn").read()[:-1]
        numEpochs = epochsLine.split()[1]
        epochsReplace = epochsLine.replace(str(numEpochs), str(10))
        os.system("sed -i 's/"+epochsLine+"/"+epochsReplace+"/g' input.nn")
        # Add option to use old weights
        # Check if it has been added before
        if os.popen("grep '^use_old_weights_short' input.nn").read() == '':
            os.system("sed -i '/^elements/ a use_old_weights_short             # Start the simulation using previous weights' input.nn")
        print('Starting scaling')
        os.system('nnp-scaling 5000 > scaling.out')
        print('Starting training')
        os.system('nnp-train > training.out')
        os.chdir('../../..')
    
    pathLong = potLoc+'long'+str(trainingNum)
    print('Starting n2p2 long training procedure')
    os.makedirs(pathLong, exist_ok=True)
    os.makedirs(potLoc+'Potentials', exist_ok=True)
    for i in range(1, 6):
        os.makedirs(pathLong+'/Seed'+str(i), exist_ok=True)
        os.makedirs(potLoc+'Potentials/Seed'+str(i), exist_ok=True)
        os.system('cp trainingInput/Seed'+str(i)+'/input.nn '+pathLong+'/Seed'+str(i)+'/')
        os.system('cat completeinput.data train.data > '+pathLong+'/Seed'+str(i)+'/input.data')
        os.system('cp '+pathShort+'/Seed'+str(i)+'/weights.079.000010.out '+pathLong+'/Seed'+str(i)+'/weights.079.data')
        os.chdir(pathLong+'/Seed'+str(i))
        # Change number of epochs
        epochsLine = os.popen("grep '^epochs' input.nn").read()[:-1]
        numEpochs = epochsLine.split()[1]
        epochsReplace = epochsLine.replace(str(numEpochs), str(50))
        os.system("sed -i 's/"+epochsLine+"/"+epochsReplace+"/g' input.nn")
        # Add option to use old weights
        # Check if it has been added before
        if os.popen("grep '^use_old_weights_short' input.nn").read() == '':
            os.system("sed -i '/^elements/ a use_old_weights_short             # Start the simulation using previous weights' input.nn")
        print('Starting scaling')
        os.system('nnp-scaling 5000 > scaling.out')
        print('Starting training')
        os.system('mpirun -n 16 nnp-train > training.out')
        os.chdir('../../..')
        os.system('cp '+pathLong+'/Seed'+str(i)+'/weights.079.000050.out '+potLoc+'Potentials/Seed'+str(i)+'/weights.079.data')
        os.system('cp '+pathLong+'/Seed'+str(i)+'/input.nn '+potLoc+'Potentials/Seed'+str(i)+'/')
        os.system('cp '+pathLong+'/Seed'+str(i)+'/scaling.data '+potLoc+'Potentials/Seed'+str(i)+'/')
        os.system('cp '+pathLong+'/Seed'+str(i)+'/nnp-train.log.0000 '+potLoc+'Potentials/Seed'+str(i)+'/')