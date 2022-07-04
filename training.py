import numpy as np
import os
from conversionEnergy import convertFile


def training(potLoc):
    ########################## Test using train.lammpstrj
    convertFile('train.lammpstrj')
    ########################################################################
    path = 'shortTraining'
    print('Starting n2p2 training procedure')
    os.mkdir(path)
    os.system('cp -r trainingInput '+path)
    for i in range(1, 6):
        os.mkdir(path+'/Seed'+str(i))
        os.system('cp trainingInput/Seed'+str(i)+'/input.nn '+path+'/Seed'+str(i)+'/')
        os.system('cp train.data '+path+'/Seed'+str(i)+'/input.data')
        os.chdir(path+'/Seed'+str(i))
        # Change number of epochs
        epochsLine = os.popen("grep '^epochs' "+path+"/Seed"+str(i)+"/input.nn").read()[:-1]
        numEpochs = epochsLine.split()[1]
        epochsReplace = epochsLine.replace(str(numEpochs), str(10))
        os.system("sed -i 's/"+epochsLine+"/"+epochsReplace+"/g' "+path+"/Seed"+str(i)+"/input.nn")
        # Add option to use old weights
        # Check if it has been added before
        if os.popen("grep '^use_old_weights_short' "+path+"/Seed"+str(i)+"/input.nn").read() == '':
            os.system("sed -i '/^elements/ a use_old_weights_short             # Start the simulation using previous weights' "+path+"/Seed"+str(i)+"/input.nn")
        os.chdir('../..')
        