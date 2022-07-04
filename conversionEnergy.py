def convertFile(fileName, folder = '', newFolder = False, fileNameOutput = None):
    if fileNameOutput == None:
        fileNameOutput = fileName[:-9] + 'data' #Assuming the fileName has termination .lammpstrj
    file1 = open(folder + fileName, 'r')
    lines = file1.readlines() #Obtain all lines from the file
    file1.close()

    #Parameters used for N2P2 file
    atomPositions = []
    atomForces = []
    atomId = []
    atomType = []
    numberAtoms = 0
    timeStep = -1 #Timestep that is being read right now
    simStep = [] #Steps of each configuration
    boxBounds = []
    data = '' #Type of data being read
    numBox = 0
    numAtoms = 0

    #Iterate through the lines and get their parameters
    for i in range(len(lines)):
        if lines[i] != None and 'ITEM:' in lines[i]: #Get type of data being read
            items = str.split(lines[i])
            if 'TIMESTEP' in lines[i]:
                data = 'TIMESTEP'
            elif 'BOX BOUNDS' in lines[i]:
                data = 'BOX BOUNDS'
                numBox = 0
                items = str.split(lines[i])
                if len(items) > 6:
                    print('This script is not yet able to handle triclinic simulation boxes')
                    return
            elif 'NUMBER OF ATOMS' in lines[i]:
                data = 'NUMBER OF ATOMS'
            elif 'ATOMS' in lines[i]:
                data = 'ATOMS'
                numAtoms = 0
            else:
                data = ''
                print('Data category not recognized in line ' + str(i))
        else:
            if data == 'TIMESTEP':
                timeStep += 1
                atomPositions.append([])
                atomForces.append([])
                boxBounds.append([])
                atomId.append([])
                atomType.append([])
                try:
                    simStep.append(int(lines[i]))
                except:
                    print('Error when obtaining the timestep of the configuration in line ' + str(i))
            elif data == 'BOX BOUNDS':
                items = str.split(lines[i])
                try:
                    vals = [float(i) for i in items]
                except:
                    print('Error when converting text to float in line ' + str(i))
                    return
                if len(vals) > 2:
                    print('This script is not yet able to handle triclinic simulation boxes')
                    return
                boxBounds[timeStep].append([])
                boxBounds[timeStep][numBox].append(vals[0])
                boxBounds[timeStep][numBox].append(vals[1])
                numBox += 1
            elif data == 'ATOMS':
                items = str.split(lines[i])
                try:
                    vals = [float(i) for i in items]
                except:
                    print('Error when converting text to float in line ' + str(i))
                    return
                if len(vals) > 8:
                    print('Wrong number of values in line ' + str(i))
                    return
                atomId[timeStep].append(vals[0])
                atomType[timeStep].append(vals[1])
                atomPositions[timeStep].append([])
                atomPositions[timeStep][numAtoms].append(vals[2])
                atomPositions[timeStep][numAtoms].append(vals[3])
                atomPositions[timeStep][numAtoms].append(vals[4])
                atomForces[timeStep].append([])
                atomForces[timeStep][numAtoms].append(vals[5])
                atomForces[timeStep][numAtoms].append(vals[6])
                atomForces[timeStep][numAtoms].append(vals[7])
                numAtoms += 1
            elif data == 'NUMBER OF ATOMS':
                try:
                    numberAtoms = int(lines[i])
                except:
                    print('Error when converting number of atoms to int in line ' + str(i))

    #Iterate through the data and convert it to N2P2 format
    linesSave = []
    for i in range(timeStep+1):
        linesSave.append('begin')
        linesSave.append('comment Configuration from file ' + fileName + ' and TIMESTEP: ' + str(simStep[i]))
        #Add the lattice values
        linesSave.append('lattice {:.16E}'.format(boxBounds[i][0][1] - boxBounds[i][0][0]) + ' {:.16E}'.format(0) + ' {:.16E}'.format(0))
        linesSave.append('lattice {:.16E}'.format(0) + ' {:.16E}'.format(boxBounds[i][1][1] - boxBounds[i][1][0]) + ' {:.16E}'.format(0))
        linesSave.append('lattice {:.16E}'.format(0) + ' {:.16E}'.format(0) + ' {:.16E}'.format(boxBounds[i][2][1] - boxBounds[i][2][0]))
        for b in range(len(atomPositions[timeStep])):
            atomLine = 'atom {:.16E}'.format(atomPositions[i][b][0]) + ' {:.16E}'.format(atomPositions[i][b][1]) + ' {:.16E}'.format(atomPositions[i][b][2])
            if atomType[timeStep][b] == 2:
                atomLine += ' Au '
            else:
                print('Unknown atom type for atom with id: ' + str(atomId[timeStep][b]))
                return
            atomLine += '{:.16E}'.format(0) + ' {:.16E}'.format(0) #Values that are not used
            atomLine += ' {:.16E}'.format(atomForces[i][b][0]) + ' {:.16E}'.format(atomForces[i][b][1]) + ' {:.16E}'.format(atomForces[i][b][2])
            linesSave.append(atomLine)
        #Energy is now read from OUTCAR files
        file3 = open(folder + fileName[:-10]+'.outcar', 'r')
        enerLines = file3.readlines()
        file3.close()
        lastEnergy = 0
        for s in range(len(enerLines)):
            if 'without' in enerLines[s]:
                lastEnergy = s
        items = str.split(enerLines[lastEnergy])
        print(fileName)
        print(items)
        linesSave.append('energy {:.16E}'.format(float(items[4]))) #Energy is ignored as training will be only with forces
        linesSave.append('charge {:.16E}'.format(0)) #The system has a neutral charge
        linesSave.append('end')
    if newFolder:
        file2 = open('ConvertedNew/' + fileNameOutput, 'w')
    else:
        file2 = open(fileNameOutput, 'w')
    file2.write('\n'.join(linesSave))
    file2.write('\n')
    file2.close()
