import os
import errno
import re

def load(fileName='Restart/sections.out'):
    if not os.path.isfile(fileName):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fileName)
        return
    f = open(fileName, 'r')
    lines = f.readlines()
    f.close()
    sections = []
    reading = False
    step = []
    dispersion = []
    for i in range(len(lines)):
        words = lines[i].split()
        if len(words) == 0:
            continue
        if re.match('^#', words[0]):
            continue
        if words[0] == 'ITERATION:':
            if reading:
                sections.append([step, dispersion])
                step = []
                dispersion = []
            else:
                reading = True
            continue
        if reading:
            if len(words) > 2:
                raise Exception('Error when reading file in line: '+str(i))
                return
            step.append(int(words[0]))
            dispersion.append(float(words[1]))
    sections.append([step, dispersion])
    return sections

def save(sections, fileName='sections.out', nameSim='LASP2 Simulation', threshold='', totalSteps='', checkEvery=''):
    f = open(fileName, 'w')
    lines = []
    lines.append('# '+nameSim)
    lines.append('# This file contains the value of the dispersion measured throughout the simulation')
    lines.append('# and it is information necessary for restarting the LAMMPS simulation after each training')
    lines.append('# ----------------------------------------------------------------------------------------')
    lines.append('# threshold = '+threshold)
    lines.append('# total number of steps = '+totalSteps)
    lines.append('# steps between dispersion checks = '+checkEvery)
    lines.append('# ----------------------------------------------------------------------------------------')
    lines.append('# The format is as follows ...')
    lines.append('# ITERATION: N')
    lines.append('# step    dispersion')
    lines.append('')
    for i in range(len(sections)):
        lines.append('ITERATION: '+str(i+1))
        for b in  range(len(sections[i][0])):
            lines.append(str(sections[i][0][b])+'    '+str(sections[i][1][b]))
    f.write('\n'.join(lines))
    f.close()