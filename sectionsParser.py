import os
import errno
import re

from matplotlib.pyplot import step

def load(fileName='Restart/sections.npy'):
    if not os.path.isfile(fileName):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fileName)
        return
    f = open(fileName, 'r')
    lines = f.readlines()
    f.close()
    sections = []
    reading = False
    step = []
    disagreement = []
    for i in range(len(lines)):
        words = lines[i].split()
        if len(words) == 0:
            continue
        if re.match('^#', words[0]):
            continue
        if words[0] == 'ITERATION:':
            if reading:
                sections.append([step, disagreement])
                step = []
                disagreement = []
            else:
                reading = True
            continue
        if reading:
            if len(words) > 2:
                raise Exception('Error when reading file in line: '+str(i))
                return
            step.append(words[0])
            disagreement.append(words[1])
    sections.append([step, disagreement])
    return sections

def save(sections, fileName='sections.out', nameSim='LASP2 Simulation'):
    f = open(fileName, 'w')
    lines = []
    lines.append('# '+nameSim)
    lines.append('# This file contains the value of the disagreement measured throughout the simulation')
    lines.append('# and it is information necessary for restarting the LAMMPS simulation after each training')
    lines.append('# It is formatted as ...')
    lines.append('# ITERATION: N')
    lines.append('# step    disagreement')
    for i in range(len(sections)):
        lines.append('ITERATION: '+str(i+1))
        for b in  range(len(sections[0])):
            lines.append(str(sections[i][0][b])+'    '+str(sections[i][1][b]))