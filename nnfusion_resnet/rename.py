#!/usr/bin/python3
import os
import re
import time
import struct
import subprocess
import logging

print('get logger: {}'.format('decompiler.' + __name__))
logger = logging.getLogger('decompiler.' + __name__)


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def cmd(commandline):
    with cd(project_dir):
        print(commandline)
        logger.debug(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


project_dir = './'


def main():
    files = os.listdir(project_dir)
    for f in files:
        if f.startswith('0') and f.endswith('.json'):
            tmp = f.split('.')
            new_f = tmp[0] + '.' + tmp[2] + '.' + tmp[3]
            new_f = new_f.replace('param-1', 'gamma')
            new_f = new_f.replace('param0', 'beta')
            new_f = new_f.replace('biases', 'mean')
            new_f = new_f.replace('param3', 'var')

            statsu, output = cmd('mv {} {} '.format(f, new_f))


if __name__ == '__main__':
    main()
