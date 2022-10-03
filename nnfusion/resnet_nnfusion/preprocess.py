#!/usr/bin/python3
import os
import re
import time
import struct
import subprocess
import logging


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
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


project_dir = './'


def main(prog_path: str):
    prog_path = os.path.abspath(prog_path)
    status, output = cmd('readelf -d {}'.format(prog_path))
    match = re.search('Library runpath: \[(.*)\]', output)
    rpath = match.group(1)
    print(rpath)

    status, output = cmd('chrpath -r "./" {}'.format(prog_path))  # change the rpath to ./


if __name__ == '__main__':
    main('/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion_resnet/resnet50_nnfusion')
