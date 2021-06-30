#!/usr/bin/python3
import subprocess
import os
import sys



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
        if status != 0:
            print(output)
        return status, output


project_dir = rootdir = r"./"


def main():
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.out') or '.' not in file:
                status, output = cmd('objcopy --remove-section .data {}'.format(file_path))
                status, output = cmd('objcopy --remove-section .bss {}'.format(file_path))
                status, output = cmd('objcopy --remove-section .ldata {}'.format(file_path))


if __name__ == '__main__':
    main()
