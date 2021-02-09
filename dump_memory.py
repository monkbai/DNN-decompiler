#!/usr/bin/python3
from subprocess import Popen, PIPE, STDOUT
import os
import subprocess

project_dir = './'


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


def run(prog_path):
    with cd(project_dir):
        # print(prog_path)
        proc = subprocess.Popen(prog_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()  # stderr: summary, stdout:  each statement
        return stdout, stderr


with cd('/home/lifter/e9patch/test'):
    p = Popen(['gdb', '--args', './demo_static_O0_exe', './number.bin'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    # p.stdin.write('b *0x403010'.encode())
    # p.stdin.flush()
    # while p.stdout.readable():
    #     print(p.stdout.readline())

    grep_stdout = p.communicate(input='b *0x403010\nr\nx/20wx 0x63ac40'.encode())[0]
    print(bytes.decode(grep_stdout))

# not convenient
# turn to e9path
