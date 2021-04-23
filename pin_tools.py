#!/usr/bin/python3
from subprocess import Popen, PIPE, STDOUT
import os
import time
import subprocess
import struct
import re


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


project_dir = './'

pin_home = '/home/lifter/pin-3.14-98223-gb010a12c6-gcc-linux'

mypintool_dir = '/home/lifter/pin-3.14-98223-gb010a12c6-gcc-linux/source/tools/MyPinTool/'

# output_path, start_addr, end_addr, program, input_data
inst_trace_cmd = "timeout 10s ../../../pin -t obj-intel64/InstTrace.so -o {} -start {} -end {} -- {} {}"
mem_read_log_cmd = "../../../pin -t obj-intel64/MemoryRead.so -o {} -start {} -end {} -- {} {}"

compile_tool_cmd = "make obj-intel64/{}.so TARGET=intel64"
tools_list = ["InstTrace", "MemoryRead"]


def compile_all_tools():
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    for tool_name in tools_list:
        status, output = cmd(compile_tool_cmd.format(tool_name))
        if status != 0:
            print(output)
    project_dir = project_dir_backup


def mem_read_log(log_path: str, start_addr: str, end_addr: str, prog_path: str, data_path: str):
    global project_dir
    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)
    status, output = cmd(mem_read_log_cmd.format(log_path, start_addr, end_addr, prog_path, data_path))
    if status != 0:
        print(output)


def inst_trace_log(log_path: str, start_addr: str, end_addr: str, prog_path: str, data_path: str):
    global project_dir
    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)
    status, output = cmd(inst_trace_cmd.format(log_path, start_addr, end_addr, prog_path, data_path))
    if status != 0:
        print(output)


if __name__ == '__main__':
    print('tmp')
