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

func_call_cmd = "../../../pin -t obj-intel64/FunCallTrace.so -o {} -addrs_file {} -- {} {}"
# output_path, start_addr, end_addr, program, input_data
inst_trace_cmd = "timeout 20s ../../../pin -t obj-intel64/InstTrace.so -o {} -start {} -end {} -- {} {}"
mem_read_log_cmd = "../../../pin -t obj-intel64/MemoryRead.so -o {} -start {} -end {} -- {} {}"

compile_tool_cmd = "make obj-intel64/{}.so TARGET=intel64"
tools_list = ["InstTrace", "MemoryRead", "FunCallTrace"]


def compile_all_tools():
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    for tool_name in tools_list:
        status, output = cmd(compile_tool_cmd.format(tool_name))
        if status != 0:
            print(output)
    project_dir = project_dir_backup


# ==============================================================
# Instrumentation tools below
# ==============================================================


def func_call_trace(prog_path: str, input_data_path: str, addr_list: list, log_path: str):
    """ input addr_list: the addresses of all operator functions
        then use pin to instrument all these addresses
    """
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    addrs_file_path = './addrs_tmp.log'
    addrs_file_path = os.path.abspath(addrs_file_path)
    with open(addrs_file_path, 'w') as f:
        for addr in addr_list:
            f.write(addr + '\n')
        f.close()

    status, output = cmd(func_call_cmd.format(log_path, addrs_file_path, prog_path, input_data_path))

    status, output = cmd("rm {}".format(addrs_file_path))

    # ------- end reset project_dir
    project_dir = project_dir_backup


def mem_read_log(log_path: str, start_addr: str, end_addr: str, prog_path: str, data_path: str):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir

    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)
    status, output = cmd(mem_read_log_cmd.format(log_path, start_addr, end_addr, prog_path, data_path))
    if status != 0:
        print(output)

    project_dir = project_dir_backup


def inst_trace_log(log_path: str, start_addr: str, end_addr: str, prog_path: str, data_path: str):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir

    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)
    status, output = cmd(inst_trace_cmd.format(log_path, start_addr, end_addr, prog_path, data_path))
    if status != 0:
        print(output)

    project_dir = project_dir_backup


if __name__ == '__main__':
    print('tmp')
