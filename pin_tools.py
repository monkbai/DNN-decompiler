#!/usr/bin/python3
from subprocess import Popen, PIPE, STDOUT
from scripts import config

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

pin_home = config.pin_home

mypintool_dir = config.pintool_dir

fun_call_rdi_rsi_cmd = pin_home + "pin -t " + \
                       mypintool_dir + "obj-intel64/FunCallRdiRsi.so -o {} -addrs_file {} -- {} {}"
fused_rdi_cmd = pin_home + "pin -t " + \
                mypintool_dir + "obj-intel64/FusedRdi.so -o {} -addrs_file {} -- {} {}"
func_call_cmd = pin_home + "pin -t " + \
                mypintool_dir + "obj-intel64/FunCallTrace.so -o {} -addrs_file {} -- {} {}"
# output_path, start_addr, end_addr, program, input_data
inst_trace_cmd = "time timeout 15s " + \
                 pin_home + "pin -t " + \
                 mypintool_dir + "obj-intel64/InstTrace.so -o {} -start {} -end {} -- {} {}"
mem_read_log_cmd = pin_home + "pin -t " + \
                   mypintool_dir + "obj-intel64/MemoryRead.so -o {} -start {} -end {} -- {} {}"
mem_write_log_cmd = pin_home + "pin -t " + \
                    mypintool_dir + "obj-intel64/MemoryWrite.so -o {} -start {} -end {} -- {} {}"
mem_dump_log_cmd = pin_home + "pin -t " + \
                   mypintool_dir + "obj-intel64/MemoryDump.so -o {} -length {} -dump_point {} -reg_num {} -- {} {}"
mem_dump_2_log_cmd = pin_home + "pin -t " + \
                     mypintool_dir + "obj-intel64/MemoryDump_2.so -o {} -length {} -dump_point {} -data_index {} -- {} {}"
mem_dump_3_log_cmd = pin_home + "pin -t " + \
                     mypintool_dir + "obj-intel64/MemoryDump_3.so -o {} -length {} -dump_point {} -dump_addr {} -- {} {}"

nnfusion_conv_cmd = pin_home + "pin -t " + \
                    mypintool_dir + "obj-intel64/NNFusion_Conv.so" \
                                    " -o {} -addrs_file {} -- {} {}"
nnfusion_gemm_cmd = pin_home + "pin -t " + \
                    mypintool_dir + "obj-intel64/NNFusion_Gemm.so" \
                                    " -o {} -addrs_file {} -- {} {}"
nnfusion_pool_cmd = pin_home + "pin -t " + \
                    mypintool_dir + "obj-intel64/NNFusion_Pool.so" \
                                    " -o {} -addrs_file {} -- {} {}"
nnfusion_trace_cmd = pin_home + "pin -t " + \
                     mypintool_dir + "obj-intel64/NNFusion_Trace.so" \
                                     " -o {} -addrs_file {} -- {} {}"

compile_tool_cmd = "make obj-intel64/{}.so TARGET=intel64"
tools_list = ["InstTrace", "MemoryRead", "FunCallTrace", "MemoryWrite",
              "MemoryDump", "MemoryDump_2", "MemoryDump_3", "FusedRdi", "FunCallRdiRsi",
              "NNFusion_Conv", "NNFusion_Gemm", "NNFusion_Pool", "NNFusion_Trace", "LocateData"]


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

# For GLOW
def fun_call_rdi_rsi(prog_path: str, input_data_path: str, addr_list: list, log_path: str):
    global project_dir

    prog_path = os.path.abspath(prog_path)
    input_data_path = os.path.abspath(input_data_path)
    log_path = os.path.abspath(log_path)

    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    addrs_file_path = './addrs_rdi_rsi_tmp.log'
    addrs_file_path = os.path.abspath(addrs_file_path)
    with open(addrs_file_path, 'w') as f:
        for addr in addr_list:
            f.write(addr + '\n')
        f.close()

    status, output = cmd(fun_call_rdi_rsi_cmd.format(log_path, addrs_file_path, prog_path, input_data_path))

    status, output = cmd("rm {}".format(addrs_file_path))

    # ------- end reset project_dir
    project_dir = project_dir_backup


# For TVM
def fused_rdi(prog_path: str, input_data_path: str, addr_list: list, log_path: str):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir
    # ------- set project_dir before instrumentation

    addrs_file_path = './addrs_fused_tmp.log'
    addrs_file_path = os.path.abspath(addrs_file_path)
    with open(addrs_file_path, 'w') as f:
        for addr in addr_list:
            f.write(addr + '\n')
        f.close()

    status, output = cmd(fused_rdi_cmd.format(log_path, addrs_file_path, prog_path, input_data_path))

    status, output = cmd("rm {}".format(addrs_file_path))

    # ------- end reset project_dir
    project_dir = project_dir_backup


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

    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)

    project_dir_backup = project_dir
    project_dir = mypintool_dir

    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)
    status, output = cmd(mem_read_log_cmd.format(log_path, start_addr, end_addr, prog_path, data_path))
    if status != 0:
        print(output)

    project_dir = project_dir_backup


def mem_write_log(log_path: str, start_addr: str, end_addr: str, prog_path: str, data_path: str):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir

    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)
    status, output = cmd(mem_write_log_cmd.format(log_path, start_addr, end_addr, prog_path, data_path))
    if status != 0:
        print(output)

    project_dir = project_dir_backup


def inst_trace_log(log_path: str, start_addr: str, end_addr: str, prog_path: str, data_path: str):
    global project_dir, logger
    project_dir_backup = project_dir
    project_dir = mypintool_dir

    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)
    status, output = cmd(inst_trace_cmd.format(log_path, start_addr, end_addr, prog_path, data_path))
    if status != 0:
        print(output)

    logger.info('Trace Logging Time')
    logger.info(output)

    project_dir = project_dir_backup


def dump_dwords(prog_path: str, input_data_path: str, inst_addr: str, dwords_len: int, log_path: str, reg_num=0):
    global project_dir
    project_dir_backup = project_dir
    project_dir = os.path.dirname(prog_path)  # project_dir = mypintool_dir

    status, output = cmd(mem_dump_log_cmd.format(log_path, dwords_len, inst_addr, reg_num, prog_path, input_data_path))
    # print(output)
    if status != 0:
        print(output)

    project_dir = project_dir_backup


def dump_dwords_2(prog_path: str, input_data_path: str, inst_addr: str, dwords_len: int, log_path: str, data_index=1):
    global project_dir
    project_dir_backup = project_dir
    project_dir = os.path.dirname(prog_path)  # project_dir = mypintool_dir

    status, output = cmd(
        mem_dump_2_log_cmd.format(log_path, dwords_len, inst_addr, data_index, prog_path, input_data_path))
    # print(output)
    if status != 0:
        print(output)

    project_dir = project_dir_backup


def dump_dwords_3(prog_path: str, input_data_path: str, inst_addr: str, dwords_len: int, log_path: str, dump_addr: str):
    global project_dir
    project_dir_backup = project_dir
    project_dir = mypintool_dir

    status, output = cmd(
        mem_dump_3_log_cmd.format(log_path, dwords_len, inst_addr, dump_addr, prog_path, input_data_path))
    # print(output)
    if status != 0:
        print(output)

    project_dir = project_dir_backup


def rm_log(log_path: str):
    status, output = cmd("rm {}".format(log_path))
    # print(output)
    if status != 0:
        print(output)


def convert_dwords2float(dwords_txt: str, float_len: int):
    def dw2fl(hex_str: str):
        return struct.unpack('!f', bytes.fromhex(hex_str))[0]

    dwords_lines = dwords_txt.split('\n')
    index = 0
    float_array = []
    while index < float_len:
        hex_str = dwords_lines[index].strip()
        if hex_str.startswith('0x') and len(hex_str) > 0:
            hex_str = hex_str[2:]
            float_array.append(dw2fl(hex_str))
        index += 1
    return float_array


# ==================================================================
def nnfusion_cmd(prog_path: str, input_data_path: str, addr_list: list, log_path: str, cmdline: str):
    global project_dir
    project_dir_backup = project_dir
    project_dir = os.path.dirname(prog_path)
    # ------- set project_dir before instrumentation

    addrs_file_path = './addrs_tmp.log'
    addrs_file_path = os.path.abspath(addrs_file_path)
    with open(addrs_file_path, 'w') as f:
        for addr in addr_list:
            f.write(addr + '\n')
        f.close()

    status, output = cmd(cmdline.format(log_path, addrs_file_path, prog_path, input_data_path))

    status, output = cmd("rm {}".format(addrs_file_path))

    # ------- end reset project_dir
    project_dir = project_dir_backup


def nnfusion_conv(prog_path: str, input_data_path: str, addr_list: list, log_path: str):
    nnfusion_cmd(prog_path, input_data_path, addr_list, log_path, nnfusion_conv_cmd)


def nnfusion_gemm(prog_path: str, input_data_path: str, addr_list: list, log_path: str):
    nnfusion_cmd(prog_path, input_data_path, addr_list, log_path, nnfusion_gemm_cmd)


def nnfusion_pool(prog_path: str, input_data_path: str, addr_list: list, log_path: str):
    nnfusion_cmd(prog_path, input_data_path, addr_list, log_path, nnfusion_pool_cmd)


def nnfusion_trace(prog_path: str, input_data_path: str, addr_list: list, log_path: str):
    nnfusion_cmd(prog_path, input_data_path, addr_list, log_path, nnfusion_trace_cmd)


# ==================================================================
def tac_cmd(log_path: str, new_path: str):
    global logger

    log_path = os.path.abspath(log_path)
    new_path = os.path.abspath(new_path)
    status, output = cmd("tac {} > {}".format(log_path, new_path))
    if status:
        pass  # TODO: error log
    logger.info('Reverse Trace - Time')
    logger.info(output)


if __name__ == '__main__':
    print('tmp')
    compile_all_tools()
