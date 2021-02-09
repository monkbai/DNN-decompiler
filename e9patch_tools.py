#!/usr/bin/python3
from subprocess import Popen, PIPE, STDOUT
import os
import subprocess
import struct
import re

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


e9patch_dir = '/home/lifter/e9patch/'

e9tool_path = '/home/lifter/e9patch/e9tool'


def func_call_trace(prog_path: str, input_data_path: str, addr_list_str: str, log_path: str):
    """ input addr_list: the addresses of all operator functions
        then use e9patch to instrument all these addresses
    """
    global project_dir
    project_dir = e9patch_dir
    cmd_str = "./e9tool --match 'addr=={}' --action 'call entry(addr,instr,size,asm)@MyPrintRAX' {} --syntax=intel"
    status, output = cmd(cmd_str.format(addr_list_str, prog_path))

    # the output should be a.out under e9patch dir
    cmd_str = "./a.out {} > {}"
    status, output = cmd(cmd_str.format(input_data_path, log_path))


def all_inst_trace_1(prog_path: str, input_data_path: str, start_addr: str, end_addr: str, log_path :str):
    global project_dir
    project_dir = e9patch_dir
    # try to improve success rate of instrumentation
    cmd_str = "./e9tool --match 'addr >= {} and addr <= {}' --action 'call record_inst_mem(addr,&mem[0],asm)@MyPrintRAX' " \
              "--syntax=intel {}"

    status, output = cmd(cmd_str.format(start_addr, end_addr, prog_path))
    if status != 0:
        print(output)
        exit(-1)

    match = re.search('num_patched[ ]+= .*\n', output)
    print('warning:', match.group())

    # the output should be a.out under e9patch dir
    cmd_str = "./a.out {} > {}"
    status, output = cmd(cmd_str.format(input_data_path, log_path))
    if status != 0:
        print(output)
        exit(-1)


def all_inst_trace_2(prog_path: str, input_data_path: str, start_addr: str, end_addr: str):
    global project_dir
    project_dir = e9patch_dir
    tmp_log_1 = './temp_1.log'
    tmp_log_2 = './temp_2.log'
    """
    # the more parameters transferred to function, the more likely to failed instruct some instructions
    # thus record_inst_mem is deprecated
    cmd_str = "./e9tool --match 'true' --action 'call record_inst_mem(addr,&mem[0],asm)@MyPrintRAX' " \
              "--syntax=intel --start {} --end {} {}"
    """

    # print all instruction
    cmd_str = "./e9tool --match 'true' --action 'call print_inst(addr,asm)@MyPrintRAX' --syntax=intel " \
              "--start {} --end {} {}"
    status, output = cmd(cmd_str.format(start_addr, end_addr, prog_path))
    if status != 0:
        print(output)
        exit(-1)
    # the output should be a.out under e9patch dir
    cmd_str = "./a.out {} > {}"
    status, output = cmd(cmd_str.format(input_data_path, tmp_log_1))
    if status != 0:
        print(output)
        exit(-1)

    # then print all mem_addr
    cmd_str = "./e9tool --match 'true' --action 'call print_mem_addr(addr,&mem[0])@MyPrintRAX' --syntax=intel " \
              "--start {} --end {} {}"
    status, output = cmd(cmd_str.format(start_addr, end_addr, prog_path))
    if status != 0:
        print(output)
        exit(-1)
    # the output should be a.out under e9patch dir
    cmd_str = "./a.out {} > {}"
    status, output = cmd(cmd_str.format(input_data_path, tmp_log_2))
    if status != 0:
        print(output)
        exit(-1)


def arith_inst_trace(prog_path: str, input_data_path: str, start_addr: str, end_addr: str, log_path: str):
    """ arithmetic instructions trace """
    global project_dir
    project_dir = e9patch_dir
    # mul, add, max... and what?
    cmd_str = "./e9tool --match 'asm == mul.* or asm == max.* or asm==add.*' " \
              "--action 'call entry(addr,instr,size,asm)@MyPrintRAX' --start {} --end {} {} --syntax=intel"
    status, output = cmd(cmd_str.format(start_addr, end_addr, prog_path))
    if status != 0:
        print(output)
        exit(-1)

    # the output should be a.out under e9patch dir
    cmd_str = "./a.out {} > {}"
    status, output = cmd(cmd_str.format(input_data_path, log_path))
    if status != 0:
        print(output)
        exit(-1)


def dump_dwords(prog_path: str, input_data_path: str, inst_addr: str, mem_addr:str, dwords_len: int, log_path: str):
    global project_dir
    project_dir = e9patch_dir
    cmd_str = "./e9tool --match 'addr == {}' --action 'call print_dword({}, {})@MyPrintRAX' {} --syntax=intel"
    status, output = cmd(cmd_str.format(inst_addr, mem_addr, dwords_len, prog_path))

    # the output should be a.out under e9patch dir
    cmd_str = "./a.out {} > {}"
    status, output = cmd(cmd_str.format(input_data_path, log_path))


def get_reg_value(prog_path: str, input_data_path: str, inst_addr: str, reg_name: str):
    global project_dir
    project_dir = e9patch_dir
    cmd_str = "./e9tool --match 'addr == {}' --action 'call print_reg({})@MyPrintRAX' {} --syntax=intel"
    status, output = cmd(cmd_str.format(inst_addr, reg_name, prog_path))

    # the output should be a.out under e9patch dir
    cmd_str = "./a.out {}"
    status, output = cmd(cmd_str.format(input_data_path))

    if output.startswith('reg value'):
        reg_value = output.split('\n')[0]
        reg_value = reg_value[reg_value.find('0x'):].strip()
        return reg_value


def convert_dwords2float(log_path: str, float_len: int):
    def dw2fl(hex_str: str):
        return struct.unpack('!f', bytes.fromhex(hex_str))[0]
    log_txt = open(log_path, 'r').read()
    log_lines = log_txt.split('\n')[1:1+float_len]
    index = 0
    float_array = []
    while index < float_len:
        hex_str = log_lines[index].strip()
        if hex_str.startswith('0x'):
            hex_str = hex_str[2:]
        float_array.append(dw2fl(hex_str))
        index += 1
    return float_array


def test():
    # test func_call_trace
    prog = '/home/lifter/e9patch/test/demo_static_O0_exe'
    in_data = '/home/lifter/e9patch/test/number.bin'
    addr_list = '0x4018B0,0x401CA0,0x401FA0,0x402790,0x402EA0,0x403500,0x4035F0,0x403A50,0x405E20,0x406690,0x406DD0,0x407680,0x407B90,0x408D90,0x4093B0,0x4099C0,0x409E90,0x40A370'
    log_path = '/home/lifter/e9patch/log.txt'
    # func_call_trace(prog, in_data, addr_list, log_path)

    # test arithmetic instructions trace
    # arith_inst_trace(prog, in_data, '0x403A50', '0x405B52', log_path)

    # test dump dwords
    inst_addr = '0x403010'
    mem_addr = '0x63ac40'
    dwords_len = 200
    # dump_dwords(prog, in_data, inst_addr, mem_addr, dwords_len, log_path)

    # test get register value
    inst_addr = '0x4030D9'
    # reg_val = get_reg_value(prog, in_data, inst_addr, 'rsi')
    # print(reg_val)

    float_array = convert_dwords2float(log_path, 200)
    for i in range(40):
        j = i*5
        print('{} {} {} {} {}'.format(float_array[j], float_array[j+1], float_array[j+2], float_array[j+3], float_array[j+4]))


if __name__ == '__main__':
    test()
