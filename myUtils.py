import re
import collections
import time
import os
import math
from scripts.e9patch_tools import all_inst_trace_2, all_inst_trace_1
import scripts.explain


# /* ===================================================================== */
# // Utilities
# /* ===================================================================== */
def get_range(asm_path: str):
    f = open(asm_path, 'r')
    asm_txt = f.read()
    f.close()
    asm_lines = asm_txt.split('\n')
    start_line = 0
    end_line = len(asm_lines) - 1
    while asm_lines[start_line].startswith(';') or len(asm_lines[start_line]) < 1:
        start_line += 1
    while asm_lines[end_line].startswith(';') or len(asm_lines[end_line]) < 1:
        end_line -= 1
    start_line = asm_lines[start_line]
    end_line = asm_lines[end_line]
    start_addr = start_line[:start_line.find(':')]
    end_addr = end_line[:end_line.find(':')]
    return start_addr, end_addr


def get_asm_line(f_b_line: str):
    line = f_b_line
    if not line.startswith('0x'):
        return None

    asm_addr, asm_code = line.split(':')
    asm_code = asm_code.strip()
    asm_addr = asm_addr.strip()
    return asm_addr, asm_code


def parse_asm_line(asm_line: str):
    blank_1 = asm_line.find(' ')
    if blank_1 == -1:
        return [asm_line]
    mnemonic = asm_line[:blank_1]
    operand_str = asm_line[blank_1:]
    operands = operand_str.split(',')
    code_list = [mnemonic.strip()]
    for op in operands:
        code_list.append(op.strip())
    return code_list


def parse_mem_line(mem_line: str):
    if mem_line.startswith('N'):
        return '0', '0'
    mem_addr = mem_line.split(':')[1].strip()
    mem_addr = int(mem_addr, 16)
    mem_addr = hex(mem_addr)
    mem_size = mem_line.split(':')[2].strip()
    return mem_addr, mem_size


def parse_three_lines(log_line: str, mem_line: str, mem_value_line: str):
    # is the end?
    if not log_line.startswith('0x') or not (mem_line[0] in 'RWN'):  # Read, Write, NoMem
        return

    # handle different instructions
    asm_addr, asm_line = get_asm_line(log_line)
    code_list = parse_asm_line(asm_line)
    mnemonic = code_list[0]
    mem_addr, mem_size = parse_mem_line(mem_line)
    if len(mem_size) == 0:
        mem_size = '0'
    mem_value = mem_value_line.split(':')[1].strip()
    return asm_addr, code_list, mnemonic, mem_addr, int(mem_size), mem_value


# /* ===================================================================== */
# // Registers/Memory Model
# /* ===================================================================== */
# -----------------------------------------------
# taint analysis
# write result(mem_state) to log file
xmm_regs = {
            # 128 bits
            'xmm0': '0', 'xmm1': '0', 'xmm2': '0', 'xmm3': '0',
            'xmm4': '0', 'xmm5': '0', 'xmm6': '0', 'xmm7': '0',
            'xmm8': '0', 'xmm9': '0', 'xmm10': '0', 'xmm11': '0',
            'xmm12': '0', 'xmm13': '0', 'xmm14': '0', 'xmm15': '0',
            # 256 bits
            'ymm0': '0', 'ymm1': '0', 'ymm2': '0', 'ymm3': '0',
            'ymm4': '0', 'ymm5': '0', 'ymm6': '0', 'ymm7': '0',
            'ymm8': '0', 'ymm9': '0', 'ymm10': '0', 'ymm11': '0',
            'ymm12': '0', 'ymm13': '0', 'ymm14': '0', 'ymm15': '0',
            }
xmm_reg_backup = xmm_regs
mem_state = {}  # collections.OrderedDict()  # slow
"""
# we only care about registers related to mov instructions
# if one register is used in 'mov' instruction, record its state
# if one register is used in arithmetic instruction (inc, mul, add, ...), remove its state
"""
reg_state = {'ah': '', 'ch': '', 'dh': '', 'bh': '',
             'al': '', 'cl': '', 'dl': '', 'bl': '', 'spl': '', 'bpl': '', 'sil': '', 'dil': '',
             'r8b': '', 'r9b': '', 'r10b': '', 'r11b': '', 'r12b': '', 'r13b': '', 'r14b': '', 'r15b': '',
             'ax': '', 'cx': '', 'dx': '', 'bx': '', 'sp': '', 'bp': '', 'si': '', 'di': '',
             'r8w': '', 'r9w': '', 'r10w': '', 'r11w': '', 'r12w': '', 'r13w': '', 'r14w': '', 'r15w': '',
             'eax': '', 'ecx': '', 'edx' : '', 'ebx' : '', 'esp' : '', 'ebp' : '', 'esi' : '', 'edi' : '',
             'r8d': '', 'r9d': '', 'r10d': '', 'r11d': '', 'r12d': '', 'r13d': '', 'r14d': '', 'r15d': '',
             'rax': '', 'rcx': '', 'rdx' : '', 'rbx' : '', 'rsp' : '', 'rbp' : '', 'rsi' : '', 'rdi' : '',
             'r8' : '', 'r9' : '', 'r10' : '', 'r11' : '', 'r12' : '', 'r13' : '', 'r14' : '', 'r15' : '', }
reg_backup = reg_state
# How to automatically know which external function is called ?
# Read the asm file to record external functions and their address
extern_functions = {'0x400cb0': 'memset', '0x400c00': 'expf'}  # we need to simulate memset if it is used to clean


# /* ===================================================================== */
# // Symbolic Execution Engine
# /* ===================================================================== */
def record_ext_funcs(func_asm_path: str):
    with open(func_asm_path, 'r') as f:
        extern_functions.clear()
        asm_txt = f.read()
        f.close()
        lines = asm_txt.split('\n')
        for line in lines:
            if not line.startswith('0x'):
                continue
            line = line[line.find('\t'):]
            line = line.strip()
            if line.startswith('call'):
                op_list = line.split(' ')
                if len(op_list) > 2:
                    ext_addr = op_list[1]
                    ext_name = op_list[2].strip('<>')
                    extern_functions[ext_addr] = ext_name


def clear_mem_state():
    global mem_state, xmm_regs, reg_state
    mem_state.clear()
    xmm_regs = xmm_reg_backup
    reg_state = reg_backup


""" Represent each memory block with its address """


def lightweight_SymEx(func_asm_path: str, log_file: str, exp_log_path: str, max_inst_num: int):
    clear_mem_state()
    record_ext_funcs(func_asm_path)

    f = open(log_file, 'r')
    log_txt = f.read()
    f.close()
    log_lines = log_txt.split('\n')

    start_time = time.time()

    index = 0
    debug_count = 0
    log_length = len(log_lines)
    while index < log_length-2:
        # print('line {}'.format(index))  # debug
        if index >= max_inst_num:
            # print('debug')
            break

        # read three lines of log
        log_line = log_lines[index]
        index += 1
        while log_line.startswith('RAX'):
            log_line = log_lines[index]
            index += 1
        mem_line = log_lines[index]
        index += 1
        while mem_line.startswith('  '):
            mem_line = log_lines[index]
            index += 1
        mem_value_line = log_lines[index]
        index += 1
        asm_addr, code_list, mnemonic, mem_addr, mem_size, mem_value = parse_three_lines(log_line, mem_line, mem_value_line)
        if mnemonic == 'call' and log_lines[index].startswith('RAX'):
            rax_line = log_lines[index]
            rax_value = rax_line.split(':')[1].strip()
            index += 1
        elif mnemonic == 'call':
            rax_value = ''

        # debug
        if asm_addr == '0x401ccd':
            debug_count += 1
            print('debug')
        #if asm_addr == '0x402ff2':
        #    print('debug')
        #if asm_addr == '0x4297dd':
        #    print('debug')
        #if '0x22f700a0,4' in mem_state.keys():
        #    print('debug')

        # TODO: should we update the mem_value of the address to be read?
        if len(mem_value) > 0 and mem_size == 8 and mem_addr.startswith('0x7ff'):
            set_mem(mem_value, mem_addr, mem_size)

        # ymm register related instructions
        if mnemonic.startswith('vmovups'):
            handle_movaps(code_list, mem_addr)  # TODO how to handle ymm register
        elif mnemonic.startswith('vmovss'):
            handle_movss(code_list, mem_addr)
        elif mnemonic.startswith('vfmadd213ss'):
            handle_vfmadd_ss(code_list, mem_addr)
        elif mnemonic.startswith('vfmadd231ss'):  # vfmadd231ss
            handle_vfmadd_ss(code_list, mem_addr)
        elif mnemonic.startswith('vfmadd213ps'):  # TODO: vfmadd213ps
            handle_vfmadd_ss(code_list, mem_addr)
        elif mnemonic.startswith('vfmadd231ps'):  # TODO: vfmadd231ps
            handle_vfmadd_ss(code_list, mem_addr)
        elif mnemonic.startswith('vmulss'):
            handle_vmulss(code_list, mem_addr)
        elif mnemonic.startswith('vaddss'):
            handle_vaddss(code_list, mem_addr)
        elif mnemonic.startswith('vmaxss'):
            handle_vmaxss(code_list, mem_addr)
        elif mnemonic.startswith('vmaxps'):
            handle_vmaxss(code_list, mem_addr)
        elif mnemonic.startswith('vxorps'):
            handle_xorps(code_list, mem_addr)
        elif mnemonic.startswith('vbroadcastss'):
            handle_vbroadcastss(code_list, mem_addr)
        elif mnemonic.startswith('vzeroupper'):
            pass
        # xmm register related instructions
        elif mnemonic.startswith('movaps') or mnemonic.startswith('movups') or mnemonic.startswith('movdqa'):
            handle_movaps(code_list, mem_addr)
        elif mnemonic.startswith('movd'):
            handle_movd(code_list, mem_addr)
        elif mnemonic.startswith('movq'):
            handle_movq(code_list, mem_addr)
        elif mnemonic.startswith('movss'):
            handle_movss(code_list, mem_addr)
        elif mnemonic.startswith('movsd'):
            handle_movss(code_list, mem_addr)
        elif mnemonic.startswith('mulps'):
            handle_mulps(code_list, mem_addr)
        elif mnemonic.startswith('divps'):
            handle_divss(code_list, mem_addr)
        elif mnemonic.startswith('divss'):
            handle_divss(code_list, mem_addr)
        elif mnemonic.startswith('addps'):
            handle_addps(code_list, mem_addr)
        elif mnemonic.startswith('addss'):
            handle_addss(code_list, mem_addr)
        elif mnemonic.startswith('subss'):
            handle_subss(code_list, mem_addr)
        elif mnemonic.startswith('maxps'):
            handle_maxps(code_list, mem_addr)
        elif mnemonic.startswith('maxss'):
            handle_maxss(code_list, mem_addr)
        elif mnemonic.startswith('xorps'):
            handle_xorps(code_list, mem_addr)
        elif mnemonic.startswith('xorss'):
            print('not implemented')
        elif mnemonic.startswith('unpcklps'):
            handle_unpcklp(code_list, mem_addr)
        elif mnemonic.startswith('unpcklpd'):
            handle_unpcklp(code_list, mem_addr)
        elif mnemonic.startswith('unpckhpd'):
            handle_unpckhp(code_list, mem_addr)
        elif mnemonic.startswith('unpckhps'):
            handle_unpckhp(code_list, mem_addr)
        elif mnemonic.startswith('punpckhdq'):
            handle_punpckhdq(code_list, mem_addr)
        elif mnemonic.startswith('movlhps'):
            handle_mov_ps(code_list, mem_addr)  # TODO
        elif mnemonic.startswith('movhlps'):
            handle_mov_ps(code_list, mem_addr)
        elif mnemonic.startswith('shufps'):
            pass  # it is hard to model, to simplify the implementation, skip it
        elif mnemonic.startswith('pxor'):
            pass
        elif mnemonic.startswith('por'):
            pass
        elif mnemonic.startswith('pshufd'):
            pass
        # regular registers
        elif mnemonic.startswith('mov'):
            handle_mov(code_list, mem_addr)
        elif mnemonic.startswith('lea'):
            handle_lea(code_list, mem_addr)
        elif mnemonic.startswith('call'):
            if code_list[1] in extern_functions.keys():
                if 'memset' == extern_functions[code_list[1]]:
                    handle_memset(code_list, rax_value)  # how to handle the function call
                elif 'expf' == extern_functions[code_list[1]]:
                    handle_expf(code_list, rax_value)
                else:
                    print('call not implemented: {}'.format(extern_functions[code_list[1]]))
                    # TODO: even if the function is unknown, we should get the RAX value
                    handle_unknown_call(code_list, rax_value)
            else:
                print('call not implemented')
                handle_unknown_call(code_list, rax_value)
        elif mnemonic == 'add' or mnemonic.startswith('sub') or \
                mnemonic.startswith('idiv') or mnemonic.startswith('imul') or \
                mnemonic.startswith('xor') or mnemonic.startswith('inc') or \
                mnemonic.startswith('shr') or mnemonic.startswith('sar') or \
                mnemonic.startswith('shl') or mnemonic.startswith('or') or \
                mnemonic.startswith('not') or mnemonic.startswith('dec') or \
                mnemonic.startswith('and') or mnemonic.startswith('sete'):
            handle_arith(code_list, mem_addr)
        elif mnemonic.startswith('j') or mnemonic.startswith('cmp') or \
                mnemonic.startswith('push') or mnemonic.startswith('pop') or mnemonic.startswith('test'):
            pass
        elif mnemonic.startswith('nop') or mnemonic.startswith('cdq') or mnemonic.startswith('cmov'):
            pass
        elif mnemonic.startswith('sete') or mnemonic.startswith('setz') or \
                mnemonic.startswith('setb') or mnemonic.startswith('setnbe'):
            pass
        elif mnemonic.startswith('data16'):
            pass
        else:
            if len(code_list) > 2 and ('[' in code_list[1] or '[' in code_list[2]):
                print(log_line)
            else:
                print(log_line)  # ret,

    # show the result

    end_time = time.time()
    duration = end_time - start_time
    print('time consumed: {}s'.format(duration))

    all_mem_key = mem_state.keys()
    all_mem_key = sorted(all_mem_key)
    with open(exp_log_path, 'w') as f:
        for key in all_mem_key:
            if key.startswith('0x7ff'):  # TODO: should we ignore all stack addresses?
                continue
            # print(key)
            # print(mem_state[key])
            f.write(key+'\n')
            f.write(mem_state[key]+'\n')
        f.flush()
        f.close()
    print('debug count {}'.format(debug_count))


# /* ===================================================================== */
# // Operations directly on the Reg/Mem Model
# /* ===================================================================== */
# -----------------------------------------------
# write global dictionaries: xmm_regs, mem_state, reg_state

def key2addr(mem_key: str):
    addr, size = mem_key.split(',')
    size = int(size)
    return addr, size


def remove_overlap_mem(mem_addr: str, size: int, set_zero=False):
    global mem_state
    mem_start = int(mem_addr, 16)
    mem_end = mem_start + size
    size_list = [4, 8, 16, 32]  # dwrod, qword, xmmword, ymmword
    overlap_key_list = []
    for key_start in range(mem_start, mem_end, 4):
        for key_size in size_list:
            key_end = key_start + key_size
            if key_end <= mem_end:
                key = hex(key_start)+','+str(key_size)
                if key in mem_state.keys():
                    overlap_key_list.append(key)
    """
    # too slow
    for key, value in mem_state.items():
        key_addr, key_size = key2addr(key)
        key_start = int(key_addr, 16)
        key_end = key_start + key_size
        if mem_start <= key_start < key_end <= mem_end:  # covered by new mem
            # mem_state.pop(key)
            overlap_key_list.append(key)
        elif mem_start <= key_start < mem_end < key_end:
            print('not implemented: overlap')
            exit(-1)
        elif key_start < mem_start < key_end <= mem_end:
            print('not implemented: overlap')
            exit(-1)
        elif key_start <= mem_start < mem_end <= key_end:
            print('not implemented: overlap')
            exit(-1)
    """
    for key in overlap_key_list:
        if set_zero:
            mem_state[key] = '0'
        else:
            mem_state.pop(key)


def check_sub_mem(mem_addr: str, size: int):
    global mem_state
    mem_start = int(mem_addr, 16)
    mem_end = mem_start + size
    size_list = [4, 8, 16, 32]
    for key_start in range(mem_start-16, mem_start+4, 4):
        for key_size in size_list:
            key_end = key_start + key_size
            if key_start <= mem_start < mem_end <= key_end:
                key = hex(key_start)+','+str(key_size)
                if key in mem_state.keys():
                    return mem_state[key]
    """
    # too slow 
    for key, value in mem_state.items():
        key_addr, key_size = key2addr(key)
        key_start = int(key_addr, 16)
        key_end = key_start + key_size
        if mem_start <= key_start < key_end <= mem_end:
            print('not implemented: sub_mem')
            exit(-1)
        elif mem_start < key_start < mem_end < key_end:
            print('not implemented: sub_mem')
            exit(-1)
        elif key_start < mem_start < key_end < mem_end:
            print('not implemented: sub_mem')
            exit(-1)
        elif key_start <= mem_start < mem_end <= key_end:  # is a sub mem
            return value
    """
    return None


def xmm2reg(reg1: str, xmm2: str):
    global reg_state, xmm_regs
    reg_state[reg1] = xmm_regs[xmm2]


def xmm2mem(xmm_name: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    assert size == 32 or size == 16 or size == 4  # TODO add ymmword
    mem_key = mem_addr + ',' + str(size)
    if len(xmm_regs[xmm_name]) > 0:
        if mem_key in mem_state.keys():
            mem_state[mem_key] = xmm_regs[xmm_name]
        else:
            # check overlap
            remove_overlap_mem(mem_addr, size)
            mem_state[mem_key] = xmm_regs[xmm_name]


def reg2mem(reg_name: str, mem_addr: str, size: int):
    global reg_state, mem_state
    mem_key = mem_addr + ',' + str(size)
    if len(reg_state[reg_name]) > 0:
        if mem_key in mem_state.keys():
            mem_state[mem_key] = reg_state[reg_name]
        else:
            # check overlap
            remove_overlap_mem(mem_addr, size)
            mem_state[mem_key] = reg_state[reg_name]


def set_mem(imme_value: str, mem_addr: str, size: int):
    global mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_key in mem_state.keys():
        mem_state[mem_key] = imme_value
    else:
        # check overlap
        remove_overlap_mem(mem_addr, size)
        mem_state[mem_key] = imme_value


def mem2xmm(xmm_name: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_key in mem_state.keys():
        xmm_regs[xmm_name] = mem_state[mem_key]
    else:
        value = check_sub_mem(mem_addr, size)
        if value:
            # print('impossible?')
            xmm_regs[xmm_name] = 'sub({})'.format(value)
        else:
            xmm_regs[xmm_name] = mem_key


def mem2reg(reg_name: str, mem_addr: str, size: int):
    global reg_state, mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_key in mem_state.keys():
        reg_state[reg_name] = mem_state[mem_key]
    else:
        value = check_sub_mem(mem_addr, size)
        if value:
            reg_state[reg_name] = 'sub({})'.format(value)
        else:
            reg_state[reg_name] = mem_key


def reg2reg(reg1: str, reg2: str):
    global reg_state, mem_state
    reg_state[reg1] = reg_state[reg2]


def reg2xmm(xmm1: str, reg2: str):
    global reg_state, xmm_regs
    xmm_regs[xmm1] = reg_state[reg2]


def set_reg(reg_name: str, value: str):
    # FIXME: update eax while updating rax's value
    global reg_state
    reg_state[reg_name] = value


def reg_arith(reg_name: str):
    global reg_state
    # if the register is used in arithmetic instruction
    # we discard the state of this register
    reg_state[reg_name] = ''


def set_xmm(xmm_name: str, value: str):
    global xmm_regs
    xmm_regs[xmm_name] = value


# /* ===================================================================== */
# // Different Instructions' Handlers
# /* ===================================================================== */
def xmm2xmm(xmm1: str, xmm2: str):
    global xmm_regs, mem_state
    xmm_regs[xmm1] = xmm_regs[xmm2]


def xmm_mul_xmm(xmm1: str, xmm2: str):
    global xmm_regs, mem_state
    xmm_regs[xmm1] = '({} * {})'.format(xmm_regs[xmm1], xmm_regs[xmm2])


def xmm_div_xmm(xmm1: str, xmm2: str):
    global xmm_regs, mem_state
    xmm_regs[xmm1] = '({} / {})'.format(xmm_regs[xmm1], xmm_regs[xmm2])


def xmm_add_xmm(xmm1: str, xmm2: str, size: int):
    global xmm_regs, mem_state
    if size == 16 or size == 32:  # TODO add ymmword
        xmm_regs[xmm1] = '({} + {})'.format(xmm_regs[xmm1], xmm_regs[xmm2])
    elif size == 4:
        # TODO why do I add the 'sub' prefix at here?
        # xmm_regs[xmm1] = '{}+sub({})'.format(xmm_regs[xmm1], xmm_regs[xmm2])
        xmm_regs[xmm1] = '({} + {})'.format(xmm_regs[xmm1], xmm_regs[xmm2])


def xmm_sub_xmm(xmm1: str, xmm2: str, size: int):
    global xmm_regs, mem_state
    if size == 16 or size == 32:  # TODO add ymmword
        xmm_regs[xmm1] = '({} + {})'.format(xmm_regs[xmm1], xmm_regs[xmm2])
    elif size == 4:
        # TODO why do I add the 'sub' prefix at here?
        # xmm_regs[xmm1] = '{}-sub({})'.format(xmm_regs[xmm1], xmm_regs[xmm2])
        xmm_regs[xmm1] = '({} - {})'.format(xmm_regs[xmm1], xmm_regs[xmm2])


def neighbor_mem_merge(mem1: str, mem2: str, mem1_size: int, mem2_size: int):
    if mem1.startswith('0x') and mem2.startswith('0x'):
        mem1_addr = int(mem1.split(',')[0], 16)
        mem2_addr = int(mem2.split(',')[0], 16)
        if mem1_addr + mem1_size == mem2_addr:
            return mem1.split(',')[0] + ',' + str(mem1_size + mem2_size)
    return '({}:{}),16'.format(mem2, mem1)  # cannot merge


def xmm_movlhps_xmm(xmm1: str, xmm2: str):
    global xmm_regs, mem_state
    if xmm_regs[xmm1].endswith('8') and xmm_regs[xmm2].endswith('8'):
        # xmm_regs[xmm1] = '({}:{}),16'.format(xmm_regs[xmm2], xmm_regs[xmm1])
        xmm_regs[xmm1] = neighbor_mem_merge(xmm_regs[xmm1], xmm_regs[xmm2], 8, 8)
    else:
        xmm_regs[xmm1] = 'lpd({}):lpd({})'.format(xmm_regs[xmm2], xmm_regs[xmm1])


def xmm_movhlps_xmm(xmm1: str, xmm2: str):
    global xmm_regs, mem_state
    xmm_regs[xmm1] = 'hpd({}):hpd({})'.format(xmm_regs[xmm1], xmm_regs[xmm2])


def xmm_unpcklpd_xmm(xmm1: str, xmm2: str):
    global xmm_regs, mem_state
    # assert xmm_regs[xmm1].endswith('4') and xmm_regs[xmm2].endswith('4')
    if xmm_regs[xmm1].endswith('8') and xmm_regs[xmm2].endswith('8'):
        # xmm_regs[xmm1] = '({}:{}),16'.format(xmm_regs[xmm2], xmm_regs[xmm1])
        xmm_regs[xmm1] = neighbor_mem_merge(xmm_regs[xmm1], xmm_regs[xmm2], 8, 8)
    else:
        xmm_regs[xmm1] = 'lpd({}):lpd({})'.format(xmm_regs[xmm2], xmm_regs[xmm1])


def xmm_unpcklps_xmm(xmm1: str, xmm2: str):
    global xmm_regs, mem_state
    if xmm_regs[xmm1].endswith('4') and xmm_regs[xmm2].endswith('4'):
        # xmm_regs[xmm1] = '({}:{}),8'.format(xmm_regs[xmm2], xmm_regs[xmm1])
        xmm_regs[xmm1] = neighbor_mem_merge(xmm_regs[xmm1], xmm_regs[xmm2], 4, 4)
    else:
        xmm_regs[xmm1] = 'lps({}):lps({})'.format(xmm_regs[xmm2], xmm_regs[xmm1])


def xmm_unpckhpd_xmm(xmm1: str, xmm2: str):
    global xmm_regs, mem_state
    xmm_regs[xmm1] = 'hpd({}):hpd({})'.format(xmm_regs[xmm2], xmm_regs[xmm1])


def xmm_unpckhps_xmm(xmm1: str, xmm2: str):
    global xmm_regs, mem_state
    xmm_regs[xmm1] = 'hps({}):hps({})'.format(xmm_regs[xmm2], xmm_regs[xmm1])


def xmm_punpckhdq_xmm(xmm1: str, xmm2: str):
    global xmm_regs
    # never used
    xmm_regs[xmm1] = 'interl({}, {})'.format(xmm_regs[xmm1], xmm_regs[xmm2])


def xmm_unpcklpd_mem(xmm_name: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    # assert xmm_regs[xmm_name].endswith('4') and mem_state[mem_key].endswith('4')
    if xmm_regs[xmm_name].endswith('8') and mem_state[mem_key].endswith('8'):
        # xmm_regs[xmm_name] = '({}:{}),16'.format(mem_state[mem_key], xmm_regs[xmm_name])
        xmm_regs[xmm_name] = neighbor_mem_merge(xmm_regs[xmm_name], mem_state[mem_key], 8, 8)
    else:
        xmm_regs[xmm_name] = 'lpd({}):lpd({})'.format(mem_state[mem_key], xmm_regs[xmm_name])


def xmm_unpcklps_mem(xmm_name: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_state[mem_key].endswith('4') and xmm_regs[xmm_name].endswith('4'):
        # xmm_regs[xmm_name] = '({}:{}),8'.format(mem_state[mem_key], xmm_regs[xmm_name])
        xmm_regs[xmm_name] = neighbor_mem_merge(xmm_regs[xmm_name], mem_state[mem_key], 4, 4)
    else:
        xmm_regs[xmm_name] = 'lps({}):lps({})'.format(mem_state[mem_key], xmm_regs[xmm_name])


def xmm_unpckhpd_mem(xmm_name: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    xmm_regs[xmm_name] = 'hpd({}):hpd({})'.format(mem_state[mem_key], xmm_regs[xmm_name])


def xmm_unpckhps_mem(xmm_name: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    xmm_regs[xmm_name] = 'hps({}):hps({})'.format(mem_state[mem_key], xmm_regs[xmm_name])


def xmm_mul_mem(xmm_name: str, mem_addr: str, size: int):
    # global xmm_regs, mem_state
    # xmm_regs[xmm1] = '({} * {})'.format(xmm_regs[xmm1], xmm_regs[xmm2])
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_key in mem_state.keys():
        xmm_regs[xmm_name] = '({} * {})'.format(xmm_regs[xmm_name], mem_state[mem_key])


def xmm_add_mem(xmm_name: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_key in mem_state.keys():
        xmm_regs[xmm_name] = '({} + {})'.format(xmm_regs[xmm_name], mem_state[mem_key])
    else:
        # TODO: this is inaccurate
        if mem_key.startswith('0x7f') or mem_key.startswith('0x4'):
            return
        xmm_regs[xmm_name] = '({} + {})'.format(xmm_regs[xmm_name], mem_key)


def xmm_sub_mem(xmm_name: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_key in mem_state.keys():
        xmm_regs[xmm_name] = '({} - {})'.format(xmm_regs[xmm_name], mem_state[mem_key])
    else:
        xmm_regs[xmm_name] = '({} - {})'.format(xmm_regs[xmm_name], mem_key)


def xmm_max_xmm(xmm1: str, xmm2: str, xmm3=''):
    global xmm_regs, mem_state
    if len(xmm3) == 0:
        xmm_regs[xmm1] = 'max({}, {})'.format(xmm_regs[xmm1], xmm_regs[xmm2])
    else:
        xmm_regs[xmm1] = 'max({}, {})'.format(xmm_regs[xmm2], xmm_regs[xmm3])


def xmm_max_mem(xmm_name: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_key in mem_state.keys():
        xmm_regs[xmm_name] = 'max({}, {})'.format(xmm_regs[xmm_name], mem_state[mem_key])
    else:
        xmm_regs[xmm_name] = 'max({}, {})'.format(xmm_regs[xmm_name], mem_key)


def xmm_vadd_mem(xmm_1: str, xmm_2: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_key in mem_state.keys():
        xmm_regs[xmm_1] = '({} + {})'.format(xmm_regs[xmm_2], mem_state[mem_key])
    else:
        xmm_regs[xmm_1] = '({} + {})'.format(xmm_regs[xmm_2], mem_key)


def xmm_vmul_mem(xmm_write: str, xmm_read: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_key in mem_state.keys():
        xmm_regs[xmm_write] = '({} * {})'.format(xmm_regs[xmm_read], mem_state[mem_key])
    else:
        # TODO sub-memory check?
        xmm_regs[xmm_write] = '({} * {})'.format(xmm_regs[xmm_read], mem_key)


def vfmadd231ss(xmm_1: str, xmm_2: str, mem_addr: str, size: int):
    global xmm_regs, mem_state
    mem_key = mem_addr + ',' + str(size)
    if mem_key in mem_state.keys():
        xmm_regs[xmm_1] = '({} * {} + {})'.format(xmm_regs[xmm_2], mem_state[mem_key], xmm_regs[xmm_1])
    else:
        # TODO sub-memory check?
        xmm_regs[xmm_1] = '({} * {} + {})'.format(xmm_regs[xmm_2], mem_key, xmm_regs[xmm_1])


# -----------------------------------------------
# different handlers
# for a sub-set instructions related to xmm registers


def handle_movaps(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    if op1 in xmm_regs.keys() and '[' in op2:
        # mem --> xmm/ymm reg
        if 'ymmword' in op2:
            size = 32
        elif 'xmmword' in op2:
            size = 16
        elif 'dword' in op2:
            size = 4
        mem2xmm(op1, mem_addr, size)
    elif op2 in xmm_regs.keys() and '[' in op1:
        # xmm/ymm rge --> mem
        if 'ymmword' in op1:
            size = 32
        elif 'xmmword' in op1:
            size = 16
        elif 'dword' in op1:
            size = 4
        xmm2mem(op2, mem_addr, size)
    elif op1 in xmm_regs.keys() and op2 in xmm_regs.keys():
        xmm2xmm(op1, op2)
    else:
        print('not implemented: movaps')
        exit(-1)


def handle_movd(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    if op1 in xmm_regs.keys() and op2 in reg_state.keys():
        # reg --> xmm
        reg2xmm(op1, op2)
    else:
        print('not implemented: movd')
        exit(-1)


def handle_movq(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    if op1 in xmm_regs.keys() and op2 in reg_state.keys():
        # reg --> xmm
        reg2xmm(op1, op2)
    elif op1 in reg_state.keys() and op2 in xmm_regs.keys():
        xmm2reg(op1, op2)
    else:
        print('not implemented: movq')
        exit(-1)


def handle_movss(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    if op1 in xmm_regs.keys() and '[' in op2:
        # mem --> xmm reg
        if 'dword' in op2:
            size = 4
        elif 'qword' in op2:
            size = 8
        mem2xmm(op1, mem_addr, size)
    elif op2 in xmm_regs.keys() and '[' in op1:
        # xmm reg --> mem
        if 'dword' in op1:
            size = 4
        elif 'qword' in op2:
            size = 8
        xmm2mem(op2, mem_addr, size)
    else:
        print('not implemented: movss')
        exit(-1)


def handle_vfmadd_ss(code_list, mem_addr):
    assert len(code_list) == 4
    op1 = code_list[1]
    op2 = code_list[2]
    op3 = code_list[3]
    if code_list[0] == 'vfmadd213ss':
        if op1 in xmm_regs.keys() and op2 in xmm_regs.keys() and '[' in op3:
            xmm_mul_xmm(op1, op2)
            if 'dword' in op3:
                size = 4
            xmm_add_mem(op1, mem_addr, size)
        else:
            print('not implemented: vfmadd213ss')
            exit(-1)
    elif code_list[0] == 'vfmadd231ss':
        if op1 in xmm_regs.keys() and op2 in xmm_regs.keys() and '[' in op3:
            if 'dword' in op3:
                size = 4
            vfmadd231ss(op1, op2, mem_addr, size)
        else:
            print('not implemented: vfmadd231ss')
            exit(-1)
    elif code_list[0] == 'vfmadd213ps':
        if op1 in xmm_regs.keys() and op2 in xmm_regs.keys() and '[' in op3:
            xmm_mul_xmm(op1, op2)
            if 'ymmword' in op3:
                size = 32
            xmm_add_mem(op1, mem_addr, size)
        else:
            print('not implemented: vfmadd213ps')
            exit(-1)
    elif code_list[0] == 'vfmadd231ps':
        if op1 in xmm_regs.keys() and op2 in xmm_regs.keys() and '[' in op3:
            if 'ymmword' in op3:
                size = 32
            vfmadd231ss(op1, op2, mem_addr, size)
        else:
            print('not implemented: vfmadd231ps')
            exit(-1)
    else:
        print('not implemented: vfmadd')
        exit(-1)


def handle_vmulss(code_list, mem_addr):
    assert len(code_list) == 4
    op1 = code_list[1]
    op2 = code_list[2]
    op3 = code_list[3]
    if op1 in xmm_regs.keys() and op2 in xmm_regs.keys() and '[' in op3:
        if 'dword' in op3:
            size = 4
        xmm_vmul_mem(op1, op2, mem_addr, size)
    else:
        print('not implemented: vmulss')
        exit(-1)


def handle_vaddss(code_list, mem_addr):
    assert len(code_list) == 4
    op1 = code_list[1]
    op2 = code_list[2]
    op3 = code_list[3]
    if op1 in xmm_regs.keys() and op2 in xmm_regs.keys() and '[' in op3:
        if 'dword' in op3:
            size = 4
        xmm_vadd_mem(op1, op2, mem_addr, size)
    else:
        print('not implemented: vaddss')
        exit(-1)


def handle_vmaxss(code_list, mem_addr):
    assert len(code_list) == 4
    op1 = code_list[1]
    op2 = code_list[2]
    op3 = code_list[3]
    if op1 in xmm_regs.keys() and op2 in xmm_regs.keys() and op3 in xmm_regs.keys():
        xmm_max_xmm(op1, op2, op3)
    else:
        print('not implemented: vmaxss')
        exit(-1)


def handle_vbroadcastss(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    if op1 in xmm_regs.keys() and '[' in op2:
        if 'dword' in op2:
            size = 4
        mem2xmm(op1, mem_addr, size)
    else:
        print('not implemented: vbroadcastss')
        exit(-1)


def handle_mulps(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    if '[' in op2:
        assert 'dword' in op2 or 'xmm' in op2
        if 'dword' in op2:
            size = 4
        elif 'xmm' in op2:
            size = 16
        mem_key = mem_addr+','+str(size)
    assert op1 in xmm_regs.keys() and (op2 in xmm_regs.keys() or mem_key in mem_state.keys())
    if op2 in xmm_regs.keys():
        xmm_mul_xmm(op1, op2)
    elif mem_key in mem_state.keys():
        xmm_mul_mem(op1, mem_addr, size)



def handle_divss(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    assert op1 in xmm_regs.keys() and op2 in xmm_regs.keys()
    xmm_div_xmm(op1, op2)


def handle_addps(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    assert op1 in xmm_regs.keys()
    if op1 in xmm_regs.keys() and op2 in xmm_regs.keys():
        xmm_add_xmm(op1, op2, 16)
    elif op1 in xmm_regs.keys() and '[' in op2:
        if 'xmmword' in op2:
            size = 16
        elif 'dword' in op2:
            size = 4
        xmm_add_mem(op1, mem_addr, size)


def handle_addss(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    assert op1 in xmm_regs.keys()
    if op1 in xmm_regs.keys() and op2 in xmm_regs.keys():
        if xmm_regs[op1] == xmm_regs[op2]:
            print('warning: skip addss')
            return
        elif xmm_regs[op1] in xmm_regs[op2]:
            print('warning: skip addss')
            set_xmm(op1, xmm_regs[op2])
            return
        elif xmm_regs[op2] in xmm_regs[op1]:
            print('warning: skip addss')
            return
        # TODO this maybe not correct, assume program will not add two same value (except for rearrange )
        xmm_add_xmm(op1, op2, 4)
    elif op1 in xmm_regs.keys() and '[' in op2:
        if 'dword' in op2:
            size = 4
        xmm_add_mem(op1, mem_addr, size)
    else:
        print('not implemented: addss')
        exit(-1)


def handle_subss(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    assert op1 in xmm_regs.keys()
    if op1 in xmm_regs.keys() and op2 in xmm_regs.keys():
        # TODO this is not correct
        xmm_sub_xmm(op1, op2, 4)
    elif op1 in xmm_regs.keys() and '[' in op2:
        if 'dword' in op2:
            size = 4
        xmm_sub_mem(op1, mem_addr, size)
    else:
        print('not implemented: subss')
        exit(-1)


def handle_maxps(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    assert op1 in xmm_regs.keys()
    if op1 in xmm_regs.keys() and op2 in xmm_regs.keys():
        xmm_max_xmm(op1, op2)
    elif op1 in xmm_regs.keys() and '[' in op2:
        if 'dword' in op2:
            size = 4
        elif 'qword' in op2:
            size = 8
        elif 'xmmword' in op2:
            size = 8
        xmm_max_mem(op1, mem_addr, size)
    else:
        print('not implemented: maxps')
        exit(-1)


def handle_maxss(code_list, mem_addr):
    assert len(code_list) == 3
    op1 = code_list[1]
    op2 = code_list[2]
    assert op1 in xmm_regs.keys()
    if op1 in xmm_regs.keys() and op2 in xmm_regs.keys():
        # TODO not correct
        xmm_max_xmm(op1, op2)
    elif op1 in xmm_regs.keys() and '[' in op2:
        if 'dword' in op2:
            size = 4
        xmm_max_mem(op1, mem_addr, size)
    else:
        print('not implemented: maxss')
        exit(-1)


def handle_xorps(code_list, mem_addr):
    if len(code_list) == 3:  # xorps
        op1 = code_list[1]
        op2 = code_list[2]
        assert op1 in xmm_regs.keys() and op2 in xmm_regs.keys()
        if op1 == op2:
            set_xmm(op1, '0')
        else:
            print('not implemented: xorps')
            exit(-1)
    elif len(code_list) == 4:  # vxorps
        op1 = code_list[1]
        op2 = code_list[2]
        op3 = code_list[3]
        assert op1 in xmm_regs.keys() and op2 in xmm_regs.keys() and op3 in xmm_regs.keys()
        if op1 == op2 == op3:
            set_xmm(op1, '0')
        else:
            print('not implemented: vxorps')
            exit(-1)
    else:
        print('not implemented: xorps')
        exit(-1)


def handle_unpcklp(code_list, mem_addr):
    op1 = code_list[1]
    op2 = code_list[2]
    assert op1 in xmm_regs.keys()
    if op1 in xmm_regs.keys() and op2 in xmm_regs.keys():
        if code_list[0] == 'unpcklpd':
            xmm_unpcklpd_xmm(op1, op2)
        elif code_list[0] == 'unpcklps':
            xmm_unpcklps_xmm(op1, op2)
        else:
            print('not implemented unpcklp')
            exit(-1)
        # xmm_unpck_xmm(op1, op2)
    elif op1 in xmm_regs.keys() and '[' in op2:
        if 'xmmword' in op2:
            size = 16
        elif 'qword' in op2:
            size = 8
        elif 'dword' in op2:
            size = 4
        if code_list[0] == 'unpckhpd':
            xmm_unpcklpd_mem(op1, mem_addr, size)
        elif code_list[0] == 'unpckhps':
            xmm_unpcklps_mem(op1, mem_addr, size)
        elif code_list[0] == 'unpcklps':
            xmm_unpcklps_mem(op1, mem_addr, size)
        else:
            print('not implemented unpcklp')
            exit(-1)
        # xmm_unpck_mem(op1, mem_addr, size)
    else:
        print('not implemented: unpcklp')
        exit(-1)


def handle_unpckhp(code_list, mem_addr):
    op1 = code_list[1]
    op2 = code_list[2]
    assert op1 in xmm_regs.keys()
    if op1 in xmm_regs.keys() and op2 in xmm_regs.keys():
        if code_list[0] == 'unpckhpd':
            xmm_unpckhpd_xmm(op1, op2)
        elif code_list[0] == 'unpckhps':
            xmm_unpckhps_xmm(op1, op2)
        else:
            print('not implemented: handle_unpckhp')
            exit(-1)
    elif op1 in xmm_regs.keys() and '[' in op2:
        if 'xmmword' in op2:
            size = 16
        elif 'qword' in op2:
            size = 8
        elif 'dword' in op2:
            size = 4
        if code_list[0] == 'unpckhpd':
            xmm_unpckhpd_mem(op1, mem_addr, size)
        elif code_list[0] == 'unpckhps':
            xmm_unpckhps_mem(op1, mem_addr, size)
        else:
            print('not implemented: handle_unpckhp')
            exit(-1)
    else:
        print('not implemented: unpckhpd, unpckhps')
        exit(-1)


def handle_punpckhdq(code_list, mem_addr):
    op1 = code_list[1]
    op2 = code_list[2]
    assert op1 in xmm_regs.keys() and op2 in xmm_regs.keys()
    xmm_punpckhdq_xmm(op1, op2)


def handle_mov_ps(code_list, mem_addr):
    op1 = code_list[1]
    op2 = code_list[2]
    assert op1 in xmm_regs.keys() and op2 in xmm_regs.keys()
    if op1 == op2:
        pass
    elif code_list[0] == 'movlhps':
        xmm_movlhps_xmm(op1, op2)
    elif code_list[0] == 'movhlps':
        xmm_movhlps_xmm(op1, op2)
    else:
        print('not implemented: movlhps, movhlps')
        exit(-1)


# -----------------------------------------------
# different handlers
# for a sub-set instructions related to regular registers


def handle_mov(code_list, mem_addr):
    op1 = code_list[1]
    op2 = code_list[2]
    if op1 in reg_state.keys() and '[' in op2:
        # mem --> reg
        if 'qword' in op2:
            size = 8
        elif 'dword' in op2:
            size = 4
        mem2reg(op1, mem_addr, size)
    elif op2 in reg_state.keys() and '[' in op1:
        # rge --> mem
        if 'qword' in op1:
            size = 8
        elif 'dword' in op1:
            size = 4
        reg2mem(op2, mem_addr, size)
    elif op1 in reg_state.keys() and op2 in reg_state.keys():
        # reg --> reg
        reg2reg(op1, op2)
    elif op1 in reg_state.keys():
        # mov immediate value to register
        imm_value = int(op2, 16)
        imm_str = hex(imm_value)
        set_reg(op1, imm_str)
    elif '[' in op1:
        # mov immediate value to register
        imm_value = int(op2, 16)
        imm_str = hex(imm_value)
        if 'qword' in op1:
            size = 8
        elif 'dword' in op1:
            size = 4
        set_mem(imm_str, mem_addr, size)
    else:
        print('not implemented: mov')
        exit(-1)


def handle_lea(code_list, mem_addr):
    op1 = code_list[1]
    op2 = code_list[2]
    if op1 in reg_state.keys() and '[' in op2:
        mem_value = int(mem_addr, 16)
        mem_str = hex(mem_value)
        set_reg(op1, mem_str)
    else:
        print('not implemented: lea')
        exit(-1)


def handle_arith(code_list, mem_addr):
    index = 1
    while index < len(code_list):
        op = code_list[index]
        if op in reg_state.keys():
            reg_arith(op)
        index += 1


def handle_memset(code_list, rax_value):
    # TODO: currently can only set memory to zero --syntax=intel
    global mem_state
    addr = reg_state['rdi']
    size = int(reg_state['edx'], 16)
    # TODO: what if the addr is not an address? e.g., it is maybe empty?
    if len(addr) > 0 and ',' not in addr:  # if addr is not an address
        remove_overlap_mem(addr, size, set_zero=True)

    handle_unknown_call(code_list, rax_value)


def handle_expf(code_list, rax_value):
    global xmm_regs
    xmm_regs['xmm0'] = 'expf({})'.format(xmm_regs['xmm0'])

    handle_unknown_call(code_list, rax_value)


def handle_unknown_call(code_list, rax_value: str):
    global reg_state
    if len(rax_value) < 2:
        return
    # function call will update rax register
    # but we don't know which function is called
    rax_int = int(rax_value, 16)
    reg_state['al'] = hex(rax_int & 0xff)
    reg_state['ah'] = hex((rax_int & 0xff00) >> 8)
    reg_state['ax'] = hex(rax_int & 0xffff)
    reg_state['eax'] = hex(rax_int & 0xffffffff)
    reg_state['rax'] = rax_value


# -----------------------------------------------
# test
def test_one_function(asm_path: str, generate_log: bool):
    program_path = '/home/lifter/e9patch/test/demo_static_O0_exe'
    input_data = '/home/lifter/e9patch/test/number.bin'
    if generate_log:
        start_addr, end_addr = get_range(asm_path)
        current_dir = os.path.dirname(__file__)
        log_path = os.path.join(current_dir, './log.txt')
        all_inst_trace_1(program_path, input_data, start_addr, end_addr, log_path)

    lightweight_SymEx('./log.txt')

    explain_tvm_conv2d_result('./mem_log.txt')


def test_glow_function():
    program_path = '/home/lifter/Documents/tvm_output/glow_dbg/mnist_8_no_pie'
    input_data = '/home/lifter/Documents/tvm_output/glow_dbg/3_1020.png'
    # start_addr = '0x402e20'  # first conv
    # end_addr = '0x40327d'
    start_addr = '0x403490'  # second conv
    end_addr = '0x403b6a'
    current_dir = os.path.dirname(__file__)
    log_path = os.path.join(current_dir, './log.txt')
    all_inst_trace_1(program_path, input_data, start_addr, end_addr, log_path)
    lightweight_SymEx('./log.txt')

    explain.explain_glow_conv2d_result('./mem_log.txt')


def test():
    test_one_function('/home/lifter/Documents/tvm_output/O0/funcs/012.txt.fused_nn_conv2d', generate_log=True)


if __name__ == '__main__':
    # explain_glow_conv2d_result('./mem_log.txt')
    # explain_tvm_conv2d_result('./mem_log.txt')
    # test()
    # test_glow_function()
    lightweight_SymEx('/home/lifter/pin-3.14-98223-gb010a12c6-gcc-linux/source/tools/MyPinTool/pinatrace.out')
    explain.explain_tvm_conv2d_result('./mem_log.txt')
