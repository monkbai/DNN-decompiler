import os
import sys
import copy
import time
import utils
import random
import explain
import pin_tools
import logging
print('get logger: {}'.format('decompiler.'+__name__))
logger = logging.getLogger('decompiler.'+__name__)


def get_early_stop(asm_path: str):
    """
        early_stop: if the outermost loop has more than 16 cycles, we can stop logging after first cycle
        for conv layer only
    """
    loop_count = 0
    with open(asm_path, 'r') as f:
        asm_lines = f.readlines()
        asm_lines.reverse()
        for line in asm_lines:
            asm = line[40:].strip()  # type: str
            if asm.startswith('cmp'):
                loop_count = asm.split(',')[1].strip()
                if loop_count.endswith('h'):
                    loop_count = loop_count.strip('h')
                    loop_count = int(loop_count, 16)
                else:
                    loop_count = int(loop_count)
                break
    if loop_count >= 16:
        early_stop = asm.split(':')[0]
        return early_stop, loop_count
    else:
        return '', loop_count


def log_trace(asm_path: str, prog_path: str, in_data: str, out_log_path: str):
    asm_path = os.path.abspath(asm_path)
    early_stop, loop_count = get_early_stop(asm_path)
    start_addr, end_addr = utils.get_func_range(asm_path)
    if len(early_stop) > 0:
        end_addr = early_stop

    log_path = os.path.abspath(out_log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(in_data)

    pin_tools.inst_trace_log(log_path, start_addr, end_addr, prog_path, data_path)
    return start_addr, end_addr


def reverse_trace(original_trace: str, new_trace: str):
    original_trace = os.path.abspath(original_trace)
    new_trace = os.path.abspath(new_trace)
    pin_tools.tac_cmd(original_trace, new_trace)


def pick_rand_addr(func_asm_path: str, prog_path: str, in_data: str, mem_write_log_path: str):
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    mem_write_log_path = os.path.abspath(mem_write_log_path)
    func_asm_path = os.path.abspath(func_asm_path)

    start_addr, end_addr = utils.get_func_range(func_asm_path)
    early_stop, loop_size = get_early_stop(func_asm_path)
    utils.mem_write_log(mem_write_log_path, start_addr, early_stop, prog_path, in_data)
    write_mem_regions = utils.memory_slices(mem_write_log_path)
    out_mem = explain.biggest_region(write_mem_regions)
    rnd_addr = random.randrange(out_mem[0], out_mem[1], 4)
    # mid_addr = out_mem[0] + (out_mem[1] - out_mem[0])/2
    # mid_addr = int(mid_addr)
    # mid_addr = hex(mid_addr)
    rnd_addr = hex(rnd_addr)
    return rnd_addr, loop_size


def before_taint(asm_path: str, prog_path: str, data_path: str, log_path: str):
    # Generate trace
    rev_log_path = log_path.replace('.log', '_rev.log')
    start_addr, end_addr = log_trace(asm_path, prog_path, data_path, log_path)
    # Random pick a target address
    tmp_mem_write_log = './tmp_mem_write.log'
    rnd_addr, loop_size = pick_rand_addr(asm_path, prog_path, data_path, tmp_mem_write_log)
    # Reverse trace
    reverse_trace(log_path, rev_log_path)
    return rev_log_path, rnd_addr, loop_size, start_addr, end_addr


# ---------- above functions are used to generate and reverse trace ----------

# ===============================================
#
# Reverse Taint Analysis (Trace Backward Slicing)
#
# ===============================================

common_regs = {'ah', 'ch', 'dh', 'bh',
               'al', 'cl', 'dl', 'bl', 'spl', 'bpl', 'sil', 'dil',
               'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b',
               'ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di',
               'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w',
               'eax', 'ecx', 'edx', 'ebx', 'esp', 'ebp', 'esi', 'edi',
               'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
               'rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 'rsi', 'rdi',
               'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', }

xmm_regs = {
    # 128 bits
    'xmm0', 'xmm1', 'xmm2', 'xmm3',
    'xmm4', 'xmm5', 'xmm6', 'xmm7',
    'xmm8', 'xmm9', 'xmm10', 'xmm11',
    'xmm12', 'xmm13', 'xmm14', 'xmm15',
    # 256 bits
    'ymm0', 'ymm1', 'ymm2', 'ymm3',
    'ymm4', 'ymm5', 'ymm6', 'ymm7',
    'ymm8', 'ymm9', 'ymm10', 'ymm11',
    'ymm12', 'ymm13', 'ymm14', 'ymm15',
}

# global state
# which regs, dwords are tainted
# need to find the last write
tainted_regs = set()
tainted_mems = set()


def set_tainted(addr_list: list):
    global tainted_mems
    for addr in addr_list:
        tainted_mems.add(addr)


def reverse_taint(re_trace_log: str, new_trace: str):
    global logger
    start_time = time()

    reverse_log = os.path.abspath(re_trace_log)
    new_trace_log = os.path.abspath(new_trace)
    final_bufs = []
    read_buf = []
    with open(reverse_log, 'r') as f:
        # print('reading...')
        # lines = f.readlines()
        # print('reading finished')
        line_idx = 0
        idx = 0
        while True:  # line_idx < len(lines):
            line = f.readline()  # line = lines[line_idx]
            line_idx += 1

            if not line:
                break

            if line.startswith('0x'):  # end of one inst, handle current inst
                read_buf.insert(0, line)
                idx += 1
                if idx % 1000000 == 0:  # debug, million
                    print(idx)
                    print('len final_bufs {}'.format(len(final_bufs)))
                    print('len tainted_mems {}'.format(len(tainted_mems)))
                    print('len tainted_regs {}'.format(len(tainted_regs)))
                # TODO: handle the current instruction
                # the core function of reverse taint
                if handle_inst(read_buf):  # handle instructions
                    final_bufs.insert(0, copy.deepcopy(read_buf))
                read_buf.clear()
            else:
                read_buf.insert(0, line)
        f.close()

    end_time = time()
    elapsed_time = end_time - start_time
    logger.info("Taint Analysis - Elapsed Time: {}s".format(elapsed_time))

    # write final_bufs to file
    with open(new_trace_log, 'w') as f:
        for r_buf in final_bufs:
            for line in r_buf:
                f.write(line)
        f.close()


def handle_inst(read_buf: list):
    if len(read_buf) < 1:
        return False
    # Return Ture if current should be kept, False otherwise
    # In which caese, current instruction should be kept?
    # Rules:
    # (1) if one tainted obj is written, rm it from tainted objs and set read objs as tainted
    # (2) if one tainted obj is read then written, kept it as tainted add other read objs as tainted
    # (3) xor eax, eax -- rm eax from tainted objs
    if read_buf[0].startswith('0x'):
        line = read_buf[0]
        # print(line)  # debug

        # Step 1: parse one trace line
        addr, line = line.split(':')
        addr = addr.strip()
        line = line.strip()
        if ' ' in line:
            opcode, operands = line[:line.find(' ')], line[line.find(' ') + 1:]
            if ',' in operands:
                operands = operands.split(',')
                for operand in operands:
                    operand = [operand.strip()]
            else:
                tmp_op = operands
                operands = [tmp_op]
        else:
            opcode = line
            operands = []

        # does this instruction shoule be checked?
        if opcode.startswith('data') or opcode.startswith('nop'):  # it's not an instruction
            return False
        elif not check_operands(operands, read_buf[1]):  # read_buf[1] -> mem read/write addr
            return False

        # debug
        # print(read_buf)

        mem_line = read_buf[1]
        mem_addr = mem_line.split(':')[1].strip()
        if opcode.startswith('mov') or opcode.startswith('lea'):
            kept = handle_mov(opcode, operands, mem_addr)
        elif opcode.startswith('vmovss') or opcode.startswith('vmovups') or opcode.startswith('vmovaps'):
            kept = handle_two(opcode, operands, mem_addr)
        elif opcode.startswith('vbroadcastss'):
            kept = handle_two(opcode, operands, mem_addr)
        elif opcode.startswith('vmaxss') or opcode.startswith('vaddss') or opcode.startswith(
                'vmulss') or opcode.startswith('vaddps'):
            kept = handle_three(opcode, operands, mem_addr)
        elif opcode.startswith('vfmadd231ss') or opcode.startswith('vfmadd213ps') or opcode.startswith('vfmadd231ps'):
            kept = handle_three(opcode, operands, mem_addr, read_op1=True)
        elif opcode.startswith('vxorps'):
            kept = handle_vxor(opcode, operands, mem_addr)
        else:
            handle_not_implemented(opcode, operands)
        if kept:  # debug
            pass  # print('added')
        else:
            pass  # print('not')
        return kept
    else:
        return False


# ===============================================
#
# Utilities
#
# ===============================================


def is_number(s: str):
    try:
        int(s)
        return True
    except ValueError:
        pass

    return False


def check_operands(operands: list, mem_line: str):
    # retur True if this instruction should be handled
    if len(operands) < 1:
        return False
    for op in operands[:1]:  # only check the written op
        op = op.strip()
        mem_addr = mem_line.split(':')[1].strip()
        if '[' in op:
            m_addr_list = split_addr_list(mem_addr, op)
            for m_addr in m_addr_list:
                # print(m_addr)  # debug
                if m_addr in tainted_mems:
                    return True
        elif op.startswith('0x'):  # imm value hex
            addr = hex(int(op, 16))
            if addr in tainted_mems:
                return True
        elif is_number(op):  # immm value int:
            continue
        elif op in common_regs:  # registers
            if op in tainted_regs:
                return True
        elif op in xmm_regs:  # xmm registers
            if op in tainted_regs:
                return True
        else:
            assert False, 'error:{} undefined op'.format(op)
    return False


def split_addr_list(mem_addr: str, op_str: str):
    m_addr_list = []
    if '[' in op_str:
        if 'ymmword ptr' in op_str:
            size = 32
        elif 'xmmword ptr' in op_str:
            size = 16
        elif 'qword ptr' in op_str:
            size = 8
        elif 'dword ptr' in op_str:
            size = 4
        # elif 'word' in op_str:
        #    size = 2
        elif 'ptr' in op_str:
            size = 8
        else:
            assert False, 'error:{} undefined size'.format(op_str)
        addr_int = int(mem_addr, 16)
        for step in range(0, size, 4):  # the smallest unit is 4 bytes
            m_addr = hex(addr_int + step)
            m_addr_list.append(m_addr)
    return m_addr_list


# ===============================================
#
# Different Handlers
#
# ===============================================


def handle_mov(opcode: str, operands: list, mem_addr: str):
    # TODO: handler of lea is not accurate
    global tainted_mems, tainted_regs
    assert len(operands) == 2, 'should has only two operands'
    kept_flag = False
    op1 = operands[0].strip()
    op2 = operands[1].strip()
    if '[' in op1 and op2 in common_regs:
        # move from register to memory
        m_addr_list = split_addr_list(mem_addr, op1)
        for m_addr in m_addr_list:
            if m_addr in tainted_mems:
                kept_flag = True
                tainted_mems.remove(m_addr)
                tainted_regs.add(op2)
    elif op1 in common_regs and '[' in op2:
        # move from memory to register
        m_addr_list = split_addr_list(mem_addr, op2)
        if op1 in tainted_regs:
            kept_flag = True
            tainted_regs.remove(op1)
            for m_addr in m_addr_list:
                tainted_mems.add(m_addr)
    else:
        assert False, 'undefined {} {}'.format(opcode, operands)
    return kept_flag


def handle_two(opcode: str, operands: list, mem_addr: str):
    global tainted_mems, tainted_regs
    assert len(operands) == 2, 'should has only two operands'
    kept_flag = False
    op1 = operands[0].strip()
    op2 = operands[1].strip()
    if '[' in op1 and op2 in xmm_regs:
        # move from register to memory
        m_addr_list = split_addr_list(mem_addr, op1)
        for m_addr in m_addr_list:
            if m_addr in tainted_mems:
                kept_flag = True
                tainted_mems.remove(m_addr)
                tainted_regs.add(op2)
    elif op1 in xmm_regs and '[' in op2:
        # move from memory to register
        m_addr_list = split_addr_list(mem_addr, op2)
        if op1 in tainted_regs:
            kept_flag = True
            tainted_regs.remove(op1)
            for m_addr in m_addr_list:
                tainted_mems.add(m_addr)
    elif op1 in xmm_regs and op2 in xmm_regs:
        # move from register to register
        if op1 in tainted_regs:
            kept_flag = True
            tainted_regs.remove(op1)
            tainted_regs.add(op2)
            # for m_addr in m_addr_list:
            #    tainted_mems.add(m_addr)
    else:
        assert False, 'undefined {} {}'.format(opcode, operands)
    return kept_flag


def handle_three(opcode: str, operands: list, mem_addr: str, read_op1=False):
    global tainted_mems, tainted_regs
    assert len(operands) == 3, 'should has 3 operands'
    kept_flag = False
    op1 = operands[0].strip()
    op2 = operands[1].strip()
    op3 = operands[2].strip()
    assert '[' not in op1 and '[' not in op2, 'not implemented'
    if op1 in xmm_regs and op2 in xmm_regs and op3 in xmm_regs:
        if op1 in tainted_regs:
            kept_flag = True
            tainted_regs.remove(op1)
            tainted_regs.add(op2)
            tainted_regs.add(op3)
            if read_op1:
                tainted_regs.add(op1)
    elif op1 in xmm_regs and op2 in xmm_regs and '[' in op3:
        if op1 in tainted_regs:
            kept_flag = True
            tainted_regs.remove(op1)
            tainted_regs.add(op2)
            if read_op1:
                tainted_regs.add(op1)
            m_addr_list = split_addr_list(mem_addr, op3)
            for m_addr in m_addr_list:
                tainted_mems.add(m_addr)
    else:
        assert False, 'undefined {} {}'.format(opcode, operands)
    return kept_flag


def handle_vxor(opcode: str, operands: list, mem_addr: str):
    global tainted_regs
    op1 = operands[0].strip()
    op2 = operands[1].strip()
    op3 = operands[2].strip()
    if op2 == op3:  # op1 == op2 == op3:
        tainted_regs.remove(op1)
        return True
    else:
        return handle_three(opcode, operands, mem_addr)


def handle_not_implemented(opcode: str, operands: list):
    print('inst not implemented')
    print('{} {}'.format(opcode, operands))
    exit(0)


# ===============================================
#
# Interface
#
# ===============================================
def get_trace(asm_path: str, prog_path: str, data_path: str, log_path: str):
    asm_path = os.path.abspath(asm_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)
    log_path = os.path.abspath(log_path)

    rev_log, rnd_addr, loop_size, start_addr, end_addr = before_taint(asm_path, prog_path, data_path, log_path)
    slice_log = log_path.replace('.log', '_slice.log')

    target_addr = rnd_addr
    mem_list = []
    addr_int = int(target_addr, 16)
    size = loop_size
    for step in range(0, size, 4):  # the smallest unit is 4 bytes
        m_addr = hex(addr_int + step)
        mem_list.append(m_addr)
    set_tainted(mem_list)
    reverse_taint(rev_log, slice_log)
    return slice_log, rnd_addr, loop_size, start_addr, end_addr


if __name__ == '__main__':
    # test
    asm_path = '/home/lifter/Documents/tvm_output/vgg16_glow/vgg16_glow_ida/0010.txt'
    prog_path = '/home/lifter/Documents/tvm_output/vgg16_glow/vgg16_strip.out'
    data_path = '/home/lifter/Documents/tvm_output/cat.bin'
    log_path = 'tmp_trace.log'
    slice_log, rnd_addr, loop_size, start_addr, end_addr = get_trace(asm_path, prog_path, data_path, log_path)

    """
    #mem_list = ['0x214fcdc0', '0x214fcdc4']
    mem_list = []
    addr_int = int('0x214fcdc0', 16)
    size = 300*4
    addr_int += size*64
    for step in range(0, size, 4):  # the smallest unit is 4 bytes
        m_addr = hex(addr_int + step)
        mem_list.append(m_addr)
    set_tainted(mem_list)
    reverse_taint('./traces/0x4017b0-0x401e91_reverse.log', './traces/0x4017b0-0x401e91_slice.log')
    """
    if len(sys.argv) == 5:
        reverse_log = sys.argv[1]
        slice_log = sys.argv[2]
        target_addr = sys.argv[3]
        length = int(sys.argv[4])
        # print('debug {} {} {} {}'.format(reverse_log, slice_log, target_addr, length))
        # exit(0)

        mem_list = []
        addr_int = int(target_addr, 16)
        size = length
        for step in range(0, size, 4):  # the smallest unit is 4 bytes
            m_addr = hex(addr_int + step)
            mem_list.append(m_addr)
        set_tainted(mem_list)
        reverse_taint(reverse_log, slice_log)
    else:
        print('Usage: this_script.py <input_reverse.log> <output_slice.log> <target address> <length/size>')
        reverse_log = './0x4029c0-0x4030f9_reverse.log'
        slice_log = '0x4029c0-0x4030f9_slice.log'
        target_addr = '0x32354f0'
        length = 256
        # print('debug {} {} {} {}'.format(reverse_log, slice_log, target_addr, length))
        # exit(0)

        mem_list = []
        addr_int = int(target_addr, 16)
        size = length
        for step in range(0, size, 4):  # the smallest unit is 4 bytes
            m_addr = hex(addr_int + step)
            mem_list.append(m_addr)
        set_tainted(mem_list)
        reverse_taint(reverse_log, slice_log)
