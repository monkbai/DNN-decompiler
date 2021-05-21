import os
import sys
import copy


def print_taget_lines(trace_log: str, addr: str):
    final_bufs = []

    def print_buf(r_buf: list):
        if len(r_buf) < 1:
            return False
        if r_buf[0][0].startswith('0x') and addr in r_buf[1][0]:
            for line, id in r_buf:
                print('{:<10}: {}'.format(id, line))

            return True
        return False

    trace_log_path = os.path.abspath(trace_log)
    with open(trace_log_path, 'r') as f:
        read_buf = []
        idx = 0
        while True:
            line = f.readline()
            idx += 1

            if not line:
                break

            if idx % 1000000 == 0:  # million
                print(idx)

            if line.startswith('0x'):
                if print_buf(read_buf):
                    final_bufs.append(copy.deepcopy(read_buf))
                read_buf.clear()
                read_buf.append((line, idx))
            else:
                read_buf.append((line, idx))
        for rbuf in final_bufs:
            for line, id in rbuf:
                print('{:<10}: {}'.format(id, line))


def count_lines(trace_log: str):
    trace_log_path = os.path.abspath(trace_log)
    with open(trace_log_path, 'r') as f:
        read_buf = []
        idx = 0
        while True:
            line = f.readline()
            idx += 1

            if not line:
                break

            if idx % 1000000 == 0:  # million
                print(idx)
        print(idx)


def count_addrs(trace_log: str):
    trace_log_path = os.path.abspath(trace_log)
    with open(trace_log_path, 'r') as f:
        read_buf = []
        idx = 0
        while True:
            line = f.readline()
            idx += 1

            if not line:
                break

            # if idx % 1000000 == 0:  # million
            #    print(idx)
            if line.startswith('0x401e4a'):
                print(line)
            elif line.startswith('0x401e67'):
                print(line)
            elif line.startswith('0x401e7a'):
                print(line)
        print(idx)


def reverse_trace(trace_log: str, new_trace: str):
    trace_log_path = os.path.abspath(trace_log)
    new_trace_path = os.path.abspath(new_trace)
    old_f = open(trace_log_path, 'rb')
    new_f = open(new_trace_path, 'w')

    idx = 0
    end_flag = False
    final_bufs = []
    read_buf = []

    old_f.seek(-2, os.SEEK_END)
    while True:
        while old_f.read(1) != b'\n':
            old_f.seek(-2, os.SEEK_CUR)  # find the last \n
            if old_f.tell() == 0:
                end_flag = True
                break
        last_read = old_f.tell()
        # print('last read', last_read)
        last_line = old_f.readline().decode()
        if not end_flag:
            old_f.seek(last_read - 2)
        line = last_line
        idx += 1

        if idx % 1000 == 0:  # debug
            print(idx)

        # print(line)
        if line.startswith('0x'):
            read_buf.append(line)
            read_buf.reverse()
            for line in read_buf:
                new_f.write(line)
            read_buf.clear()
        else:
            read_buf.append(line)

        if end_flag:
            break

    old_f.close()
    new_f.close()


def reverse_trace_mem(trace_log: str, new_trace: str):
    """
        This function will consume lots of memory.
    """
    trace_log_path = os.path.abspath(trace_log)
    new_trace_path = os.path.abspath(new_trace)
    old_f = open(trace_log_path, 'r')
    new_f = open(new_trace_path, 'w')

    old_lines = old_f.readlines()
    print('reading finished')
    read_buf = []
    idx = len(old_lines) - 1
    line_count = 0
    while idx >= 0:
        line = old_lines[idx]
        idx -= 1

        line_count += 1
        if line_count % 1000000 == 0:  # million
            print(line_count)

        # print(line)
        if line.startswith('0x'):
            read_buf.append(line)
            read_buf.reverse()
            for line in read_buf:
                new_f.write(line)
            read_buf.clear()
        else:
            read_buf.append(line)

    old_f.close()
    new_f.close()


def cut_trace(trace_log: str, new_trace: str):
    trace_log_path = os.path.abspath(trace_log)
    new_trace_path = os.path.abspath(new_trace)
    old_f = open(trace_log_path, 'r')
    new_f = open(new_trace_path, 'w')

    old_lines = old_f.readlines()
    print('reading finished')
    read_buf = []
    idx = 0
    line_count = 0
    start_flag = False
    while idx < len(old_lines):
        line = old_lines[idx]
        idx += 1

        line_count += 1
        if line_count % 1000000 == 0:  # million
            print(line_count)

        if not start_flag and idx >= 230000000 * 3 and line.startswith('0x'):
            start_flag = True
            new_f.write(line)
        elif start_flag:
            new_f.write(line)

    old_f.close()
    new_f.close()


# ---------- upper functions are deprecated ----------

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
        elif opcode.startswith('vmovss') or opcode.startswith('vmovups'):
            kept = handle_two(opcode, operands, mem_addr)
        elif opcode.startswith('vbroadcastss'):
            kept = handle_two(opcode, operands, mem_addr)
        elif opcode.startswith('vmaxss') or opcode.startswith('vaddss') or opcode.startswith('vmulss'):
            kept = handle_three(opcode, operands, mem_addr)
        elif opcode.startswith('vfmadd231ss') or opcode.startswith('vfmadd213ps'):
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
            assert False, 'error:{} undefined size'.format(op)
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


if __name__ == '__main__':
    # count_lines('./inst_trace.log')
    # count_addrs('./inst_trace.log')
    # print_taget_lines('./inst_trace.log', '0x214fd1a0')
    # reverse_trace('./inst_trace.log', './reverse_trace.log')  # slow
    # reverse_trace_mem('./inst_trace.log', './reverse_trace_mem.log')  # fast
    # cut_trace('./reverse_trace_mem.log', './cut_reverse_trace.log')
    # exit(0)

    # Do not use this script to reverse traces, it's slow
    # Using tac

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
