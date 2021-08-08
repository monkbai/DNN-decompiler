import os
import re


def get_op_list(func_asm: str):
    op_list = []
    with open(func_asm, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith(';'):
                continue
            op = l[44:].strip()
            if ' ' in op:
                op = op[:op.find(' ')]
            if op.startswith('db'):
                continue
            op_list.append(op)
    return op_list


def is_glow_entry(func_asm: str):
    op_list = get_op_list(func_asm)
    index = 0
    call_count = 0
    entry_flag = False
    while index < len(op_list):
        if op_list[index].startswith('call'):
            if op_list[index-1]=='mov' and op_list[index-2]=='mov':
                call_count += 1
        index += 1
    if call_count > len(op_list)/7:
        entry_flag = True
        for op in op_list:
            if op.startswith('j'):
                entry_flag = False
    return entry_flag


def is_glow(funcs_dir: str, string_txt: str):
    files = os.listdir(funcs_dir)
    for f in files:
        if f.endswith('.txt') and 'label' not in f:
            f = os.path.join(funcs_dir, f)
            if f.endswith('0008.inception_v1.txt'):
                print('debug')
            if is_glow_entry(f):
                print(os.path.basename(f))
                return True
    return False


def is_tvm_alloca(func_asm: str):
    op_list = get_op_list(func_asm)
    pattern_list = ['sub'] + ['mov']*5 + ['call', 'add', 'retn']
    if op_list == pattern_list:
        return True
    pattern_list = ['mov'] * 6 + ['jmp']
    if op_list == pattern_list:
        return True
    return False


def is_tvm_opt(string_txt: str):
    opt = False
    with open(string_txt, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith('Assert fail: (num_args ==') and 'concat' not in l:
                match = re.search('num_args == ([0-9]+)', l)
                if match:
                    num_arg = int(match.group(1))
                    if num_arg > 3:
                        opt = True
                        break
    return opt


def is_tvm(funcs_dir: str, string_txt: str):
    tvm = False
    opt = False
    files = os.listdir(funcs_dir)
    for f in files:
        if f.endswith('.txt') and 'label' not in f:
            f = os.path.join(funcs_dir, f)
            if is_tvm_alloca(f):
                tvm = True
                break
    if tvm:
        opt = is_tvm_opt(string_txt)
    return tvm, opt


def main(root_dir: str):
    files = os.listdir(root_dir)
    for f in files:
        if f.endswith('.txt'):
            print(f[:-4])
            f_path = os.path.join(root_dir, f)
            funcs_dir = os.path.join(root_dir, f[:-4]+'_funcs')
            print('is glow?')
            print(is_glow(funcs_dir, f_path))
            print('is tvm? opt?')
            print(is_tvm(funcs_dir, f_path))


if __name__ == '__main__':
    glow_dir = "/home/lifter/Documents/tvm_output/scripts/ida/TVM_binaries/GLOW_binaries"
    tvm_o0_dir = "/home/lifter/Documents/tvm_output/scripts/ida/TVM_binaries/TVM_binaries/O0"
    tvm_o3_dir = "/home/lifter/Documents/tvm_output/scripts/ida/TVM_binaries/TVM_binaries/O3"
    main(glow_dir)
    main(tvm_o0_dir)
    main(tvm_o3_dir)
