import os
import time


id2cat = {
    1: 'pad',
    2: 'contrib_conv2d',
    3: 'add',
    4: 'relu',
    5: 'max_pool2d',
    6: 'softmax',
    7: 'reshape',
    8: 'dense',
    9: 'conv2d_1',
    10: 'conv2d_2',
    11: 'conv2d_3',
}
cat2id = {
    'pad': 1,
    'contrib_conv2d': 2,
    'add': 3,
    'relu': 4,
    'max_pool2d': 5,
    'softmax': 6,
    'reshape': 7,
    'dense': 8,
    'conv2d_1': 9,
    'conv2d_2': 10,
    'conv2d_3': 11,
}

xmm_names = {
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

reg_names = {'ah', 'ch', 'dh', 'bh',
             'al', 'cl', 'dl', 'bl', 'spl', 'bpl', 'sil', 'dil',
             'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b',
             'ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di',
             'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w',
             'eax', 'ecx', 'edx' , 'ebx' , 'esp' , 'ebp' , 'esi' , 'edi' ,
             'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
             'rax', 'rcx', 'rdx' , 'rbx' , 'rsp' , 'rbp' , 'rsi' , 'rdi' ,
             'r8' , 'r9' , 'r10' , 'r11' , 'r12' , 'r13' , 'r14' , 'r15' , }


def get_op_type(operand: str):
    op = operand.strip()
    if op in xmm_names:
        return 'xmm'
    elif op in reg_names:
        return 'reg'
    elif '[' in op:
        return 'mem'
    else:
        # tmp = int(op, 16)
        return 'imm'


def get_category(f_name):
    for name, id in cat2id.items():
        if name in f_name:
            return id
    return -1


pattern_threshold = 0.5


# -----------------------------------------------
#

def patterns_generation(func_path: str, min_len: int, max_len: int, dataset_dir: str):
    func_dir, func_name = os.path.split(func_path)
    cat_id = get_category(func_name)

    pattern_list = []

    func_body = open(func_path, 'r').read()
    asm_list = get_asm_list(func_body)

    visited_patterns = []
    start_time = time.time()
    for length in range(min_len, max_len+1):
        start_index = 0
        while start_index < len(asm_list)-length:
            end_index = start_index + length
            # generate the potential pattern
            inst_list = []
            for idx in range(start_index, end_index):
                inst_list.append(parse_line(asm_list[idx]))
            pattern = ','.join(inst_list)

            # skip duplicated pattern
            if pattern not in visited_patterns:
                visited_patterns.insert(0, pattern)

                # evaluate the pattern
                score, ave_freq, all_ave_freq = evaluate_pattern(pattern, length, cat_id, dataset_dir)
                if score > pattern_threshold and ave_freq >= 3:
                    print('store the pattern', pattern)
                    pattern_list.append([(length, pattern, score, cat_id), ave_freq, all_ave_freq])

            start_index += 1
        current_time = time.time()
        print('length {}, time consumed {}'.format(length, current_time-start_time))
    pattern_list.sort(key=lambda x: x[1], reverse=True)
    return pattern_list


def evaluate_pattern(pattern: str, length: int, category, dataset_dir: str):
    files = os.listdir(dataset_dir)
    result = {}
    for f in files:
        if '.txt.' not in f:
            continue

        # print('evaluate on', f)
        cat_id = get_category(f)  # TODO check name
        f = os.path.join(dataset_dir, f)
        func_body = open(f, 'r').read()
        match_count = match_pattern(func_body, pattern, length)

        if cat_id not in result.keys():
            result[cat_id] = (1, match_count)
        else:
            old = result[cat_id]
            result[cat_id] = (old[0]+1, old[1]+match_count)
    # give a final confidence score
    average_list = []
    for cat_id, match_tuple in result.items():
        average_list.append((cat_id, match_tuple[1]/match_tuple[0]))
    average_list.sort(key=lambda x: x[1], reverse=True)
    if average_list[0][0] != category:
        return 0.0, 0, 0
    else:
        all_ave_match_count = sum(i for _, i in average_list)
        score = average_list[0][1] / all_ave_match_count
        # debug
        if score > 0.5 and average_list[0][1] >= 3:
            print('average count {}, score {}'.format(average_list[0][1], score))
        return score, average_list[0][1], all_ave_match_count


def get_asm_list(f_b: str):
    _asm_list = []
    fb_list = f_b.split('\n')
    for line in fb_list:
        if line.startswith(';'):
            continue
        asm_code = line[50:].strip()
        _asm_list.append(asm_code)
    return _asm_list


def parse_line(line: str):
    if line.find(' ') != -1:
        mnemonic, operands = line.split(' ', 1)
        operands_list = operands.split(',')
    else:
        mnemonic = line
        operands_list = []
    result = mnemonic
    for op in operands_list:
        result += ' ' + get_op_type(op)
    return result


def match_pattern(f_body: str, pattern: str, length: int):
    asm_list = get_asm_list(f_body)
    match_count = 0
    start_index = 0
    while start_index < len(asm_list):  # all func body
        end_index = start_index + length
        asm_snippet_list = asm_list[start_index:end_index]  # TODO check
        if _match(asm_snippet_list, pattern, length):
            match_count += 1
            start_index = end_index  # TODO check
            continue

        start_index += 1
    return match_count


def _match(asm_snippet_list, pattern_str, length):
    pat_list = pattern_str.split(',')
    for i in range(length):  # the pattern list
        # pre process pattern
        inst = pat_list[i]
        pat_op_list = inst.split(' ')
        # preprocess asm line
        line = asm_snippet_list[i]
        if line.find(' ') != -1:
            mnemonic, ops = line.split(' ', 1)
            asm_op_list = ops.split(',')
        else:
            mnemonic = line
            asm_op_list = []
        # check the mnemonic
        if (not mnemonic == pat_op_list[0]) or (len(asm_op_list) != len(pat_op_list)-1):
            return False
        # check operands
        for j in range(1, len(pat_op_list)):
            asm_op_type = get_op_type(asm_op_list[j - 1])
            if pat_op_list[j] != asm_op_type:
                return False
    return True


def test():
    patterns_generation('/home/lifter/Documents/tvm_output/O0/funcs/007.txt.fused_nn_conv2d_3', 3, 6, '/home/lifter/Documents/tvm_output/O0/funcs/')


if __name__ == '__main__':
    test()
