import os
import time


id2cat = {
    1: 'pad',
    2: 'transpose',
    3: 'add',
    4: 'relu',
    5: 'max_pool2d',
    6: 'softmax',
    7: 'reshape',
    8: 'dense',
    9: 'conv2d',

    10: 'batch_flatten',
    11: 'expand_dims',
    12: 'global_avg_pool2d',
    13: 'multiply',
    14: 'negative',
    15: 'divide',
    16: 'sqrt',
    17: 'lrn',  # Local Response Normalization
    18: 'concatenate',
    19: 'nn_avg_pool2d',
    20: 'minimum',
    21: 'maximum',
    22: 'split',
    23: 'mean',
    24: 'clip',
    25: 'squeeze',
    26: 'layout_transform',
}
cat2id = {
    # O0
    'pad': 1,
    'transpose': 2,
    'add': 3,
    'relu': 4,
    'max_pool2d': 5,
    'softmax': 6,
    'reshape': 7,
    'dense': 8,
    'conv2d': 9,
    'batch_flatten': 10,
    'expand_dims': 11,
    'global_avg_pool2d': 12,
    'multiply': 13,
    'negative': 14,
    'divide': 15,
    'sqrt': 16,
    'lrn': 17,  # Local Response Normalization
    'concatenate': 18,
    'nn_avg_pool2d': 19,
    'minimum': 20,
    'maximum': 21,
    'split': 22,
    'mean': 22,
    'clip': 23,
    'squeeze': 24,
    'layout_transform': 25,
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


catid_pattern_dict = {}  # value should be a list?

pattern_threshold = 0.5

# -----------------------------------------------


def extract_patterns(func_body: str, length: int):
    patterns_list = []
    asm_list = get_asm_list(func_body)
    start_index = 0
    while start_index < len(asm_list) - length:
        end_index = start_index + length
        # generate the potential pattern
        inst_list = []
        ignore_pattern = False
        for idx in range(start_index, end_index):
            if asm_list[idx].startswith('j'):
                ignore_pattern = True
            elif asm_list[idx].startswith('push'):
                ignore_pattern = True
            elif asm_list[idx].startswith('pop'):
                ignore_pattern = True
            elif asm_list[idx].startswith('nop'):
                ignore_pattern = True
            inst_list.append(parse_line(asm_list[idx]))
        pattern = ','.join(inst_list)
        if not ignore_pattern and ('ps' in pattern or 'ss' in pattern or 'xmm' in pattern or 'ymm' in pattern):
            patterns_list.append(pattern)
        start_index += 1
    # patterns_list.sort()
    return patterns_list


def prepare_patterns(dataset_dir: str, length: int):
    global catid_pattern_dict
    catid_pattern_dict.clear()
    file_num = 0
    for home, dirs, files in os.walk(dataset_dir):
        for f in files:
            if '.txt.' not in f:
                continue

            file_num += 1
            print('prepare patterns for', f, ',num', file_num)
            cat_id = get_category(f)  # TODO
            f = os.path.join(home, f)
            func_body = open(f, 'r').read()
            patterns_list = extract_patterns(func_body, length)
            if cat_id not in catid_pattern_dict.keys():
                catid_pattern_dict[cat_id] = patterns_list
            else:
                new_list = patterns_list + catid_pattern_dict[cat_id]
                catid_pattern_dict[cat_id] = new_list


# -----------------------------------------------


def patterns_generation(func_path: str, min_len: int, max_len: int, dataset_dir: str):
    func_dir, func_name = os.path.split(func_path)
    cat_id = get_category(func_name)

    pattern_list = []

    func_body = open(func_path, 'r').read()
    asm_list = get_asm_list(func_body)

    start_time = time.time()
    for length in range(min_len, max_len+1):

        prepare_patterns(dataset_dir, length)

        patterns = extract_patterns(func_body, length)

        # check pattern one by one
        patterns = list(set(patterns))
        for pattern in patterns:
            score, match_count, total_count = evaluate_pattern(pattern, length, cat_id)
            if score > 0.8:
                pattern_list.append((pattern, score, match_count, total_count))
        current_time = time.time()
        print('length {}, time consumed {}'.format(length, current_time-start_time))
    pattern_list.sort(key=lambda x: x[1], reverse=True)
    with open('patterns.log', 'w') as f:
        for pat in pattern_list:
            f.write(str(pat)+'\n')
    return pattern_list


def evaluate_pattern(pattern: str, length: int, category):
    result = {}  # [cat_id]=(match_count)
    for catid, pattern_list in catid_pattern_dict.items():
        count = pattern_list.count(pattern)
        if catid in result.keys():
            result[catid] += count
        else:
            result[catid] = count

    # give a final confidence score
    total_count = 0
    for cat_id, match_count in result.items():
        total_count += match_count
    if category not in result.keys():
        return 0.0, 0, total_count
    else:
        if total_count != 0:
            score = result[category] / total_count
        else:
            score = 0.0
        return score, result[category], total_count


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


# -----------------------------------------------
#
def test_func_name(dataset_dir: str):
    for home, dirs, files in os.walk(dataset_dir):
        for f in files:
            cat_id = get_category(f)
            func_path = os.path.join(home, f)
            if cat_id == -1:
                print(func_path)


def test():
    patterns_generation('/home/lifter/Documents/tvm_output/O0/funcs/007.txt.fused_nn_conv2d_3', 3, 6, '/home/lifter/Documents/tvm_output/dataset/O0')


if __name__ == '__main__':
    test()
    # test_func_name('/home/lifter/Documents/tvm_output/dataset/O0')
    # test_func_name('/home/lifter/Documents/tvm_output/dataset/O3')
