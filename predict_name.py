import os
import collections

categories = {
    1: 'pad',
    2: 'conv',
    3: 'add',
    4: 'relu',
    5: 'max_pool',
    6: 'softmax',
    7: 'reshape',
    8: 'dense',
}

patterns = {
    # name: length, pattern, score, category
    # from O0
    'max_pool2d_1_?': (3, 'lea,movsxd,maxss', 0.6, 5),
    'add_1_?': (3, 'movaps,addps,movaps', 0.3, 3),
    'add_2_?': (3, 'movss,addss,movss', 0.6, 3),
    'pad_1_?': (3, 'lea,movsxd,mov 0', 0.6, 1),
    'pad_2_?': (4, 'lea,movsxd,movss,cdqe', 0.6, 1),
    'pad_3_?': (3, 'lea,cdqe,mov 0', 0.6, 1),
    'conv2d_a1_?': (4, 'mov,or,movups,movups', 0.3, 2),
    'conv2d_b1_?': (5, 'mov,lea,sar,mov,mov', 0.3, 2),
    'conv2d_c1_?': (10, 'mov,lea,movsxd,movss,shufps 0,movaps,mulps,movaps,mulps,movaps', 0.9, 2),
    'relu_1_?': (3, 'movss,maxss,movss', 0.6, 4),
    'pad_4_?': (2, 'movaps,movups', 0.3, 1),
    'pad_5_?': (2, 'movups,movaps', 0.3, 1),
    'pad_6_?': (1, 'movqps xmm0,movaps xmm0', 0.6, 1),
    'conv2d_a2_?': (3, 'lea,movsxd,movups', 0.3, 2),
    'conv2d_b2_?': (4, 'lea,movsxd,mov,mov', 0.3, 2),
    'conv2d_c2_?': (8, 'mov,movss,shufps 0,movaps,mulps,movaps,mulps,movaps', 0.6, 2),
    'conv2d_c3_?': (6, 'movss,shufps 0,movaps,mulps,movaps,mulps', 0.9, 2),
    'add_3_?': (3, 'movups,addps,movups', 0.6, 3),
    'relu_2_?': (3, 'movups,maxps,movups', 0.6, 4),
    'add_4_?': (6, 'mov,lea,sar,movss,addss,movss', 0.6, 3),
    'softmax_?': (1, 'call <expf>', 0.9, 6),
    'dense_1_?': (4, 'addps,addps,addps,addps', 0.6, 8),
    'dense_2_?': (4, 'addss,addss,addss,addss', 0.6, 8),
    # 'dense_3_?': (4, 'add,add,add,add'),
    'dense_4_?': (3, 'movaps,shufps,addss', 0.3, 8),
    'max_pool2d_2_?': (4, 'mov,lea,sar,maxss', 0.6, 5),
    'max_pool2d_3_?': (3, 'lea,sar,maxss', 0.6, 5),
    'reshape_1_?': (4, 'movaps,movaps,movaps,movaps', 0.3, 7),
    # from O3
    'O3_max_pool2d_4_?': (6, 'lea,movsxd,movups,shl,movaps,maxaps', 0.6, 5),
    'O3_max_pool2d_5_?': (3, 'movups,movaps,maxps', 0.6, 5),
    'O3_max_pool2d_6_?': (5, 'lea,movsxd,movups,shl,maxaps', 0.6, 5),
    'O3_max_pool2d_7_?': (3, 'or,movups,maxps', 0.6, 5),
    'O3_layout_reshape_2_?': (6, 'por,pshufd,movq,shr,movq,punpckhdq', 0.3, 7),
    'O3_conv2d_1_?': (10, 'mov,lea,movsxd,movsss,shufps 0,movaps,mulps,movaps,mulps,movaps', 0.9, 2),
    'O3_conv2d_relu_1_?': (6, 'lea,movss,addss,maxss,movsxd,movss', 0.6, 4),
    'O3_pad_layout_1_?': (4, 'mov 0,mov 0,mov 0,mov 0', 0.6, 1),
    # 'O3_pad_7_?': (3, 'lea,cdqe,mov 0', 0.6),
}


def predict_name(func_body: str):
    def get_asm_list(f_b: str):
        _asm_list = []
        fb_list = f_b.split('\n')
        for line in fb_list:
            if line.startswith(';'):
                continue
            asm_code = line[50:].strip()
            _asm_list.append(asm_code)
        return _asm_list

    def match_pat(pat: str, asm_code: str):
        asm_code += ' '
        if ' ' in pat:
            p_l = pat.split(' ')
            if len(p_l) > 2:
                print('not implemented!')
                return False
            if asm_code.startswith(p_l[0]+' ') and ' '+p_l[1] in asm_code:
                return True
        else:
            if asm_code.startswith(pat+' '):
                return True
        return False

    asm_list = get_asm_list(func_body)

    predict_result = []
    final_result = {}
    for name, pattern in patterns.items():
        count = 0
        pat_len = pattern[0]
        pat_list = pattern[1].split(',')
        pat_score = pattern[2]
        pat_category = pattern[3]
        index = 0
        while index < len(asm_list):
            # for index in range(len(asm_list)):
            match_flag = True
            for pat_index in range(pat_len):
                if not match_pat(pat_list[pat_index], asm_list[index + pat_index]):
                    match_flag = False
                    break
            if match_flag:
                count += 1
                index += (pat_len - 1)
            index += 1
        if count > 0:
            predict_result.append((name, count, count*pat_len*pat_score))
            if pat_category not in final_result.keys():
                final_result[pat_category] = count*pat_len*pat_score
            else:
                final_result[pat_category] += count * pat_len * pat_score

    # sort
    predict_result.sort(key=lambda x: x[2])
    predict_result.reverse()
    print(predict_result)

    result = sorted(final_result.items(), key=lambda x: x[1], reverse=True)
    print('final predication:', categories[result[0][0]])


def main():
    asm_dir = '/home/lifter/Documents/tvm_output/O3/funcs/'
    file_list = os.listdir(asm_dir)
    file_list.sort()
    for f in file_list:
        if '.txt' in f:
            f_path = os.path.join(asm_dir, f)
            print('predict file {}'.format(f))
            func_body = open(f_path, 'r').read()
            predict_name(func_body)
            print()


if __name__ == '__main__':
    main()
