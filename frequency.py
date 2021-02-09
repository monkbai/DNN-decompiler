import collections
import os
from myUtils import get_range
from e9patch_tools import arith_inst_trace

program_path = '/home/lifter/e9patch/test/demo_static_O0_exe'
input_data = '/home/lifter/e9patch/test/number.bin'


def frequency_set(log_txt: str):
    def filt_freq_set(freq_set: dict):
        all_dict_list = []
        visited_keys = []
        key_list = list(freq_set.keys())
        key_list.sort(reverse=True)
        for k in key_list:
            if k in visited_keys:
                continue
            new_dict = dict()
            new_key_list = [k]
            for k_1 in key_list:
                last_k = new_key_list[len(new_key_list)-1]
                if k_1 <= last_k and last_k % k_1 == 0:
                    new_dict[k_1] = freq_set[k_1]
                    new_key_list.append(k_1)
                    visited_keys.append(k_1)
            all_dict_list.append(new_dict)
        return all_dict_list

    log_lines = log_txt.split('\n')
    freq_set = collections.OrderedDict()
    skip_set = []
    for line in log_lines:
        line = line.strip()
        if line in skip_set or len(line) < 10 or not line.startswith('00000'):
            continue
        else:
            skip_set.append(line)
        freq = log_txt.count(line)
        if freq in freq_set.keys():
            tmp = freq_set[freq]
            if line not in tmp:
                tmp.append(line)
                freq_set[freq] = tmp
        else:
            tmp = [line, ]
            freq_set[freq] = tmp
    # freq_set = sorted(freq_set)
    # print(freq_set)
    all_freq_set = filt_freq_set(freq_set)
    return all_freq_set


def get_shapes(freq_set, target_inst_prefix):
    final_shape_str = []
    shape_str = ''

    freq_nums = freq_set.keys()
    freq_nums = list(freq_nums)
    freq_nums.sort()
    freq_nums.insert(0, 1)
    shape_str += str(freq_nums[0])
    for i in range(1, len(freq_nums)):
        if freq_nums[i] % freq_nums[i - 1] != 0:
            print("not implemented!")
            exit(-1)
        shape_str += '*{}'.format(int(freq_nums[i] / freq_nums[i - 1]))
        inst_list = freq_set[freq_nums[i]]
        inst_str = ';'.join(inst_list)
        if target_inst_prefix[:-1] in inst_str:
            tmp_shape = shape_str
            tmp_shape += '('
            if target_inst_prefix.endswith('*'):
                tmp_shape += '{}*4 + {}'.format(inst_str.count(target_inst_prefix[:-1] + 'ps'),
                                                inst_str.count(target_inst_prefix[:-1] + 'ss'))
            else:
                tmp_shape += '{}*1'.format(inst_str.count(target_inst_prefix))
            tmp_shape += ')'
            final_shape_str.append(tmp_shape)
        elif i == len(freq_nums)-1:
            final_shape_str.append(shape_str)
    return final_shape_str


def pre_shape(asm_path: str, target_inst_prefix='mul*'):

    # instrumentation --> trace
    start_addr, end_addr = get_range(asm_path)
    current_dir = os.path.dirname(__file__)
    log_path = os.path.join(current_dir, './tmp.log')
    arith_inst_trace(program_path, input_data, start_addr, end_addr, log_path)

    # frequency set
    log_txt = open(log_path, 'r').read()
    freq_set = frequency_set(log_txt)
    if len(freq_set) < 1:
        return []

    # preliminary shape (according to the target instruction)
    all_shape_list = []
    for f_set in freq_set:
        all_shape_list += get_shapes(f_set, target_inst_prefix)
    return all_shape_list


def test():
    # log_txt = open('/home/lifter/e9patch/log.txt', 'r').read()
    # frequency_set(log_txt)
    shapes = pre_shape('/home/lifter/Documents/tvm_output/O0/funcs/007.txt.fused_nn_conv2d_1')
    print(shapes)


def main():
    asm_dir = '/home/lifter/Documents/tvm_output/O0/funcs/'
    file_list = os.listdir(asm_dir)
    file_list.sort()
    for f in file_list:
        if '.txt' in f:
            f_path = os.path.join(asm_dir, f)
            print('predict file {}'.format(f))

            # the function type should be got from name prediction
            func_type = f
            if 'conv2d' in func_type:
                shape_list = pre_shape(f_path, 'mul*')
            elif 'add' in func_type:
                shape_list = pre_shape(f_path, 'add*')
            elif 'relu' in func_type or 'max' in func_type:
                shape_list = pre_shape(f_path, 'max*')
            elif 'softmax' in func_type:
                shape_list = pre_shape(f_path, 'expf')
            else:
                shape_list = pre_shape(f_path)
            print(shape_list)
            print()


if __name__ == '__main__':
    # test()
    main()
