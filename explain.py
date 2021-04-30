import os
import re
import math
import collections


# ==============================================================
# Utils
# ==============================================================
def choose_one_4bytes(exp_log_path: str):
    """ Choose one expression from the exp_log to recover the filter shape """
    exp_log_path = os.path.abspath(exp_log_path)
    with open(exp_log_path, 'r') as f:
        exp_txt = f.read()
        lines = exp_txt.split('\n')
        f.close()
        index = 0
        length = len(lines)
        name = ''
        exp = ''
        while index < length-2:
            name = lines[index]
            index += 1
            exp = lines[index].strip('<>')
            index += 1
            if name.endswith('16') or name.startswith('0x7ff'):
                continue
            else:  # choose the first expression of one 4 bytes memory block
                return name, exp
        return '', ''


def choose_one_16bytes(exp_log_path: str):
    exp_log_path = os.path.abspath(exp_log_path)
    with open(exp_log_path, 'r') as f:
        exp_txt = f.read()
        lines = exp_txt.split('\n')
        f.close()
        index = 0
        length = len(lines)
        name = ''
        exp = ''
        while index < length-2:
            name = lines[index]
            index += 1
            exp = lines[index].strip('<>')
            index += 1
            if (not name.endswith('16')) or name.startswith('0x7ff'):
                continue
            elif exp_txt.count(name) > 1:
                continue
            else:  # choose the first expression of one 4 bytes memory block
                return name, exp
        return name, exp


def get_output_channel(exp: str, one_channel_size: int, mem_regions: list, compiler='tvm', on_the_right=True):
    # get one weight address
    """
    if compiler == 'tvm' and on_the_right:
        it = re.search(r'\* (0x[0-9a-f]+),16\)', exp)
        addr_str = it.group(1)
        addr = int(addr_str, 16)
    elif compiler == 'tvm' and not on_the_right:
        it = re.search(r'(0x[0-9a-f]+),16 \*', exp)
        addr_str = it.group(1)
        addr = int(addr_str, 16)
    for mem_blk in mem_regions:
        if mem_blk[0] <= addr <= mem_blk[1]:
            weights_region = mem_blk
            break
    output_channel = (weights_region[1] - weights_region[0]) / 4 / filter_size
    """
    big_mem = (0, 0)
    if compiler == 'tvm':
        for mem_blk in mem_regions:
            if (mem_blk[1]-mem_blk[0]) > (big_mem[1]-big_mem[0]):
                big_mem = mem_blk
    output_channel = ((big_mem[1] - big_mem[0]) / one_channel_size) / 4
    return big_mem, output_channel


def get_input_shape(name, exp, mem_regions, input_channel, size):
    offset_list = get_addr_list(exp, 'tvm', size)
    input_start_addr = min(offset_list)
    print(hex(input_start_addr))
    for mem_start, mem_end in mem_regions:
        if mem_start <= input_start_addr < mem_end:
            return math.sqrt(((mem_end-input_start_addr)/input_channel)/4)


# ==============================================================
# Heuristics used to recover shape for TVM Conv2d
# ==============================================================


# -----------------------------------------------
# how to interpret the taint analysis result
# can we assume that we know the start addresses of inputs and output?
def explain_tvm_conv2d_result(exp_log_path: str, mem_read_regions: list, mem_write_regions: list):
    # assume the mem_log comes from a convolution layer

    name, exp = choose_one_4bytes(exp_log_path)
    if len(name) == 0:
        name, exp = choose_one_16bytes(exp_log_path)
        return explain_tvm_conv2d_result_16(name, exp, mem_read_regions, mem_write_regions)
    mem_list = [(name, exp)]

    # TODO: here assume width==height
    input_shape = [1, 1, 1, 1]
    filter_shape = [1, 1, 1, 1]
    output_shape = [1, 1, 1, 1]

    # get the filter shape and input shape from first output
    offset_list = get_offset_list(mem_list[0][1], compiler='tvm')  # analyze the first expression (with the smallest address)
    stride = offset_list[1] - offset_list[0]
    index = 0
    while index < len(offset_list) - 1:
        if offset_list[index+1] - offset_list[index] > stride:
            tmp1 = offset_list[index + 1]  # input[1] * input[2]
            tmp2 = offset_list[index] + stride  # filter[1] * filter[2]
            filter_shape[3] = len(offset_list)/tmp2
            filter_shape[2] = filter_shape[3]  # TODO assume
            filter_shape[1] = tmp2/filter_shape[2]
            # input[1] = filter[1]
            input_shape[1] = filter_shape[1]
            input_shape[2] = tmp1 / input_shape[1]
            input_shape[3] = input_shape[2]  # TODO assume
            break
        elif offset_list[index+1] - offset_list[index] < stride:
            filter_shape[3] = index + 1
            filter_shape[2] = filter_shape[3]  # TODO assume
            filter_shape[1] = len(offset_list) / (filter_shape[2] * filter_shape[3])
            # input[1] = filter[1]
            input_shape[1] = filter_shape[1]
            input_shape[2] = input_shape[3] = get_input_shape(name, exp, mem_read_regions, input_shape[1], 4)
            break
        index += 1

    output_shape[2] = input_shape[2] - filter_shape[2] + 1
    output_shape[3] = input_shape[3] - filter_shape[3] + 1
    # get output shape
    # TODO: cannot get output_channel easily, because we do not have all mem_log (too huge and too slow)
    # filter_size = filter_shape[1] * filter_shape[2] * filter_shape[3]
    one_channel_size = output_shape[2] * output_shape[3]
    weights_region, output_channel = get_output_channel(mem_list[0][1], one_channel_size, mem_write_regions, compiler='tvm')
    # XXX: Did I made it ?

    output_shape[1] = output_channel
    filter_shape[0] = output_shape[1]
    output_shape[2] = input_shape[2] - filter_shape[2] + 1
    output_shape[3] = input_shape[3] - filter_shape[3] + 1

    # final shape
    print('input shape', input_shape)
    print('filter shape', filter_shape)
    print('output shape', output_shape)
    return filter_shape, input_shape, output_shape


def explain_tvm_conv2d_result_16(name: str, exp: str, mem_read_regions: list, mem_write_regions: list):
    mem_list = [(name, exp)]

    # TODO: here assume width==height
    input_shape = [1, 1, 1, 1]
    filter_shape = [1, 1, 1, 1]
    output_shape = [1, 1, 1, 1]

    offset_list = get_offset_list(mem_list[0][1], compiler='tvm', size=16)
    # print(offset_list)
    stride = offset_list[1] - offset_list[0]
    index = 0
    while index < len(offset_list) - 1:
        if offset_list[index+1] - offset_list[index] != stride:
            tmp1 = offset_list[index + 1]  # input[1] * input[2]
            tmp2 = offset_list[index] + stride  # filter[1] * filter[2]
            filter_shape[3] = index + 1
            filter_shape[2] = filter_shape[3]  # TODO assume
            filter_shape[1] = len(offset_list) / (filter_shape[2] * filter_shape[3])
            # input[1] = filter[1]
            input_shape[1] = filter_shape[1]
            input_shape[2] = input_shape[3] = get_input_shape(name, exp, mem_read_regions, input_shape[1], 16)
            break
        index += 1

    output_shape[2] = input_shape[2] - filter_shape[2] + 1
    output_shape[3] = input_shape[3] - filter_shape[3] + 1
    # get output shape
    # TODO: cannot get output_channel easily, because we do not have all mem_log (too huge and too slow)
    # filter_size = filter_shape[1] * filter_shape[2] * filter_shape[3]
    one_channel_size = output_shape[2] * output_shape[3]
    weights_region, output_channel = get_output_channel(mem_list[0][1], one_channel_size, mem_write_regions, compiler='tvm', on_the_right=False)
    output_shape[1] = output_channel
    filter_shape[0] = output_shape[1]

    # final shape
    print('input shape', input_shape)
    print('filter shape', filter_shape)
    print('output shape', output_shape)
    return filter_shape, input_shape, output_shape


def get_offset_list(value: str, compiler: str, size=4):
    times = value.count('*')
    if compiler == 'tvm':
        offset_list = get_addr_list(value, 'tvm', size)
    elif compiler == 'glow':
        offset_list = get_addr_list(value, 'glow', size)
    else:
        print('at get_offset_list')
        print('compiler not supported:', compiler)
        exit(-1)
        return
    start_addr = min(offset_list)
    for i in range(len(offset_list)):
        offset_list[i] = (offset_list[i] - start_addr) / 4
    if size == 4:
        offset_list.reverse()  # offset_list.sort()
    elif size == 16:
        offset_list.reverse()
    return offset_list


input_on_the_left = True


def get_addr_list(value: str, compiler: str, size=4):
    global input_on_the_left
    """

    :param value: the expression
    :param compiler: 'tvm' or 'glow'
    :return: list of used input addresses
    """
    addr_list = []
    if compiler == 'tvm' and size == 4:
        it = re.finditer(r'(0x[0-9a-f]+),4 \*', value)
        for match in it:
            addr = match.group(1)
            addr_list.append(int(addr, 16))
        return addr_list
    if compiler == 'tvm' and size == 16:
        it = re.finditer(r'\* (0x[0-9a-f]+),4', value)
        for match in it:
            addr = match.group(1)
            addr_list.append(int(addr, 16))
        return addr_list
    elif compiler == 'glow':
        # assume the input is on the left
        if input_on_the_left:
            it = re.finditer(r'(0x[0-9a-f]+),4 \*', value)
            for match in it:
                addr = match.group(1)
                addr_list.append(int(addr, 16))
            addr_list.sort()
        else:
            # input on the right
            addr_list.clear()
            it = re.finditer(r'\* (0x[0-9a-f]+),4', value)
            for match in it:
                addr = match.group(1)
                addr_list.append(int(addr, 16))
            addr_list.sort()
        if addr_list[-1] - addr_list[0] == (len(addr_list) - 1) * 4:
            input_on_the_left = False
            addr_list = get_addr_list(value, compiler)
        return addr_list


# ==============================================================
# Heuristics used to recover shape for Glow Conv2d
# ==============================================================
def explain_glow_conv2d_result(mem_log_path: str):
    # read the mem_log
    with open(mem_log_path, 'r') as f:
        mem_log = f.read()
    log_lines = mem_log.split('\n')
    index = 0
    mem_sta = collections.OrderedDict()
    longest_expression = [('', ''), ]
    while index < len(log_lines)-1:
        key = log_lines[index]
        index += 1
        value = log_lines[index]
        if key.startswith('0x7ff') or key.endswith(',32'):
            index += 1
            continue
        mem_sta[key] = value.strip('<>')

        # looking for the longest expression
        if len(mem_sta[key]) > len(longest_expression[0][1]):
            longest_expression[0] = (key, mem_sta[key])

        index += 1
    mem_list = list(mem_sta.items())
    # TODO: here assume width==height
    input_shape = [1, 1, 1, 1]
    filter_shape = [1, 1, 1, 1]
    output_shape = [1, 1, 1, 1]

    # get the filter shape and input shape from first output
    offset_list = get_offset_list(longest_expression[0][1], compiler='glow')
    stride = offset_list[1]-offset_list[0]
    index = 0
    while index < len(offset_list) - 1:
        if offset_list[index+1] - offset_list[index] > stride:
            tmp1 = offset_list[index + 1]  # input[1] * input[2]
            tmp2 = offset_list[index] + stride  # filter[1] * filter[2]
            filter_shape[3] = len(offset_list)/tmp2
            filter_shape[2] = filter_shape[3]  # TODO assume
            filter_shape[1] = tmp2/filter_shape[2]
            # input[1] = filter[1]
            input_shape[1] = filter_shape[1]
            input_shape[2] = tmp1 / input_shape[1]
            input_shape[3] = input_shape[2]  # TODO assume
            break
        index += 1

    # get output shape
    output_channel = 0
    first_addr_list = get_addr_list(longest_expression[0][1], compiler='glow')
    for key, value in mem_list:
        current_addr_list = get_addr_list(value, compiler='glow')
        if current_addr_list == first_addr_list:
            output_channel += 1

    output_shape[1] = output_channel
    filter_shape[0] = output_shape[1]
    # without padding
    output_shape[2] = math.sqrt(len(mem_list)/output_shape[1])
    output_shape[3] = output_shape[2]

    # final shape
    print('input shape', input_shape)
    print('filter shape', filter_shape)
    print('output shape', output_shape)


# ==============================================================
# Heuristics used to recover shape for TVM dense/fully-connected layer
# ==============================================================
def explain_tvm_dense_result(exp_log_path: str, mem_write_regions: list):
    # assume the mem_log comes from a convolution layer

    name, exp = choose_one_4bytes(exp_log_path)
    if len(name) == 0:
        exit(-1)

    input_size = exp.count('*') * 4
    output_size = 0
    big_mem = (0, 0)
    for mem_blk in mem_write_regions:
        if (mem_blk[1] - mem_blk[0]) > (big_mem[1] - big_mem[0]):
            big_mem = mem_blk
    output_size = (big_mem[1] - big_mem[0]) / 4
    return input_size, output_size


if __name__ == '__main__':
    pass
    # explain_tvm_conv2d_result('./mem_log.txt')
