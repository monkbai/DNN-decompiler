import re
import math
import collections


# -----------------------------------------------
# how to interpret the taint analysis result
# can we assume that we know the start addresses of inputs and output?
def explain_tvm_conv2d_result(mem_log_path: str):
    # assume the mem_log comes from a convolution layer

    # read the mem_log
    with open(mem_log_path, 'r') as f:
        mem_log = f.read()
    log_lines = mem_log.split('\n')
    index = 0
    mem_sta = collections.OrderedDict()
    while index < len(log_lines)-1:
        key = log_lines[index]
        index += 1
        value = log_lines[index]
        if key.startswith('0x7ff') or key.endswith('16'):
            index += 1
            continue
        mem_sta[key] = value.strip('<>')

        index += 1
    mem_list = list(mem_sta.items())
    # TODO: here assume width==height
    input_shape = [1, 1, 1, 1]
    filter_shape = [1, 1, 1, 1]
    output_shape = [1, 1, 1, 1]

    # get the filter shape and input shape from first output
    offset_list = get_offset_list(mem_list[0][1], compiler='tvm')  # analyze the first expression (with the smallest address)
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
    first_addr_list = get_addr_list(mem_list[0][1], compiler='tvm')
    for key, value in mem_list:
        current_addr_list = get_addr_list(value, compiler='tvm')
        if current_addr_list == first_addr_list:
            output_channel += 1

    output_shape[1] = output_channel
    filter_shape[0] = output_shape[1]
    output_shape[2] = input_shape[2] - filter_shape[2] + 1
    output_shape[3] = input_shape[3] - filter_shape[3] + 1

    # final shape
    print('input shape', input_shape)
    print('filter shape', filter_shape)
    print('output shape', output_shape)


def get_offset_list(value: str, compiler: str):
    times = value.count('*')
    if compiler == 'tvm':
        offset_list = get_addr_list(value, 'tvm')
    elif compiler == 'glow':
        offset_list = get_addr_list(value, 'glow')
    else:
        print('at get_offset_list')
        print('compiler not supported:', compiler)
        exit(-1)
        return
    offset_list.sort()
    start_addr = offset_list[0]
    for i in range(len(offset_list)):
        offset_list[i] = (offset_list[i] - start_addr) / 4
    return offset_list


input_on_the_left = True


def get_addr_list(value: str, compiler: str):
    global input_on_the_left
    """

    :param value: the expression
    :param compiler: 'tvm' or 'glow'
    :return: list of used input addresses
    """
    addr_list = []
    if compiler == 'tvm':
        it = re.finditer(r'(0x[0-9a-f]+),4 \*', value)
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


if __name__ == '__main__':
    explain_tvm_conv2d_result('./mem_log.txt')
