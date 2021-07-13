import os
import re
import math
import collections


# ==============================================================
# Utils
# ==============================================================
def biggest_region(mem_regions: list, target_addr=0):
    big_mem = (0, 0)
    target_mem = (0, 0)
    for mem_blk in mem_regions:
        if (mem_blk[1] - mem_blk[0]) > (big_mem[1] - big_mem[0]):
            big_mem = mem_blk
        if mem_blk[0] <= target_addr <= mem_blk[1]:
            target_mem = mem_blk
    if target_addr == 0:
        return big_mem
    else:
        return target_mem


def choose_one_4bytes(exp_log_path: str, mem_write_regions=[], num=0):
    """ Choose one expression from the exp_log to recover the filter shape """
    if len(mem_write_regions) == 0:
        out_mem = (0, 0xffffffff)
    else:
        out_mem = biggest_region(mem_write_regions)
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
            if (not name.endswith(',4')) or name.startswith('0x7ff'):
                continue
            else:  # choose the first expression of one 4 bytes memory block
                if out_mem[0] <= int(name.split(',')[0], 16) <= out_mem[1]:
                    num -= 1
                if num < 0:
                    return name, exp
        return '', ''


def choose_one_16bytes(exp_log_path: str, mem_write_regions: list, num=0):
    if len(mem_write_regions) == 0:
        out_mem = (0, 0xffffffff)
    else:
        out_mem = biggest_region(mem_write_regions)
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
                if out_mem[0] <= int(name.split(',')[0], 16) <= out_mem[1]:
                    num -= 1
                if num < 0:
                    return name, exp
        return name, exp


def choose_one_bytes(exp_log_path: str, mem_write_regions: list, size=4, num=0):
    out_mem = biggest_region(mem_write_regions)
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
            if (not name.endswith(str(size))) or name.startswith('0x7ff'):
                continue
            elif exp.count('(') == 0:  # or exp_txt.count(name) > 1
                continue
            else:  # choose the first expression of one 4 bytes memory block
                if out_mem[0] <= int(name.split(',')[0], 16) <= out_mem[1]:
                    num -= 1
                if num < 0:
                    return name, exp
        return '', ''


def choose_one_max(exp_log_path: str, out_mem: tuple, size=4):
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
            name1 = lines[index]
            index += 1
            exp1 = lines[index].strip('<>')
            index += 1
            if (not name1.endswith(str(size))) or name1.startswith('0x7ff'):
                continue
            elif exp_txt.count(name1) > 1 or exp1.count('max') == 0:
                continue
            else:  # choose the first expression of one 4 bytes memory block
                if out_mem[0] <= int(name1.split(',')[0], 16) <= out_mem[1]:
                    if len(exp1) > len(exp):
                        exp = exp1
                        name = name1
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
    if compiler == 'tvm' or compiler == 'glow' :
        for mem_blk in mem_regions:
            if (mem_blk[1]-mem_blk[0]) > (big_mem[1]-big_mem[0]):
                big_mem = mem_blk
    output_channel = ((big_mem[1] - big_mem[0]) / one_channel_size) / 4
    return big_mem, output_channel


def get_input_shape(name, exp, mem_regions, input_channel, size):
    offset_list = get_addr_list(exp, 'tvm', size)
    input_start_addr = min(offset_list)
    print('input start addr', hex(input_start_addr))
    for mem_start, mem_end in mem_regions:
        if mem_start <= input_start_addr < mem_end:
            return math.sqrt(((mem_end-input_start_addr)/input_channel)/4)


# ==============================================================
# Heuristics used to recover shape for TVM Conv2d
# ==============================================================


# -----------------------------------------------
# how to interpret the taint analysis result
# can we assume that we know the start addresses of inputs and output?
def explain_tvm_conv2d_result(exp_log_path: str, mem_read_regions: list, mem_write_regions: list, guess_stride=1, optimized=False):
    # assume the mem_log comes from a convolution layer
    tmp_mem_write_regions = []
    if not optimized:
        tmp_mem_write_regions = mem_write_regions

    name, exp = choose_one_4bytes(exp_log_path, tmp_mem_write_regions)
    if len(name) == 0:
        name, exp = choose_one_16bytes(exp_log_path, tmp_mem_write_regions)  # TODO
        return explain_tvm_conv2d_result_16(name, exp, mem_read_regions, mem_write_regions, guess_stride, optimized)
    mem_list = [(name, exp)]
    mem_list.append(tuple(choose_one_4bytes(exp_log_path, tmp_mem_write_regions, 1)))

    # TODO: here assume width==height
    input_shape = [1, 1, 1, 1]
    filter_shape = [1, 1, 1, 1]
    output_shape = [1, 1, 1, 1]

    blk_size = 0
    if len(mem_read_regions) > 10:
        kernel_num, input_num, blk_size = kernel_1_1(name, exp, mem_read_regions, mem_write_regions)
        filter_shape[1] = kernel_num
        input_shape[1] = kernel_num
        input_shape[2] = input_shape[3] = input_num
        output_shape[2] = output_shape[3] = math.ceil(input_num/2)
        print('special case: stride 2')
    else:
        # get the filter shape and input shape from first output
        offset_list = get_offset_list(mem_list[0][1], compiler='tvm')  # analyze the first expression (with the smallest address)
        stride = offset_list[1] - offset_list[0]  # not the real stride
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
        """
        addr_list_0 = get_addr_list(mem_list[0][1], 'tvm', 4)
        if addr_list_0[0] > addr_list_0[-1]:
            addr_list_0.reverse()  # addr_list_0.sort()
        addr_list_1 = get_addr_list(mem_list[1][1], 'tvm', 4)
        if addr_list_1[0] > addr_list_1[-1]:
            addr_list_1.reverse()  # addr_list_1.sort()
        addr_1 = addr_list_1[0]
        # idx_0 = addr_list_0.index(addr_1)
        idx_0 = 0
        while idx_0 < len(addr_list_0):
            if addr_list_0[idx_0] >= addr_1:
                break
            idx_0 += 1
        if idx_0 == 1:
            stride = 1
        elif idx_0 <= 3:
            stride = idx_0  # / filter_shape[1]  # TODO: calculate the stride, can be wrong
        else:
            stride = idx_0 / filter_shape[1]
        """
        stride = guess_stride  # if we cannot get accurate stride, guess one

        output_shape[2] = math.ceil((input_shape[2] - filter_shape[2] + 1)/stride)
        output_shape[3] = math.ceil((input_shape[3] - filter_shape[3] + 1)/stride)
    # get output shape
    # TODO: cannot get output_channel easily, because we do not have all mem_log (too huge and too slow)
    # filter_size = filter_shape[1] * filter_shape[2] * filter_shape[3]
    one_channel_size = output_shape[2] * output_shape[3]
    weights_region, output_channel = get_output_channel(mem_list[0][1], one_channel_size, mem_write_regions, compiler='tvm')
    # XXX: Did I made it ?

    output_shape[1] = output_channel
    filter_shape[0] = output_shape[1]

    # since the stride and padding are guessed, we need to check if the shapes are reasonable
    ignore_flag = is_ignore(mem_list, mem_read_regions, filter_shape)

    if not ignore_flag:
        # try to get the weights layout indicators
        ind_a, ind_b, smooth = get_weights_layout_info(mem_list[0][1], mem_read_regions)
        # final shape
        print('input shape', input_shape)
        print('filter shape', filter_shape)
        print('output shape', output_shape)
        print('layout indicators: {}, {}'.format(ind_a, ind_b))
        if blk_size:  # kernel --> 1, 1
            ind_a = blk_size
            layout_shape = [filter_shape[0]/ind_b, filter_shape[1]/ind_a, filter_shape[2], filter_shape[3], ind_a, ind_b]
        elif not smooth:
            layout_shape = [filter_shape[0]/ind_b, filter_shape[1]/ind_a, filter_shape[2], filter_shape[3], ind_a, ind_b]
        else:
            layout_shape = [filter_shape[0]/ind_b, ind_a, filter_shape[2], filter_shape[3], filter_shape[1]/ind_a, ind_b]
        print('layout shape', layout_shape)
        return filter_shape, input_shape, output_shape, layout_shape
    else:
        print('not a reasonable guess, ignored')
        return (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0, 0)


def kernel_1_1(name, exp, mem_read_regions: list, mem_write_regions: list, compiler='tvm'):
    """ function to handle layer with 1*1 kernel """
    mem_start = 0x7f0000000000
    mem_end = 0
    target_size = dict()
    for mem_blk in mem_read_regions:
        mem_size = mem_blk[1] - mem_blk[0]
        if mem_size in target_size:
            target_size[mem_size] += 1
        else:
            target_size[mem_size] = 1

    target_list = list(target_size.items())
    target_list = sorted(target_list, key=lambda x: x[1])
    mem_size = target_list[-1][0]
    for mem_blk in mem_read_regions:
        if mem_blk[1] - mem_blk[0] == mem_size:
            if mem_blk[0] < mem_start:
                mem_start = mem_blk[0]
            if mem_blk[1] > mem_end:
                mem_end = mem_blk[1]
    offset_list = get_offset_list(exp, compiler=compiler)
    blk_size = 1
    stride = offset_list[1] - offset_list[0]
    while blk_size < len(offset_list):
        if offset_list[blk_size] - offset_list[blk_size - 1] != stride:
            break
        blk_size += 1

    kernel_num = len(offset_list)
    input_shape = math.sqrt((mem_end - mem_start + mem_size)/4/kernel_num)
    input_shape = math.ceil(input_shape)
    return kernel_num, input_shape, blk_size


def explain_tvm_conv2d_result_16(name: str, exp: str, mem_read_regions: list, mem_write_regions: list,
                                 guess_stride=1, optimized=False):
    mem_list = [(name, exp)]

    # TODO: here assume width==height
    input_shape = [1, 1, 1, 1]
    filter_shape = [1, 1, 1, 1]
    output_shape = [1, 1, 1, 1]

    blk_size = 0
    if len(mem_read_regions) > 10:
        kernel_num, input_num, blk_size = kernel_1_1(name, exp, mem_read_regions, mem_write_regions)
        filter_shape[1] = kernel_num
        input_shape[1] = kernel_num
        input_shape[2] = input_shape[3] = input_num
        output_shape[2] = output_shape[3] = math.ceil(input_num / 2)
        print('special case: stride 2')
    else:
        offset_list = get_offset_list(mem_list[0][1], compiler='tvm', size=16)
        # print(offset_list)
        stride = offset_list[1] - offset_list[0]
        index = 0
        while index < len(offset_list) - 1:
            if offset_list[index + 1] - offset_list[index] > stride:
                tmp1 = offset_list[index + 1]  # input[1] * input[2]
                tmp2 = offset_list[index] + stride  # filter[1] * filter[2]
                stride_2 = tmp1 - tmp2
                length = index
                while length < len(offset_list) - 1:
                    length += 1
                    tmp = offset_list[length]-offset_list[length - 1]
                    if tmp != stride and tmp != stride + stride_2:
                        break
                filter_shape[3] = length / tmp2
                filter_shape[2] = filter_shape[3]  # TODO assume
                filter_shape[1] = tmp2 / filter_shape[2]
                filter_shape[1] *= (len(offset_list) / length)
                # input[1] = filter[1]
                input_shape[1] = filter_shape[1]
                input_shape[2] = tmp1 / input_shape[1]
                input_shape[2] *= (len(offset_list) / length)
                input_shape[3] = input_shape[2]  # TODO assume
                break
            elif offset_list[index + 1] - offset_list[index] < stride:
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

        output_shape[2] = math.ceil((input_shape[2] - filter_shape[2] + 1) / guess_stride)
        output_shape[3] = math.ceil((input_shape[3] - filter_shape[3] + 1) / guess_stride)
    # get output shape
    # TODO: cannot get output_channel easily, because we do not have all mem_log (too huge and too slow)
    # filter_size = filter_shape[1] * filter_shape[2] * filter_shape[3]
    one_channel_size = output_shape[2] * output_shape[3]
    weights_region, output_channel = get_output_channel(mem_list[0][1], one_channel_size, mem_write_regions, compiler='tvm', on_the_right=False)
    output_shape[1] = output_channel
    filter_shape[0] = output_shape[1]

    ignore_flag = is_ignore(mem_list, mem_read_regions, filter_shape)

    if not ignore_flag:
        # try to get the weights layout indicators
        ind_a, ind_b, smooth = get_weights_layout_info(mem_list[0][1], mem_read_regions, size=16)
        # final shape
        print('input shape', input_shape)
        print('filter shape', filter_shape)
        print('output shape', output_shape)
        print('layout indicators: {}, {}'.format(ind_a, ind_b))
        if blk_size:  # kernel --> 1, 1
            ind_a = blk_size
            layout_shape = [filter_shape[0] / ind_b, filter_shape[1] / ind_a, filter_shape[2], filter_shape[3], ind_a,
                            ind_b]
        elif not smooth:
            layout_shape = [filter_shape[0] / ind_b, filter_shape[1] / ind_a, filter_shape[2], filter_shape[3], ind_a,
                            ind_b]
        else:
            layout_shape = [filter_shape[0] / ind_b, ind_a, filter_shape[2], filter_shape[3], filter_shape[1] / ind_a,
                            ind_b]
        print('layout shape', layout_shape)
        return filter_shape, input_shape, output_shape, layout_shape
    else:
        print('not a reasonable guess, ignored')
        return (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0, 0)


def is_ignore(mem_list: list, mem_read_regions: list, filter_shape: list):
    # since the stride and padding are guessed, we need to check if the shapes are reasonable
    weights_addrs = get_weights_addrs(mem_list[0][1], size=16)
    weights_mem = (0, 0)
    for mem_blk in mem_read_regions:
        if mem_blk[0] <= weights_addrs[0] <= mem_blk[1]:
            weights_mem = mem_blk

    ignore_flag = True
    if int(filter_shape[0]) == filter_shape[0]:
        weights_size = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3] * 4  # float --> 4 bytes
        if weights_size == weights_mem[1] - weights_mem[0]:
            # then it is a possible shape
            ignore_flag = False
    return ignore_flag


def get_offset_list(value: str, compiler: str, size=4, in_blk=(0, 0)):
    times = value.count('*')
    if compiler == 'tvm':
        offset_list = get_addr_list(value, 'tvm', size)
    elif compiler == 'glow':
        offset_list = get_addr_list(value, 'glow', size, in_blk=in_blk)
    else:
        print('at get_offset_list')
        print('compiler not supported:', compiler)
        exit(-1)
        return
    start_addr = min(offset_list)
    for i in range(len(offset_list)):
        offset_list[i] = (offset_list[i] - start_addr) / 4
    if size == 4 and offset_list[0] > offset_list[-1]:
        offset_list.reverse()  # offset_list.sort()
    elif size == 16 and offset_list[0] > offset_list[-1]:
        offset_list.reverse()
    return offset_list


input_on_the_left = True


def get_addr_list(value: str, compiler: str, size=4, in_blk=(0, 0)):
    global input_on_the_left
    """

    :param value: the expression
    :param compiler: 'tvm' or 'glow'
    :return: list of used input addresses
    """
    addr_list = []
    if compiler == 'tvm' and size == 4:
        match = re.search(r'(0x[0-9a-f]+),4 \*', value)
        if match:
            it = re.finditer(r'(0x[0-9a-f]+),4 \*', value)
            input_on_the_left = True
        else:
            it = re.finditer(r'\* (0x[0-9a-f]+),4', value)
            input_on_the_left = False
        for match in it:
            addr = match.group(1)
            addr_list.append(int(addr, 16))
        return addr_list
    if compiler == 'tvm' and size == 16:
        match = re.search(r'\* (0x[0-9a-f]+),4', value)
        if match:
            it = re.finditer(r'\* (0x[0-9a-f]+),4', value)
            input_on_the_left = False
        else:
            it = re.finditer(r'(0x[0-9a-f]+),4 \*', value)
            input_on_the_left = True
        for match in it:
            addr = match.group(1)
            addr_list.append(int(addr, 16))
        return addr_list
    elif compiler == 'glow':
        reg_str = r'((0x[0-9a-f]+,{} \*)|(\* 0x[0-9a-f]+,{}))'.format(size, size)
        # print(reg_str)  # debug
        it = re.finditer(reg_str, value)
        for match in it:
            addr = match.group(1).strip()
            if addr.startswith('0x'):
                addr = addr.split(',')[0]
            else:
                addr = addr[addr.find('0x'):]
                addr = addr.split(',')[0]
            addr = int(addr, 16)
            if in_blk[0] == 0 or in_blk[0] <= addr <= in_blk[1]:
                addr_list.append(addr)
        addr_list.sort()
        return addr_list


def get_weights_layout_info(value: str, mem_read_regions: list, compiler='tvm', size=4):
    weights_addrs = get_weights_addrs(value, size=16)
    weights_mem = (0, 0)
    for mem_blk in mem_read_regions:
        if mem_blk[0] <= weights_addrs[0] <= mem_blk[1]:
            weights_mem = mem_blk

    offset_list = get_weights_list(value, compiler=compiler, size=size)
    if offset_list[1] < offset_list[0]:
        offset_list.reverse()

    for i in range(1, len(offset_list)):
        offset_list[i] = (offset_list[i]-offset_list[0])/4
    offset_list[0] = 0
    for i in range(len(offset_list)):
        print(offset_list[i], end=', ')
        if (i + 1) % 3==0:
            print('')
            if i > 3 and i < len(offset_list)-1 and offset_list[i+1] - offset_list[i-2] != 32:
                print('not')

    a = 0
    b = 0
    ab = (offset_list[1] - offset_list[0]) / 4
    index = 2
    while index < len(offset_list):
        tmp = (offset_list[index] - offset_list[index - 1])/4
        if tmp != ab:
            tmp_index = index - 1
            while tmp_index >= 0 and offset_list[index] < offset_list[tmp_index]:
                tmp_index -= 1
            tmp = (offset_list[index] - offset_list[tmp_index]) / 4
            a = ab / tmp
            b = tmp
            break
        index += 1
    if a == 0:
        smooth = True
        a = (weights_mem[1] - weights_mem[0]) / (max(offset_list) - min(offset_list) + ab * 4)
        return a, ab, smooth
    smooth = False
    return a, b, smooth


def get_weights_list(value: str, compiler='tvm', size=4):
    global input_on_the_left
    addr_list = []
    if compiler == 'tvm' and size == 4:
        if input_on_the_left:
            it = re.finditer(r'\* (0x[0-9a-f]+),16', value)  # it = re.finditer(r'(0x[0-9a-f]+),4 \*', value)
        else:
            it = re.finditer(r'(0x[0-9a-f]+),16 \*', value)
        for match in it:
            addr = match.group(1)
            addr_list.append(int(addr, 16))
        return addr_list
    if compiler == 'tvm' and size == 16:
        if input_on_the_left:
            it = re.finditer(r'\* (0x[0-9a-f]+),16', value)
        else:
            it = re.finditer(r'(0x[0-9a-f]+),16 \*', value)
        for match in it:
            addr = match.group(1)
            addr_list.append(int(addr, 16))
        return addr_list
    elif compiler == 'glow':
        assert False, 'does Glow need this function?'


# ==============================================================
# Heuristics used to recover shape for Glow Conv2d
# ==============================================================
def explain_glow_conv2d_result(exp_log_path: str, mem_read_regions: list, mem_write_regions: list, in_addr=0, guess_stride=1, guess_padding=0):
    name, exp = choose_one_4bytes(exp_log_path, mem_write_regions)
    mem_list = [(name, exp)]
    element_size = 4  # mem_list.append(tuple(choose_one_4bytes(exp_log_path, mem_write_regions, 1)))
    if len(name) == 0:
        name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=32)
        # return explain_tvm_conv2d_result_16(name, exp, mem_read_regions, mem_write_regions)
        mem_list = [(name, exp)]
        element_size = 32

    with_relu = False
    if 'max(' in exp:
        # print('with relu')
        with_relu = True
    else:
        # print('without relu')
        with_relu = False

    # TODO: here assume width==height
    input_shape = [1, 1, 1, 1]
    filter_shape = [1, 1, 1, 1]
    output_shape = [1, 1, 1, 1]

    if len(mem_read_regions) > 10:
        kernel_num, input_num, blk_size = kernel_1_1(name, exp, mem_read_regions, mem_write_regions, compiler='glow')
        input_num = math.ceil(input_num)
        filter_shape[1] = kernel_num
        input_shape[1] = kernel_num
        input_shape[2] = input_shape[3] = input_num
        output_shape[2] = output_shape[3] = math.ceil(input_num/2)
    else:
        in_mem = biggest_region(mem_read_regions, target_addr=in_addr)
        # get the filter shape and input shape from first output
        offset_list = get_offset_list(mem_list[0][1], compiler='glow', in_blk=in_mem)
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
        # cannot get stride in the case of glow, because glow use NHWC
        # the output shape can be wrong, because of the implicit padding
        output_shape[2] = math.ceil((input_shape[2] + guess_padding*2 - filter_shape[2] + 1) / guess_stride)
        output_shape[3] = math.ceil((input_shape[3] + guess_padding*2 - filter_shape[3] + 1) / guess_stride)

    # get output shape
    output_channel = 0
    one_channel_size = output_shape[2] * output_shape[3]
    weights_region, output_channel = get_output_channel(mem_list[0][1], one_channel_size, mem_write_regions,
                                                        compiler='glow')

    output_shape[1] = output_channel
    filter_shape[0] = output_shape[1]

    # since the stride and padding are guessed, we need to check if the shapes are reasonable
    weights_addrs = get_weights_addrs(mem_list[0][1], size=element_size)
    weights_mem = (0, 0)
    for mem_blk in mem_read_regions:
        if mem_blk[0] <= weights_addrs[0] <= mem_blk[1]:
            weights_mem = mem_blk

    ignore_flag = True
    if int(filter_shape[0]) == filter_shape[0]:
        weights_size = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3] * 4  # float --> 4 bytes
        if weights_size == weights_mem[1] - weights_mem[0]:
            # then it is a possible shape
            ignore_flag = False

    if not ignore_flag:
        # final shape
        print('input shape', input_shape)
        print('filter shape', filter_shape)
        print('output shape', output_shape)
        print('with_relu', with_relu)
        return filter_shape, input_shape, output_shape, with_relu
    else:
        print('not a reasonable guess, ignored')
        return (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), False


def get_weights_addrs(exp: str, size=4):
    addr_list = []
    it = re.finditer(r'\* (0x[0-9a-f]+),'+str(size), exp)
    for match in it:
        addr = match.group(1)
        addr_list.append(int(addr, 16))
    print('one weights addr', hex(addr_list[0]))
    return addr_list


# ==============================================================
# Heuristics used to recover shape for TVM dense/fully-connected layer
# ==============================================================
def explain_tvm_dense_result(exp_log_path: str, mem_write_regions: list):
    name, exp = choose_one_4bytes(exp_log_path, mem_write_regions)
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


# ==============================================================
# Heuristics used to recover shape for TVM add layer
# ==============================================================
def explain_tvm_add_result(exp_log_path: str, mem_read_regions: list, mem_write_regions: list):
    # TODO: cannot assume the order of input and bias
    # name, exp = choose_one_16bytes(exp_log_path)
    # match = re.search(r'\+ 0x([0-9a-f]+),4', exp)
    # bias_addr = int(match.group(1), 16)

    output_size = 0
    small_blk = (0, 0x7ffffff)
    for mem_blk in mem_read_regions:
        if 4 < (mem_blk[1] - mem_blk[0]) < (small_blk[1] - small_blk[0]):
            small_blk = mem_blk
    output_size = (small_blk[1] - small_blk[0]) / 4
    return output_size


# ==============================================================
# Heuristics used to recover shape for TVM max-pool2d layer
# ==============================================================
def explain_tvm_maxpool_result(exp_log_path: str, mem_write_regions: list):
    out_mem = (0, 0)
    for mem_blk in mem_write_regions:
        if (mem_blk[1] - mem_blk[0]) > (out_mem[1] - out_mem[1]):
            out_mem = mem_blk
    with open(exp_log_path, 'r') as f:
        exp_txt = f.read()
        lines = exp_txt.split('\n')
        idx = 0
        while idx < len(lines):
            name1 = lines[idx]
            idx += 1
            exp1 = lines[idx]
            idx += 1
            if out_mem[0] <= int(name1.split(',')[0].strip(), 16) <= out_mem[1] and \
                    int(math.sqrt(exp1.count('max')))**2 == exp1.count('max'):
                break
        name2 = lines[idx]
        idx += 1
        exp2 = lines[idx]
        idx += 1

    if name1.endswith(',4') and name2.endswith(',4'):
        kernel_size = math.sqrt(exp1.count('max'))
        match = re.search(r', 0x([0-9a-f]+),4\)', exp1[exp1.find(')'):])
        if not match:
            match = re.search(r'\(0x([0-9a-f]+),4, ', exp1)
        addr1 = int(match.group(1), 16)
        match = re.search(r', 0x([0-9a-f]+),4\)', exp2[exp2.find(')'):])
        if not match:
            match = re.search(r'\(0x([0-9a-f]+),4, ', exp2)
        addr2 = int(match.group(1), 16)
        stride = (addr2 - addr1) / 4
        return kernel_size, stride
    elif name1.endswith(',16') and name2.endswith(',16'):
        kernel_size = math.sqrt(exp1.count('max'))
        match = re.findall(r', 0x([0-9a-f]+),16\)', exp1)[-1]  # match = re.search(r', 0x([0-9a-f]+),16\)', exp1[exp1.find(')'):])
        #if not match:
        #    match = re.search(r'\(0x([0-9a-f]+),16, ', exp1)
        addr1 = int(match, 16)
        match = re.findall(r', 0x([0-9a-f]+),16\)', exp2)[-1]  # match = re.search(r', 0x([0-9a-f]+),16\)', exp2[exp2.find(')'):])
        #if not match:
        #    match = re.search(r'\(0x([0-9a-f]+),16, ', exp2)
        addr2 = int(match, 16)
        # stride = ((addr2 - addr1) / 16) * kernel_size  # stride = (addr2 - addr1) / 8  # which one?
        stride = (addr2 - addr1) / 8  # stride = ((addr2 - addr1) / 4) / kernel_size
        return kernel_size, stride
    elif name1.endswith(',32') and name2.endswith(',32'):
        kernel_size = math.sqrt(exp1.count('max'))
        match = re.search(r'\(0x([0-9a-f]+),32, ', exp1[:exp1.find(')')])
        addr1 = int(match.group(1), 16)
        match = re.search(r'\(0x([0-9a-f]+),32, ', exp2[:exp2.find(')')])
        addr2 = int(match.group(1), 16)
        stride = (addr2 - addr1) / 16
        return kernel_size, stride


# ==============================================================
# Heuristics used to recover shape for TVM avg_pool2d layer
# ==============================================================
def explain_tvm_avgpool_result(exp_log_path: str, mem_write_regions: list):
    name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=4)
    addr_list = []
    it = re.finditer(r'(0x[0-9a-f]+),4', exp)
    for match in it:
        addr = match.group(1)
        addr_list.append(int(addr, 16))
    min_addr = min(addr_list)
    offset_list = []
    for addr in addr_list:
        addr = (addr - min_addr) / 4
        offset_list.append(addr)
    return math.sqrt(len(offset_list)), 1


# ==============================================================
# Heuristics used to recover shape for GLOW dense/matmul layer
# ==============================================================
def explain_glow_dense_result(exp_log_path: str, mem_write_regions: list):
    name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=32)
    if len(name) == 0:
        name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=4)

    input_size = exp.count('*')
    output_size = 0
    big_mem = (0, 0)
    for mem_blk in mem_write_regions:
        if (mem_blk[1] - mem_blk[0]) > (big_mem[1] - big_mem[0]):
            big_mem = mem_blk
    output_size = (big_mem[1] - big_mem[0]) / 4
    return input_size, output_size


# ==============================================================
# Heuristics used to recover shape for TVM max-pool2d layer
# ==============================================================
def explain_glow_maxpool_result(exp_log_path: str, mem_read_regions: list, mem_write_regions: list):
    out_mem = biggest_region(mem_write_regions)
    in_mem = biggest_region(mem_read_regions)

    name, exp = choose_one_max(exp_log_path, out_mem, )
    """
    with open(exp_log_path, 'r') as f:
        exp_txt = f.read()
        lines = exp_txt.split('\n')
        idx = 0
        while idx < len(lines):
            name1 = lines[idx]
            idx += 1
            exp1 = lines[idx]
            idx += 1
            if out_mem[0] <= int(name1.split(',')[0].strip(), 16) <= out_mem[1] and \
                    int(math.sqrt(exp1.count('max')))**2 == exp1.count('max'):
                break
        name2 = lines[idx]
        idx += 1
        exp2 = lines[idx]
        idx += 1
    """
    if name.endswith(',4'):
        kernel_size = math.sqrt(exp.count('max'))
        match = re.search(r', 0x([0-9a-f]+),4\)', exp[exp.find(')'):])
        if not match:
            match = re.search(r'\(0x([0-9a-f]+),4, ', exp)
        addr1 = int(match.group(1), 16)
        assert in_mem[0] <= addr1 <= in_mem[1]
        size_diff = (in_mem[1]-in_mem[0]) / (out_mem[1]-out_mem[0])
        stride = math.sqrt(size_diff)
        return kernel_size, stride


def explain_glow_avgpool_result(exp_log_path: str, mem_write_regions: list, mem_read_regions: list, vector_size=0):
    name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=4)
    block_size = 4
    if len(name) == 0:
        name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=32)
        block_size = 32
    addr_list = []
    it = re.finditer(r'(0x[0-9a-f]+),'+str(block_size), exp)
    input_mem = biggest_region(mem_read_regions)
    for match in it:
        addr = match.group(1)
        addr = int(addr, 16)
        if input_mem[0] <= addr <= input_mem[1]:
            addr_list.append(addr)
    min_addr = min(addr_list)
    offset_list = []
    for addr in addr_list:
        addr = (addr - min_addr) / 4
        offset_list.append(addr)
    # TODO: is it possible that stride != kernel ?
    return len(offset_list), 1


if __name__ == '__main__':
    pass
    # explain_tvm_conv2d_result('./mem_log.txt')
