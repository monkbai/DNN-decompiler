import os
import re
import math
import utils
import collections


# ==============================================================
# Utils
# ==============================================================
def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def smallest_region(mem_regions: list, target_addr=0):
    small_mem = (0, 0xdeadbeaf)
    target_mem = (0, 0)
    for mem_blk in mem_regions:
        if (mem_blk[1] - mem_blk[0]) < (small_mem[1] - small_mem[0]):
            small_mem = mem_blk
        if mem_blk[0] <= target_addr <= mem_blk[1]:
            target_mem = mem_blk
    if target_addr == 0:
        return small_mem
    else:
        return target_mem


def biggest_region(mem_regions: list, target_addr=0):
    big_mem = (0, 0)
    target_mem = (0, 0)
    for mem_blk in mem_regions:
        if (mem_blk[1] - mem_blk[0]) > (big_mem[1] - big_mem[0]):
            big_mem = mem_blk
        if mem_blk[0] <= target_addr <= mem_blk[1]:
            target_mem = mem_blk
        elif (target_mem[1] - target_mem[0]) != 0 and (target_mem[1] - target_mem[0]) == mem_blk[1] - mem_blk[0]:
            target_mem = mem_blk
    if target_addr == 0:
        return big_mem
    else:
        return target_mem


def region_with_target(mem_regions: list, target_addr: int):
    target_mem = (0, 0)
    for mem_blk in mem_regions:
        if mem_blk[0] <= target_addr <= mem_blk[1]:
            target_mem = mem_blk
    return target_mem


def biggest_last_region(mem_regions: list, target_addr=0):
    big_mem = (0, 0)
    target_mem = (0, 0)
    for mem_blk in mem_regions:
        if (mem_blk[1] - mem_blk[0]) >= (big_mem[1] - big_mem[0]):
            big_mem = mem_blk
        if mem_blk[0] <= target_addr <= mem_blk[1]:
            target_mem = mem_blk
        elif (target_mem[1] - target_mem[0]) != 0 and (target_mem[1] - target_mem[0]) == mem_blk[1] - mem_blk[0]:
            target_mem = mem_blk
    if target_addr == 0:
        return big_mem
    else:
        return target_mem


def get_write_addr_list(exp_log_path: str):
    write_addr_list = []
    with open(exp_log_path, 'r') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines) - 2:
            write_addr = lines[idx]
            idx += 1
            exp = lines[idx]
            idx += 1
            write_addr = int(write_addr.split(',')[0], 16)
            write_addr_list.append(write_addr)
    return write_addr_list


def choose_one_4bytes(exp_log_path: str, mem_write_regions=[], num=0):
    """ Choose one expression from the exp_log to recover the filter shape """
    if len(mem_write_regions) == 0:
        out_mem = (0, 0xffffffff)
    else:
        out_mem = biggest_region(mem_write_regions)
    write_addr_list = get_write_addr_list(exp_log_path)
    use_out_mem_flag = False
    for addr in write_addr_list:
        if out_mem[0] <= addr <= out_mem[1]:
            use_out_mem_flag = True
            break
    if not use_out_mem_flag:
        out_mem = (0, 0xffffffff)
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
            tmp_name = lines[index]
            index += 1
            tmp_exp = lines[index].strip('<>')
            index += 1
            if (not tmp_name.endswith(',4')) or tmp_name.startswith('0x7ff'):
                continue
            else:  # choose the longest expression of one 4 bytes memory block
                if out_mem[0] <= int(tmp_name.split(',')[0], 16) <= out_mem[1]:
                    if tmp_exp.count('*') > exp.count('*'):
                        name = tmp_name
                        exp = tmp_exp
        return name, exp


def choose_one_8bytes(exp_log_path: str, mem_write_regions=[], num=0):
    if len(mem_write_regions) == 0:
        out_mem = (0, 0xffffffff)
    else:
        out_mem = biggest_region(mem_write_regions)
    write_addr_list = get_write_addr_list(exp_log_path)
    use_out_mem_flag = False
    for addr in write_addr_list:
        if out_mem[0] <= addr <= out_mem[1]:
            use_out_mem_flag = True
            break
    if not use_out_mem_flag:
        out_mem = (0, 0xffffffff)
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
            tmp_name = lines[index]
            index += 1
            tmp_exp = lines[index].strip('<>')
            index += 1
            if (not tmp_name.endswith(',8')) or tmp_name.startswith('0x7ff'):
                continue
            else:  # choose the longest expression of one 4 bytes memory block
                if out_mem[0] <= int(tmp_name.split(',')[0], 16) <= out_mem[1]:
                    if tmp_exp.count('*') > exp.count('*'):
                        name = tmp_name
                        exp = tmp_exp
        return name, exp


def choose_one_16bytes(exp_log_path: str, mem_write_regions: list, num=0):
    if len(mem_write_regions) == 0:
        out_mem = (0, 0xffffffff)
    else:
        out_mem = biggest_region(mem_write_regions)
    write_addr_list = get_write_addr_list(exp_log_path)
    use_out_mem_flag = False
    for addr in write_addr_list:
        if out_mem[0] <= addr <= out_mem[1]:
            use_out_mem_flag = True
            break
    if not use_out_mem_flag:
        out_mem = (0, 0xffffffff)
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
            tmp_name = lines[index]
            index += 1
            tmp_exp = lines[index].strip('<>')
            index += 1
            if (not tmp_name.endswith('16')) or tmp_name.startswith('0x7ff'):
                continue
            elif exp_txt.count(tmp_name) > 1:
                continue
            else:  # choose the first expression of one 4 bytes memory block
                if out_mem[0] <= int(tmp_name.split(',')[0], 16) <= out_mem[1]:
                    if tmp_exp.count('*') > exp.count('*'):
                        name = tmp_name
                        exp = tmp_exp
        return name, exp


def choose_one_bytes(exp_log_path: str, mem_write_regions: list, size=4, num=0):
    out_mem = biggest_region(mem_write_regions)
    write_addr_list = get_write_addr_list(exp_log_path)
    use_out_mem_flag = False
    for addr in write_addr_list:
        if out_mem[0] <= addr <= out_mem[1]:
            use_out_mem_flag = True
            break
    if not use_out_mem_flag:
        out_mem = (0, 0xffffffff)
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
    # print('input start addr', hex(input_start_addr))  # for debugging
    for mem_start, mem_end in mem_regions:
        if mem_start <= input_start_addr < mem_end:
            return math.floor(math.sqrt(((mem_end-mem_start)/input_channel)/4))


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
        max_value = None
        if 'min(' in exp:
            max_value_addr = get_max_value_addr(exp)
            max_value = get_max_value(max_value_addr)
            print('max value of clip:', max_value)
        return explain_tvm_conv2d_result_16(name, exp, mem_read_regions, mem_write_regions, guess_stride, optimized)
    mem_list = [(name, exp)]

    # clip merged into conv2d
    max_value = None
    if 'min(' in exp:
        max_value_addr = get_max_value_addr(exp)
        max_value = get_max_value(max_value_addr)
        print('max value of clip:', max_value)

    # TODO: here assume width==height
    input_shape = [1, 1, 1, 1]
    filter_shape = [1, 1, 1, 1]
    output_shape = [1, 1, 1, 1]

    blk_size = 0
    if len(mem_read_regions) > 18:
        kernel_num, input_num, blk_size = kernel_1_1(name, exp, mem_read_regions, mem_write_regions)
        filter_shape[1] = kernel_num
        input_shape[1] = kernel_num
        input_shape[2] = input_shape[3] = input_num
        output_shape[2] = output_shape[3] = math.ceil(input_num/2)
        # print('special case: stride 2')
        weights_mem = (0, 0)  # will be identified in function
        weight_list = []
        special_flag = False
    else:
        # get the filter shape and input shape from first output
        if 'sub(' in exp:
            offset_list, weight_list = get_offset_list(mem_list[0][1], compiler='tvm', size=16, weight_addr=True)
            addr_list = get_addr_list(mem_list[0][1], compiler='tvm', size=16)
        else:
            offset_list, weight_list = get_offset_list(mem_list[0][1], compiler='tvm', weight_addr=True)  # analyze the first expression (with the smallest address)
            addr_list = get_addr_list(mem_list[0][1], compiler='tvm')
        # print('debug input offset_list', offset_list)  # debug
        in_mem = region_with_target(mem_read_regions, addr_list[0])  # in_mem = biggest_region(mem_read_regions)
        out_mem = biggest_region(mem_write_regions)

        # try to get the weight_mem region
        if hex(weight_list[0]).lower().startswith('0x7ff'):
            prev_read_regions = utils.previous_read_mem_regions()
            weights_mem = smallest_region(prev_read_regions)
        else:
            weights_mem = region_with_target(mem_read_regions, weight_list[0])

        special_flag = False
        stride = offset_list[1] - offset_list[0]  # not the real stride
        index = 0
        while index < len(offset_list) - 1:
            if offset_list[index+1] - offset_list[index] > stride:
                tmp1 = offset_list[index + 1]  # input[1] * input[2]
                tmp2 = offset_list[index] + stride  # filter[1] * filter[2]
                filter_shape[3] = math.ceil(len(offset_list)/tmp2)
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

        # add case: filter shape is [1 x 1]
        # if filter_shape[3] > 9 and (filter_shape[1] != int(filter_shape[1]) or filter_shape[1] < 3) and tmp1 >= the_threshold:
        if (not is_integer_num(filter_shape[1]) or filter_shape[3] > 9) and (filter_shape[1] != int(filter_shape[1]) or filter_shape[1] < 3) and index < 4:  # index == 3
            tmp1 = offset_list[index + 1]
            tmp2 = tmp1 / (index+1)   # input[2] * input[3]
            input_shape[2] = input_shape[3] = math.sqrt(tmp2)
            filter_shape[2] = filter_shape[3] = 1
            input_shape[1] = filter_shape[1] = len(offset_list)

            filter_shape[0] = output_shape[1] = (weights_mem[1] - weights_mem[0]) / 4 / filter_shape[1]
            # tmp_value = math.sqrt((out_mem[1] - out_mem[0]) / 4 / filter_shape[0])
            # input_shape[2] = input_shape[3] = output_shape[2] = output_shape[3] = math.ceil(tmp_value)
            # blk_size = 1  # TODO: do we still need this flag for the special case?
            special_flag = True
        # case [3 x 3], [5 x 5], ...
        # elif filter_shape[3] > 9 and (filter_shape[1] != int(filter_shape[1]) or filter_shape[1] < 3) and tmp1 < the_threshold:
        elif (not is_integer_num(filter_shape[1]) or filter_shape[3] > 9) and (filter_shape[1] != int(filter_shape[1]) or filter_shape[1] < 3) and index >= 4:
            tmp1 = offset_list[index + 1]
            tmp2 = tmp1 / 4  # input[2] or input[3]
            input_shape[2] = input_shape[3] = tmp2
            filter_shape[2] = filter_shape[3] = (index+1) / 4
            input_shape[1] = filter_shape[1] = len(offset_list) / filter_shape[2] / filter_shape[3]

            filter_shape[0] = output_shape[1] = (weights_mem[1] - weights_mem[0]) / 4 / (filter_shape[1] * filter_shape[2] * filter_shape[3])
            # tmp_value = math.sqrt((out_mem[1] - out_mem[0]) / 4 / filter_shape[0])
            # input_shape[2] = input_shape[3] = output_shape[2] = output_shape[3] = math.ceil(tmp_value)
            special_flag = True
        # add case: group conv  # exists in shufflenet, efficientnet
        elif (index + 1) ** 2 == len(offset_list):  # or index == len(offset_list) - 1:  # condition not verified
            index = int(math.sqrt(len(offset_list))) - 1
            filter_shape[1] = 1
            filter_shape[2] = filter_shape[3] = index + 1
            filter_shape[0] = output_shape[1] = (weights_mem[1] - weights_mem[0]) / 4 / (filter_shape[2] * filter_shape[3])
            input_shape[1] = filter_shape[0]
            input_shape[2] = input_shape[3] = math.sqrt((in_mem[1] - in_mem[0]) / 4 / input_shape[1])
            special_flag = True

        stride = guess_stride  # if we cannot get accurate stride, guess one

        output_shape[2] = math.ceil((input_shape[2] - filter_shape[2] + 1)/stride)
        output_shape[3] = math.ceil((input_shape[3] - filter_shape[3] + 1)/stride)
    # get output shape
    # TODO: cannot get output_channel easily, because we do not have all mem_log (too huge and too slow)
    # filter_size = filter_shape[1] * filter_shape[2] * filter_shape[3]
    one_channel_size = output_shape[2] * output_shape[3]
    weights_region, output_channel = get_output_channel(mem_list[0][1], one_channel_size, mem_write_regions, compiler='tvm')
    # TODO: fix it <get_output_channel>
    # XXX: Did I made it ?

    output_shape[1] = output_channel
    filter_shape[0] = output_shape[1]

    # since the stride and padding are guessed, we need to check if the shapes are reasonable
    ignore_flag = is_ignore(mem_list, mem_read_regions, filter_shape, weights_mem)

    if not ignore_flag:
        # try to get the weights layout indicators
        ind_a, ind_b, smooth = get_weights_layout_info(mem_list[0][1], mem_read_regions, weights_mem=weights_mem,
                                                       weights_offset_list=weight_list, special_flag=special_flag)
        # print('ind_a {}, ind_b {}, smooth {}'.format(ind_a, ind_b, smooth))
        # final shape, for debugging
        # print('input shape', input_shape)
        # print('filter shape', filter_shape)
        # print('output shape', output_shape)
        # print('layout indicators: {}, {}'.format(ind_a, ind_b))
        if optimized and not special_flag:
            if blk_size:  # kernel --> 1, 1
                ind_a = blk_size
                layout_shape = [filter_shape[0]/ind_b, filter_shape[1]/ind_a, filter_shape[2], filter_shape[3], ind_a, ind_b]
            elif not smooth:
                layout_shape = [filter_shape[0]/ind_b, filter_shape[1]/ind_a, filter_shape[2], filter_shape[3], ind_a, ind_b]
            elif filter_shape[1] > ind_a and (filter_shape[1] % ind_a) == 0:
                layout_shape = [filter_shape[0]/ind_b, ind_a, filter_shape[2], filter_shape[3], filter_shape[1]/ind_a, ind_b]
            elif filter_shape[1] <= ind_a or (filter_shape[1] % ind_a) != 0:
                layout_shape = [filter_shape[0]/ind_b, 1, filter_shape[2], filter_shape[3], filter_shape[1], ind_b]
        elif optimized and special_flag:
            if filter_shape[1] == 1:  # group conv
                layout_shape = [filter_shape[0] / ind_b, 1, filter_shape[2], filter_shape[3], 1, ind_b]
            elif filter_shape[0] > ind_a and (filter_shape[0] % ind_a) == 0:
                layout_shape = [ind_a, filter_shape[1] / ind_b, filter_shape[2], filter_shape[3], filter_shape[0]/ind_a, ind_b]
            elif filter_shape[0] <= ind_a or (filter_shape[0] % ind_a) != 0:
                layout_shape = [1, filter_shape[1] / ind_b, filter_shape[2], filter_shape[3], filter_shape[0], ind_b]
            else:
                assert False, 'currently not implemented.'
        else:
            layout_shape = filter_shape
        # print('layout shape', layout_shape)
        # print('stride {}'.format(guess_stride))
        return filter_shape, input_shape, output_shape, layout_shape
    else:
        #print('input shape', input_shape)
        #print('filter shape', filter_shape)
        #print('output shape', output_shape)
        # print('not a reasonable guess, ignored')
        return filter_shape, input_shape, output_shape, (0, 0, 0, 0, 0, 0)


def get_splited_in_mem(mem_read_regions: list):
    target_size = dict()
    for mem_blk in mem_read_regions:
        mem_size = mem_blk[1] - mem_blk[0]
        if mem_size in target_size:
            target_size[mem_size] += 1
        else:
            target_size[mem_size] = 1

    mem_start = 0x7f0000000000
    mem_end = 0
    target_list = list(target_size.items())
    target_list = sorted(target_list, key=lambda x: x[1])
    mem_size = target_list[-1][0]
    tail_size = 0
    for i in range(len(mem_read_regions)):
        mem_blk = mem_read_regions[i]
        if mem_blk[1] - mem_blk[0] == mem_size:
            if mem_blk[0] < mem_start:
                mem_start = mem_blk[0]
            if mem_blk[1] > mem_end:
                mem_end = mem_blk[1]

            if tail_size == 0 and i < (len(mem_read_regions) - 3):
                blk_2 = mem_read_regions[i+1]
                blk_3 = mem_read_regions[i+2]
                if (blk_2[1]-blk_2[0]) == (blk_3[1]-blk_3[0]) == mem_size:
                    if blk_3[1] - blk_2[1] == blk_2[1] - mem_blk[1]:
                        tail_size = blk_2[1] - mem_blk[1]
    return mem_start, mem_end + tail_size


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
        #print('special case: stride 2')
        guess_stride = 2
    else:
        in_mem = biggest_region(mem_read_regions)
        out_mem = biggest_region(mem_write_regions)

        # offset_list = get_offset_list(mem_list[0][1], compiler='tvm', size=16)
        offset_list, weight_list = get_offset_list(mem_list[0][1], compiler='tvm', size=16,
                                                   in_blk=in_mem, weight_addr=True)
        # print(offset_list)
        weights_mem = region_with_target(mem_read_regions, weight_list[0])

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
                filter_shape[3] = math.ceil(length / tmp2)
                filter_shape[2] = filter_shape[3]  # TODO assume
                print(tmp2, filter_shape[2])
                filter_shape[1] = tmp2 / filter_shape[2]
                filter_shape[1] *= (len(offset_list) / length)
                filter_shape[1] = math.floor(filter_shape[1])
                if filter_shape[1] == 0:
                    break  # another case
                # input[1] = filter[1]
                input_shape[1] = filter_shape[1]
                print(tmp1)
                input_shape[2] = tmp1 / input_shape[1]
                print(len(offset_list), length)
                input_shape[2] *= (len(offset_list) / length)
                input_shape[2] = math.floor(input_shape[2])
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

        if (input_shape[3] > 225 or filter_shape[1] == 0 or filter_shape[1] != int(filter_shape[1]) or filter_shape[1] < 3) and index < 4:  # index == 3
            tmp1 = offset_list[index + 1]
            tmp2 = tmp1 / (index+1)   # input[2] * input[3]
            input_shape[2] = input_shape[3] = math.sqrt(tmp2)
            filter_shape[2] = filter_shape[3] = 1
            input_shape[1] = filter_shape[1] = len(offset_list)

            filter_shape[0] = output_shape[1] = (weights_mem[1] - weights_mem[0]) / 4 / filter_shape[1]
            # tmp_value = math.sqrt((out_mem[1] - out_mem[0]) / 4 / filter_shape[0])
            # input_shape[2] = input_shape[3] = output_shape[2] = output_shape[3] = math.ceil(tmp_value)
            blk_size = 1

        output_shape[2] = math.ceil((input_shape[2] - filter_shape[2] + 1) / guess_stride)
        output_shape[3] = math.ceil((input_shape[3] - filter_shape[3] + 1) / guess_stride)
    # get output shape
    # TODO: cannot get output_channel easily, because we do not have all mem_log (too huge and too slow)
    # filter_size = filter_shape[1] * filter_shape[2] * filter_shape[3]
    one_channel_size = output_shape[2] * output_shape[3]
    weights_region, output_channel = get_output_channel(mem_list[0][1], one_channel_size, mem_write_regions, compiler='tvm', on_the_right=False)
    # TODO: fix it <get_output_channel>
    output_shape[1] = output_channel
    filter_shape[0] = output_shape[1]

    ignore_flag = is_ignore(mem_list, mem_read_regions, filter_shape)

    if not ignore_flag:
        # try to get the weights layout indicators
        ind_a, ind_b, smooth = get_weights_layout_info(mem_list[0][1], mem_read_regions, size=16)
        #print('ind_a {}, ind_b {}, smooth {}'.format(ind_a, ind_b, smooth))
        # final shape
        #print('input shape', input_shape)
        #print('filter shape', filter_shape)
        #print('output shape', output_shape)
        #print('layout indicators: {}, {}'.format(ind_a, ind_b))
        if optimized:
            if blk_size:  # kernel --> 1, 1
                ind_a = blk_size
                layout_shape = [filter_shape[0] / ind_b, filter_shape[1] / ind_a, filter_shape[2], filter_shape[3], ind_a,
                                ind_b]
            elif not smooth:
                layout_shape = [filter_shape[0] / ind_b, filter_shape[1] / ind_a, filter_shape[2], filter_shape[3], ind_a,
                                ind_b]
            elif filter_shape[1] > ind_a:
                layout_shape = [filter_shape[0] / ind_b, ind_a, filter_shape[2], filter_shape[3], filter_shape[1] / ind_a,
                                ind_b]
            elif filter_shape[1] <= ind_a:
                layout_shape = [filter_shape[0] / ind_b, 1, filter_shape[2], filter_shape[3], filter_shape[1], ind_b]
        else:
            layout_shape = filter_shape
        #print('layout shape', layout_shape)
        #print('stride {}'.format(guess_stride))
        return filter_shape, input_shape, output_shape, layout_shape
    else:
        #print('input shape', input_shape)
        #print('filter shape', filter_shape)
        #print('output shape', output_shape)
        #print('not a reasonable guess, ignored')
        return filter_shape, input_shape, output_shape, (0, 0, 0, 0, 0, 0)


def is_ignore(mem_list: list, mem_read_regions: list, filter_shape: list, weights_mem=(0, 0)):
    # since the stride and padding are guessed, we need to check if the shapes are reasonable
    # Check if the size of weights region is as expected

    if weights_mem[0] == 0:
        weights_addrs = get_weights_addrs(mem_list[0][1], size=16)
        if len(weights_addrs) == 0:
            weights_addrs = get_weights_addrs(mem_list[0][1], size=16, on_the_right=False)
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


def get_offset_list(value: str, compiler: str, size=4, in_blk=(0, 0), weight_addr=False):
    times = value.count('*')
    if compiler == 'tvm' and not weight_addr:
        offset_list = get_addr_list(value, 'tvm', size)
        # (offset_list)  #debug
    elif compiler == 'tvm' and weight_addr:
        offset_list, weight_list = get_addr_list(value, 'tvm', size, weight_addr=weight_addr)
        # (offset_list)  #debug
    elif compiler == 'glow' and not weight_addr:
        offset_list = get_addr_list(value, 'glow', size, in_blk=in_blk)
    elif compiler == 'glow' and weight_addr:
        offset_list, weight_list = get_addr_list(value, 'glow', size, in_blk=in_blk, weight_addr=weight_addr)
    else:
        print('at get_offset_list')
        print('compiler not supported:', compiler)
        assert False, "Not implemented."
        return
    assert len(offset_list) != 0, ("the symbolic expression is not corectly genreated.\n"
                                   "It is verly likely due to the randome target address is not correctly picked.\n"
                                   "You may want to delete the corresponding *_slice.log file and try again.\n"
                                   "Hopefully, a new target address picked by trace_filter will solve this.\n"
                                  )  # Glow
    start_addr = min(offset_list)
    for i in range(len(offset_list)):
        offset_list[i] = (offset_list[i] - start_addr) / 4
    if size == 4 and offset_list[0] > offset_list[-1]:
        offset_list.reverse()  # offset_list.sort()
    elif size == 16 and offset_list[0] > offset_list[-1]:
        offset_list.reverse()
    if weight_addr:
        return offset_list, weight_list
    else:
        return offset_list


input_on_the_left = True


def get_addr_list(value: str, compiler: str, size=4, in_blk=(0, 0), weight_addr=False):
    global input_on_the_left
    """

    :param value: the expression
    :param compiler: 'tvm' or 'glow'
    :return: list of used input addresses
    """
    addr_list = []
    weight_addr_list = []
    if compiler == 'tvm' and size == 4:
        match = re.search(r'(0x[0-9a-f]+),4 \*', value)
        if match:
            it = re.finditer(r'(0x[0-9a-f]+),[0-9]+ \*', value)
            weight_it = re.finditer(r' \* (0x[0-9a-f]+),[0-9]+', value)
            input_on_the_left = True
        else:
            it = re.finditer(r'\* (0x[0-9a-f]+),[0-9]+', value)
            weight_it = re.finditer(r'(0x[0-9a-f]+),[0-9]+ \*', value)
            input_on_the_left = False
        for match in it:
            addr = match.group(1)
            addr_list.append(int(addr, 16))
        for match in weight_it:
            addr = match.group(1)
            weight_addr_list.append(int(addr, 16))
        if weight_addr:
            return addr_list, weight_addr_list
        else:
            return addr_list
    if compiler == 'tvm' and size == 16:
        match = re.search(r'\* (0x[0-9a-f]+),4', value)
        match2 = re.search(r'(0x[0-9a-f]+),4 \*', value)
        match3 = re.search(r'\* (0x[0-9a-f]+),16', value)
        match4 = re.search(r'(0x[0-9a-f]+),16 \*', value)
        if match:
            it = re.finditer(r'\* (0x[0-9a-f]+),4', value)
            weight_it = re.finditer(r'(0x[0-9a-f]+),[0-9]+ \*', value)
            input_on_the_left = False
        elif match2:
            it = re.finditer(r'(0x[0-9a-f]+),[0-9]+ \*', value)
            weight_it = re.finditer(r'\* (0x[0-9a-f]+),[0-9]+', value)
            input_on_the_left = True
        elif match4:
            it = re.finditer(r'(0x[0-9a-f]+),16 \*', value)
            weight_it = re.finditer(r'\* (0x[0-9a-f]+),[0-9]+', value)
            input_on_the_left = True
        elif match3:
            it = re.finditer(r'\* (0x[0-9a-f]+),16', value)
            weight_it = re.finditer(r'(0x[0-9a-f]+),[0-9]+ \*', value)
            input_on_the_left = False

        for match in it:
            addr = match.group(1)
            addr_list.append(int(addr, 16))
        for match in weight_it:
            addr = match.group(1)
            weight_addr_list.append(int(addr, 16))
        if weight_addr:
            return addr_list, weight_addr_list
        else:
            return addr_list
    elif compiler == 'glow':
        reg_str_list = [r'(0x[0-9a-f]+,{} \*)'.format(size), r'(\* 0x[0-9a-f]+,{})'.format(size), r'(\* 0x[0-9a-f]+,32)', r'(0x[0-9a-f]+,32 \*)']  # ,32 is used for extracting weights addr
        # reg_str = r'((0x[0-9a-f]+,{} \*)|(\* 0x[0-9a-f]+,{}))'.format(size, size)
        for reg_str in reg_str_list:
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
                elif weight_addr:
                    weight_addr_list.append(addr)
        addr_list.sort()
        if weight_addr:
            return addr_list, weight_addr_list
        else:
            return addr_list


def get_weights_layout_info(value: str, mem_read_regions: list, compiler='tvm', size=4, weights_mem=(0, 0),
                            weights_offset_list=[], special_flag=False):
    if weights_mem[0] == 0:
        weights_addrs = get_weights_addrs(value, size=16)
        if len(weights_addrs) == 0:
            weights_addrs = get_weights_addrs(value, size=16, on_the_right=False)
        weights_mem = (0, 0)
        for mem_blk in mem_read_regions:
            if mem_blk[0] <= weights_addrs[0] <= mem_blk[1]:
                weights_mem = mem_blk

    if len(weights_offset_list) == 0:
        # print('weights_addrs', weights_addrs)  # debug
        offset_list = get_weights_list(value, compiler=compiler, size=size)
    else:
        offset_list = weights_offset_list

    if offset_list[1] < offset_list[0]:
        offset_list.reverse()

    for i in range(1, len(offset_list)):
        offset_list[i] = (offset_list[i]-offset_list[0])/4
    offset_list[0] = 0
    # print('offset_list', offset_list)  # debug
    '''
    # debug
    for i in range(len(offset_list)):
        print(offset_list[i], end=', ')
        if (i + 1) % 3==0:
            print('')
            if i > 3 and i < len(offset_list)-1 and offset_list[i+1] - offset_list[i-2] != 32:
                print('not')
    '''
    a = 0
    b = 0
    ab = (offset_list[1] - offset_list[0])  
    index = 2
    while index < len(offset_list):
        tmp = (offset_list[index] - offset_list[index - 1])
        if tmp != ab:
            tmp_index = index - 1
            while tmp_index >= 0 and offset_list[index] < offset_list[tmp_index]:
                tmp_index -= 1
            tmp = (offset_list[index] - offset_list[tmp_index]) 
            a = ab / tmp
            b = tmp
            break
        index += 1
    if a == 0:
        smooth = True
        a = (weights_mem[1] - weights_mem[0]) / (max(offset_list) - min(offset_list) + ab)
        a2 = ((weights_mem[1] - weights_mem[0]) / 4 ) / (max(offset_list) - min(offset_list) + ab)
        if special_flag:
            return a2, ab, smooth
        else:
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

    with_max_value = False
    max_value = None
    if 'min(' in exp:
        with_max_value = True
        max_value_addr = get_max_value_addr(exp)
        max_value = get_max_value(max_value_addr)


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
        out_mem = biggest_region(mem_write_regions)
        # get the filter shape and input shape from first output
        offset_list, weight_list = get_offset_list(mem_list[0][1], compiler='glow', in_blk=in_mem, weight_addr=True)
        weights_mem = region_with_target(mem_read_regions, weight_list[0])
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

        # add case: filter shape is [1 x 1]  # inception
        if index == len(offset_list) - 1:
            filter_shape[3] = filter_shape[2] = 1
            input_shape[1] = filter_shape[1] = len(offset_list)
            filter_shape[0] = output_shape[1] = (weights_mem[1] - weights_mem[0]) / 4 / filter_shape[1]
            tmp_value = math.sqrt((out_mem[1] - out_mem[0]) / 4 / filter_shape[0])
            input_shape[2] = input_shape[3] = output_shape[2] = output_shape[3] = math.ceil(tmp_value)
        # add case: group conv  # exists in shufflenet
        elif (index+1)**2 == len(offset_list):
            filter_shape[1] = 1
            filter_shape[2] = filter_shape[3] = index+1
            filter_shape[0] = output_shape[1] = (weights_mem[1] - weights_mem[0]) / 4 / (filter_shape[2]*filter_shape[3])
            input_shape[1] = filter_shape[0]
            input_shape[2] = input_shape[3] = math.sqrt((in_mem[1]-in_mem[0])/4/input_shape[1])
        # cannot get stride in the case of glow, because glow use NHWC
        # the output shape can be wrong, because of the implicit padding
        output_shape[2] = math.ceil((input_shape[2] + guess_padding*2 - filter_shape[2] + 1) / guess_stride)
        output_shape[3] = math.ceil((input_shape[3] + guess_padding*2 - filter_shape[3] + 1) / guess_stride)

    # get output shape
    output_channel = 0
    one_channel_size = output_shape[2] * output_shape[3]
    # weights_region, output_channel = get_output_channel(mem_list[0][1], one_channel_size, mem_write_regions, compiler='glow')
    out_mem = biggest_region(mem_write_regions)
    output_channel = ((out_mem[1] - out_mem[0]) / one_channel_size) / 4

    filter_shape[0] = output_shape[1] = output_channel

    # since the stride and padding are guessed, we need to check if the shapes are reasonable
    # Maybe we should not use function <get_weights_addrs> anymore  # weights_addrs = get_weights_addrs(mem_list[0][1], size=element_size)
    _, weights_addrs = get_offset_list(mem_list[0][1], compiler='glow', in_blk=in_mem, weight_addr=True)
    weights_mem = (0, 0)
    for mem_blk in mem_read_regions:
        if mem_blk[0] <= weights_addrs[0] <= mem_blk[1]:
            weights_mem = mem_blk
            break

    ignore_flag = True
    if int(filter_shape[0]) == filter_shape[0]:
        weights_size = filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3] * 4  # float --> 4 bytes
        if weights_size == weights_mem[1] - weights_mem[0]:
            # then it is a possible shape
            ignore_flag = False
    else:  # add case: inception glow 2022 0091, but why this happens?
        output_channel = ((out_mem[1] - out_mem[0]) / (one_channel_size + 1)) / 4  # why there is an additional line in output? caused by optimization?
        filter_shape[0] = output_shape[1] = output_channel
        filter_number = (weights_mem[1] - weights_mem[0]) / (filter_shape[1] * filter_shape[2] * filter_shape[3] * 4)
        if int(filter_number) == filter_number and output_channel == filter_number:
            ignore_flag = False

    if not ignore_flag:
        # final shape
        print('input shape', input_shape)
        print('filter shape', filter_shape)
        print('output shape', output_shape)
        print('with_relu', with_relu)
        if max_value:
            print('with max value: {}'.format(max_value))
        print('stride {}, padding {}'.format(guess_stride, guess_padding))
        return filter_shape, input_shape, output_shape, with_relu, max_value
    else:
        # print('not a reasonable guess, ignored')
        return (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), False, None


def get_weights_addrs(exp: str, size=4, on_the_right=True):
    addr_list = []
    if on_the_right:
        it = re.finditer(r'\* (0x[0-9a-f]+),'+str(size), exp)
    else:
        it = re.finditer(r'(0x[0-9a-f]+),'+str(size)+' \*', exp)
    for match in it:
        addr = match.group(1)
        addr_list.append(int(addr, 16))
    # print('one weights addr', hex(addr_list[0]))
    return addr_list


def get_max_value_addr(exp: str):
    mat = re.search('min\(.*, 0x([0-9a-f]+),[0-9]+\)', exp)
    if mat:
        return int(mat.group(1), 16)


def get_max_value(dump_addr: int):
    dump_addr = hex(dump_addr)
    log_path = 'tmp_dump.log'
    max_value = utils.extract_single_dword(log_path, dump_addr)
    return max_value

# ==============================================================
# Heuristics used to recover shape for TVM dense/fully-connected layer
# ==============================================================
def explain_tvm_dense_result(exp_log_path: str, mem_read_regions: list, mem_write_regions: list, func_info=[]):
    name, exp = choose_one_4bytes(exp_log_path, mem_write_regions)
    if len(name) == 0:
        name, exp = choose_one_8bytes(exp_log_path, mem_write_regions)
    if len(name) == 0:
        name, exp = choose_one_16bytes(exp_log_path, mem_write_regions)
    if len(name) == 0:
        assert False, ('explain: explain_tvm_dense_result(): failed to choose expression')
        exit(-1)
    if len(func_info) > 0:
        input_start = func_info[3][0]
        input_mem = (0, 0)
        for mem_blk in mem_read_regions:
            if mem_blk[1] > int(input_start, 16) >= mem_blk[0]:
                input_mem = mem_blk
                break
        input_size = int((input_mem[1] - input_mem[0]) / 4)
    else:
        input_size = exp.count('*')
        if input_size == exp.count('16 *'):  # why use this heuristic?
            input_size *= 4
        elif input_size == exp.count('32 *'):
            input_size *= 8

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
                    int(math.sqrt(exp1.count('max')))**2 == exp1.count('max') and \
                        name1.split(',')[1] == lines[idx].split(',')[1]:  # ,4 and ,16 may appear at the same time
                break
        name2 = lines[idx]
        idx += 1
        exp2 = lines[idx]
        idx += 1

    if name1.endswith(',4') and name2.endswith(',4'):
        kernel_size = math.sqrt(exp1.count('max'))
        match = re.search(r', 0x([0-9a-f]+),4\)', exp1[exp1.find(')'):])
        match_list = re.findall(r', 0x([0-9a-f]+),4\)', exp1[exp1.find(')'):])
        if not match:
            match = re.search(r'\(0x([0-9a-f]+),4, ', exp1)
            match_list = re.findall(r'\(0x([0-9a-f]+),4, ', exp1)
        addr1 = int(match.group(1), 16)
        addr1_list = [x.replace(', ', '').strip('()') for x in match_list]

        match = re.search(r', 0x([0-9a-f]+),4\)', exp2[exp2.find(')'):])
        match_list = re.findall(r', 0x([0-9a-f]+),4\)', exp2[exp2.find(')'):])
        if not match:
            match = re.search(r'\(0x([0-9a-f]+),4, ', exp2)
            match_list = re.findall(r'\(0x([0-9a-f]+),4, ', exp2)
        addr2 = int(match.group(1), 16)
        addr2_list = [x.replace(', ', '').strip('()') for x in match_list]
        stride = (addr2 - addr1) / 4
        index = 1
        while stride == 0:
            stride = (int(addr2_list[index], 16) - int(addr1_list[index], 16)) / 4
            index += 1
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
def explain_tvm_avgpool_result(exp_log_path: str, mem_read_regions: list, mem_write_regions: list, is2d=False):
    item_size = 4
    name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=item_size)
    if len(name) == 0:
        item_size = 16
        name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=item_size)
    if len(name) == 0:
        item_size = 32
        name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=item_size)

    read_mem = biggest_region(mem_read_regions)
    addr_list = []
    it = re.finditer(r'(0x[0-9a-f]+),'+str(item_size), exp)
    for match in it:
        addr = match.group(1)
        addr = int(addr, 16)
        if read_mem[0] <= addr <= read_mem[1]:
            addr_list.append(addr)
    min_addr = min(addr_list)
    offset_list = []
    for addr in addr_list:
        addr = (addr - min_addr) / 4
        offset_list.append(addr)

    stride_size = offset_list[1] - offset_list[0]
    kernel_size = (len(offset_list), 1)
    '''
    for h in range(1, len(offset_list)):
        if offset_list[h]-offset_list[h-1] != stride_size:
            kernel_size = (h, len(offset_list)/h)  # TODO: check resnet
            break
    '''
    dimension_flag = 1
    for i in range(len(offset_list) - 1):
        if offset_list[i+1] - offset_list[i] != stride_size:
            dimension_flag = 2
    if dimension_flag == 2 or is2d:
        kernel_size = (math.sqrt(kernel_size[0]), kernel_size[1])
    else:
        pass
    return kernel_size, 1


# ==============================================================
# Heuristics used to recover embedding (TVM take layer)
# ==============================================================
def explain_tvm_embedding_result(exp_log_path: str, mem_read_regions: list, mem_write_regions: list):
    one_vec = biggest_region(mem_read_regions)
    vec_size = (one_vec[1] - one_vec[0]) / 4
    return vec_size


def explain_tvm_lrn_result(exp_log_path: str, mem_read_regions: list, mem_write_regions: list):
    """
    For local response normalization.
    Get thet amount of neighbouring channels used for normalization.
    :param exp_log_path:
    :param mem_write_regions:
    :param mem_read_regions:
    :return:
    """
    name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=4, num=-1)  # num=-1 --> find the first one, no need in the out_mem
    block_size = 4
    if len(name) == 0:
        name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=16, num=-1)
        block_size = 16
    if len(name) == 0:
        name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=32, num=-1)
        block_size = 32
    it = re.finditer(r'(0x[0-9a-f]+,[0-9]+) \* (0x[0-9a-f]+,[0-9]+)', exp)
    addr_set = set()
    for match in it:
        addr1 = match.group(1)
        addr2 = match.group(2)
        if addr1 == addr2:
            addr_set.add(addr1)
    size = len(addr_set)
    return size


def explain_tvm_clip_result(exp_log_path: str, mem_read_regions: list, mem_write_regions: list):
    name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=4,num=-1)
    block_size = 4
    if len(name) == 0:
        name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=16, num=-1)
        block_size = 16
    if len(name) == 0:
        name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=32, num=-1)
        block_size = 32
    with_max_value = False
    max_value = None
    if 'min(' in exp:
        with_max_value = True
        max_value_addr = get_max_value_addr(exp)
        max_value = get_max_value(max_value_addr)
    return max_value


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
    if len(mem_read_regions) > 10:
        in_mem = get_splited_in_mem(mem_read_regions)
    else:
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
    name, exp = choose_one_max(exp_log_path, out_mem, size=32)
    if name.endswith(',32'):
        kernel_size = math.sqrt(exp.count('max'))
        match = re.search(r', 0x([0-9a-f]+),32\)', exp[exp.find(')'):])
        if not match:
            match = re.search(r'\(0x([0-9a-f]+),32, ', exp)
        addr1 = int(match.group(1), 16)
        assert in_mem[0] <= addr1 <= in_mem[1]
        size_diff = (in_mem[1]-in_mem[0]) / (out_mem[1]-out_mem[0])
        stride = math.sqrt(size_diff)
        return kernel_size, stride


def explain_glow_avgpool_result(exp_log_path: str, mem_write_regions: list, mem_read_regions: list, is2d=False):
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
    dimension_flag = 1
    offset_step = offset_list[1] - offset_list[0]
    for i in range(len(offset_list) - 1):
        if offset_list[i+1] - offset_list[i] != offset_step:
            dimension_flag = 2
    if dimension_flag == 2 or is2d:
        return math.sqrt(len(offset_list)), 1
    else:
        return len(offset_list), 1


def explain_glow_insert_tensor(exp_log_path: str, mem_write_regions: list, mem_read_regions: list, func_info: list):
    offset = 0
    output_addr = func_info[3][0]
    output_addr = int(output_addr, 16)
    min_addr = int('0xffffffff', 16)
    with open(exp_log_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            addr = lines[i].split(',')[0]
            addr = int(addr, 16)
            min_addr = min(min_addr, addr)
    offset = (min_addr - output_addr) / 4
    return offset


def explain_glow_extract_tensor(exp_log_path: str, mem_write_regions: list, mem_read_regions: list, func_info: list):
    offset = 0
    input_addr = func_info[3][0]
    input_addr = int(input_addr, 16)
    output_addr = func_info[4]
    output_addr = int(output_addr, 16)

    min_addr = int('0xffffffff', 16)
    with open(exp_log_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            addr = lines[i].split(',')[0]
            addr = int(addr, 16)
            min_addr = min(min_addr, addr)
    # it is strange that min_addr is lower than output_addr
    # I suppose it caused by confusing design in Glow
    offset = (min_addr - output_addr) / 4

    in_mem = biggest_region(mem_read_regions)
    out_mem = biggest_region(mem_write_regions)
    return in_mem, out_mem


def explain_glow_transpose(exp_log_path: str, mem_write_regions: list, mem_read_regions: list, func_info: list):
    in_mem = biggest_region(mem_read_regions)
    out_mem = biggest_region(mem_write_regions)
    return in_mem, out_mem


def explain_glow_lrn(exp_log_path: str, mem_write_regions: list, mem_read_regions: list):
    """
    For local response normalization.
    Get thet amount of neighbouring channels used for normalization.
    :param exp_log_path:
    :param mem_write_regions:
    :param mem_read_regions:
    :return:
    """
    name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=4)
    block_size = 4
    if len(name) == 0:
        name, exp = choose_one_bytes(exp_log_path, mem_write_regions, size=32)
        block_size = 32
    size = exp.count('*')
    return size


if __name__ == '__main__':
    value = "max(((0x1f8dafc,4 * 0x4343bc,4 + (0x1f8daf8,4 * 0x4343b8,4 + (0x1f8daf4,4 * 0x4343b4,4 + (0x1f8daf0,4 * 0x4343b0,4 + (0x1f8daec,4 * 0x4343ac,4 + (0x1f8dae8,4 * 0x4343a8,4 + (0x1f8dae4,4 * 0x4343a4,4 + (0x1f8dae0,4 * 0x4343a0,4 + (0x1f8dadc,4 * 0x43439c,4 + (0x1f8dad8,4 * 0x434398,4 + (0x1f8dad4,4 * 0x434394,4 + (0x1f8dad0,4 * 0x434390,4 + (0x1f8dacc,4 * 0x43438c,4 + (0x1f8dac8,4 * 0x434388,4 + (0x1f8dac4,4 * 0x434384,4 + (0x1f8dac0,4 * 0x434380,4 + (0x1f8dabc,4 * 0x43437c,4 + (0x1f8dab8,4 * 0x434378,4 + (0x1f8dab4,4 * 0x434374,4 + (0x1f8dab0,4 * 0x434370,4 + (0x1f8daac,4 * 0x43436c,4 + (0x1f8daa8,4 * 0x434368,4 + (0x1f8daa4,4 * 0x434364,4 + (0x1f8daa0,4 * 0x434360,4 + (0x1f8da9c,4 * 0x43435c,4 + (0x1f8da98,4 * 0x434358,4 + (0x1f8da94,4 * 0x434354,4 + (0x1f8da90,4 * 0x434350,4 + (0x1f8da8c,4 * 0x43434c,4 + (0x1f8da88,4 * 0x434348,4 + (0x1f8da84,4 * 0x434344,4 + (0x1f8da80,4 * 0x434340,4 + (0x1f8da7c,4 * 0x43433c,4 + (0x1f8da78,4 * 0x434338,4 + (0x1f8da74,4 * 0x434334,4 + (0x1f8da70,4 * 0x434330,4 + (0x1f8da6c,4 * 0x43432c,4 + (0x1f8da68,4 * 0x434328,4 + (0x1f8da64,4 * 0x434324,4 + (0x1f8da60,4 * 0x434320,4 + (0x1f8da5c,4 * 0x43431c,4 + (0x1f8da58,4 * 0x434318,4 + (0x1f8da54,4 * 0x434314,4 + (0x1f8da50,4 * 0x434310,4 + (0x1f8da4c,4 * 0x43430c,4 + (0x1f8da48,4 * 0x434308,4 + (0x1f8da44,4 * 0x434304,4 + (0x1f8da40,4 * 0x434300,4 + (0x1f8da3c,4 * 0x4342fc,4 + (0x1f8da38,4 * 0x4342f8,4 + (0x1f8da34,4 * 0x4342f4,4 + (0x1f8da30,4 * 0x4342f0,4 + (0x1f8da2c,4 * 0x4342ec,4 + (0x1f8da28,4 * 0x4342e8,4 + (0x1f8da24,4 * 0x4342e4,4 + (0x1f8da20,4 * 0x4342e0,4 + (0x1f8da1c,4 * 0x4342dc,4 + (0x1f8da18,4 * 0x4342d8,4 + (0x1f8da14,4 * 0x4342d4,4 + (0x1f8da10,4 * 0x4342d0,4 + (0x1f8da0c,4 * 0x4342cc,4 + (0x1f8da08,4 * 0x4342c8,4 + (0x1f8da04,4 * 0x4342c4,4 + (0x1f8da00,4 * 0x4342c0,4 + 0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))) + 0x204ab00,4), 0)"
    # print(reg_str)  # debug
    it = re.finditer(r'((0x[0-9a-f]+,4 \*)|(\* 0x[0-9a-f]+,4))', value)
    for match in it:
        addr = match.group(1).strip()
        if addr.startswith('0x'):
            addr = addr.split(',')[0]
        else:
            addr = addr[addr.find('0x'):]
            addr = addr.split(',')[0]
        addr = int(addr, 16)

    exit(0)
    # explain_tvm_conv2d_result('./mem_log.txt')
    import pin_tools
    import mem_slices
    prog_path = '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.9.dev/resnet18_tvm_O3/resnet18_tvm_O3_strip'
    data_path = '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/cat.bin'
    start_addr = '0x40b450'
    end_addr = '0x40d5d7'
    mem_read_log_path = './mem_read.log'
    mem_write_log_path = './mem_write.log'
    pin_tools.mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, data_path)
    read_mem_regions = mem_slices.memory_slices(mem_read_log_path)
    pin_tools.mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
    write_mem_regions = mem_slices.memory_slices(mem_write_log_path)
    explain_tvm_conv2d_result('./mem_exp.log', read_mem_regions, write_mem_regions, guess_stride=1, optimized=False)
