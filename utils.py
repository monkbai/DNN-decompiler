#! /usr/bin/python3
import copy
import os
import random
import sys
import json
import numpy as np
import warnings

import pin_tools
from pin_tools import func_call_trace, inst_trace_log, mem_read_log, mem_write_log
from pin_tools import dump_dwords, dump_dwords_2, dump_dwords_3
from pin_tools import convert_dwords2float, rm_log, fun_call_rdi_rsi, compile_all_tools, fused_rdi
from se_engine import lightweight_SymEx
from mem_slices import memory_slices
from explain import explain_tvm_conv2d_result, explain_tvm_dense_result
from explain import explain_tvm_add_result, explain_tvm_maxpool_result, explain_tvm_avgpool_result
from explain import explain_glow_conv2d_result, explain_glow_dense_result, explain_glow_maxpool_result
from explain import explain_glow_avgpool_result, explain_tvm_embedding_result
import explain


def list_to_json(dict_obj: dict, output_path: str):
    j = json.dumps(dict_obj, sort_keys=True, indent=4)
    with open(output_path, 'w') as f:
        f.write(j)


def dict_to_json(dict_obj: dict, output_path: str):
    j = json.dumps(dict_obj, sort_keys=True, indent=4)
    with open(output_path, 'w') as f:
        f.write(j)


def json_to_list(json_path: str):
    if not os.path.exists(json_path):
        return dict()
    with open(json_path, 'r') as f:
        j_txt = f.read()
        list_obj = json.loads(s=j_txt)
        return list_obj


def json_to_dict(json_path: str):
    if not os.path.exists(json_path):
        return dict()
    with open(json_path, 'r') as f:
        j_txt = f.read()
        dict_obj = json.loads(s=j_txt)
        return dict_obj


addr2label = dict()
addr2funcs = dict()
addr2param = dict()
func2param = dict()

funcs_dir = './vgg16_glow_funcs/'


# ==============================================================
# Generate the function call trace
# ==============================================================


def rm_duplicated_call(log_path: str):
    """ Seems caused by bugs inside TVM"""
    with open(log_path, 'r') as f:
        old_trace = f.read()
    new_trace = ''
    old_lines = old_trace.split('\n')
    for i in range(len(old_lines)):
        if i > 0 and old_lines[i - 1] == old_lines[i]:
            continue
        new_trace += old_lines[i] + '\n'

    with open(log_path, 'w') as f:
        f.write(new_trace)


def get_addr_list(label_path: str, fused=False):
    addr_list = []
    with open(label_path, 'r') as f:
        label_txt = f.read()
        lines = label_txt.split('\n')
        for line in lines:
            if ':' not in line:
                continue
            name, label = line.split(':')
            if len(label.strip()) > 0:
                start_addr, end_addr = get_func_range(os.path.join(funcs_dir, name.strip()))
                addr = start_addr.lower()

                addr2label[addr] = label.strip()
                addr2funcs[addr] = name.strip()
                if fused and ('fused' not in label and 'entry' not in label):
                    continue
                elif not fused and ('fused' in label or 'entry' in label):
                    continue

                addr_list.append(addr)
    return addr_list


def get_funcs_trace(prog_path: str, in_data: str, log_path: str, label_file: str, compiler='glow', only_fused=False):
    prog_path = os.path.abspath(prog_path)
    if len(in_data) > 0:
        in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    label_file = os.path.abspath(label_file)

    if compiler == 'tvm':
        addr_list = get_addr_list(label_file, fused=only_fused)
    else:
        addr_list = get_addr_list(label_file)

    if compiler == 'tvm' and only_fused:
        fused_rdi(prog_path, in_data, addr_list, log_path)
        # rm_duplicated_call(log_path)  # to track the index of add/multiply, have to keep duplicated calls
    elif compiler == 'glow':
        fun_call_rdi_rsi(prog_path, in_data, addr_list, log_path)
    elif compiler == 'tvm' and not only_fused:
        func_call_trace(prog_path, in_data, addr_list, log_path)
    else:
        assert False, 'unknown compiler: {}'.format(compiler)

    dict_to_json(addr2label, './addr2label.json')
    dict_to_json(addr2funcs, './addr2funcs.json')


def print_layer_label_tvm(trace_log_path: str, config_path='', only_fused=False):
    global addr2label, addr2funcs, addr2param
    addr2label = json_to_dict('./addr2label.json')  # type: dict
    addr2funcs = json_to_dict('./addr2funcs.json')  # type: dict
    func2param = dict()
    if len(config_path) > 0:
        config_path = os.path.abspath(config_path)
        func2param = json_to_dict(config_path)  # type: dict

    trace_log_path = os.path.abspath(trace_log_path)
    param_list = []
    if not only_fused:
        with open(trace_log_path, 'r') as f:
            trace_txt = f.read()
            lines = trace_txt.split('\n')
            for line in lines:
                if not line.startswith('0x'):
                    continue
                addr = hex(int(line.split(':')[0].strip(), 16))
                print('{}: {}'.format(addr, addr2label[addr]))
                # if 'reshape' != addr2label[addr]:
                #    print('{}: {}'.format(addr, addr2label[addr]))
    else:
        with open(trace_log_path, 'r') as f:
            trace_txt = f.read()
            lines = trace_txt.split('\n')
            for line in lines:
                if not line.startswith('0x'):
                    continue
                addr = hex(int(line.split(':')[0].strip(), 16))
                addr_list = list(addr2label.keys())  # type: list
                addr_list.sort()
                idx = addr_list.index(addr) + 1
                label = addr2label[addr_list[idx]]

                # print the func type
                print('{}: {:<18}:'.format(addr, label), end=' ')
                params = line.split(':')[1].strip()
                params = params.split(',')[:-1]
                param_labels = []
                for key, labels_list in func2param.items():
                    if key in label:
                        param_labels = labels_list
                        break
                # print the func parameters 
                with_label = True
                if len(params) != len(param_labels):
                    with_label = False

                for i in range(len(params)):
                    if i != len(params) - 1:
                        if with_label:
                            print('{} {},'.format(param_labels[i], params[i]), end=' ')
                        else:
                            print('{},'.format(params[i]), end=' ')
                    else:
                        if with_label:
                            print('{} {}'.format(param_labels[i], params[i]))
                        else:
                            print('{}'.format(params[i]))

                for i in range(len(params)):
                    params[i] = int(params[i].strip(), 16)
                param_list += params
    param_list.sort()
    return param_list


def print_input_id(trace_log_path: str, compiler='tvm', addr2param=dict()):
    global addr2label, addr2funcs  # , addr2param
    addr2label = json_to_dict('./addr2label.json')  # type: dict
    addr2funcs = json_to_dict('./addr2funcs.json')  # type: dict

    trace_log_path = os.path.abspath(trace_log_path)

    params_list = []
    id = 0
    id2addr = dict()
    if compiler == 'tvm':
        with open(trace_log_path, 'r') as f:
            trace_txt = f.read()
            lines = trace_txt.split('\n')
            for line in lines:
                if not line.startswith('0x'):
                    continue
                addr = hex(int(line.split(':')[0].strip(), 16))
                params = line.split(':')[1].strip(' ,')
                params = params.split(',')
                inputs = params[:-1]
                output = params[-1]

                addr_list = list(addr2label.keys())  # type: list
                addr_list.sort()
                idx = addr_list.index(addr) + 1
                label = addr2label[addr_list[idx]]  # type: str

                id2addr[id] = (addr_list[idx], addr)

                if 'add add' not in label:
                    params_list.append((id, label, inputs, output))
                    id += 1
                # if 'add add' not in label:  # TODO: why it looks like this?
                #     params_list.append((id, label, inputs, output))
                #     id += 1
                #     if 'relu' in label:
                #         params_list.append((id, 'relu', [output], output))
                #         id += 1
                else:
                    conv2d_label = label.replace('add ', '', 1)
                    params_list.append((id, conv2d_label, inputs[1:], output))
                    id += 1
                    params_list.append((id, 'add', [inputs[0], output], output))
                    id += 1
                    if 'relu' in label:
                        params_list.append((id, 'relu', [output], output))
                        id += 1
    elif compiler == 'glow':
        assert len(addr2param) > 0, 'addr2param not provided.'
        with open(trace_log_path, 'r') as f:
            trace_txt = f.read()
            lines = trace_txt.split('\n')
            for line in lines:
                if not line.startswith('0x'):
                    continue
                addr = hex(int(line.split(':')[0].strip(), 16))
                func_name = addr2funcs[addr]
                params = line.split(':')[1].strip(' ,')
                params = params.split(',')  # not used, inputs and output is produced in utils.print_layer_label()
                inputs = addr2param[id][2][0]
                assert len(addr2param[id][2][1]) == 1, 'len(output_list) should be 1.'
                output = addr2param[id][2][1][0]
                label = addr2label[addr]  # type: str

                params_list.append((id, label, inputs, output))
                id += 1
    else:
        assert False, "compiler {} is currently not supported.".format(compiler)

    # generate input_id_list
    input_id_list = []
    for i in range(len(params_list) - 1, -1, -1):
        input_addrs = params_list[i][2]
        input_id = []
        for input_addr in input_addrs:
            j = i - 1
            while j >= 0:
                params = params_list[j]
                if params[3] == input_addr:
                    input_id.append(j)
                    break
                j -= 1
        input_id_list.insert(0, input_id)

    print(input_id_list)
    output_dict = dict()
    topology_list = []
    func_count = dict()
    for param in params_list:
        if compiler == 'tvm':
            func_addr, entry_addr = id2addr[param[0]]
        else:
            func_addr = entry_addr = addr2param[param[0]][0]  # for glow, there will be only one address
        print('addr:', func_addr, end=' ')
        print('func:', addr2funcs[func_addr], end=' ')
        print(param, end=' ')
        print(input_id_list[param[0]])
        func_name = addr2funcs[func_addr]
        # (name, shape, fused_func, type, padding, stride, param_index)
        output_dict[param[0]] = [func_name, [], entry_addr, param[1], None, None, None]
        if addr2funcs[func_addr] not in func_count.keys():
            func_count[addr2funcs[func_addr]] = 0
        else:
            func_count[addr2funcs[func_addr]] += 1
        topology_list.append([param[0],  # id
                              addr2funcs[func_addr],  # func name
                              param[1],  # label
                              param[2],  # input addresses
                              param[3],  # output address
                              input_id_list[param[0]],  # input ids
                              func_count[addr2funcs[func_addr]]  # func_count/the number of occurrences
                              ])
    return output_dict, topology_list


def refine_glow_config(f2p: dict()) -> dict:
    new_dict = copy.deepcopy(f2p)  # type: dict
    for key, value in f2p.items():
        if len(value) < 4:
            for i in range(len(value), 4):
                value.append('none')
            new_dict[key] = value
    return new_dict


def print_layer_label(trace_log_path: str, config_path=''):  # for glow
    global addr2label, addr2funcs, addr2param, func2param
    addr2label = json_to_dict('./addr2label.json')
    addr2funcs = json_to_dict('./addr2funcs.json')
    # addr2param = json_to_dict('./addr2param.json')  # should be empty right now
    if len(config_path) > 0:
        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            warnings.warn('Glow config file does not exist.')
        else:
            func2param = json_to_dict(config_path)  # type: dict
            func2param = refine_glow_config(func2param)
    else:
        warnings.warn('No Glow config file provided.')

    trace_log_path = os.path.abspath(trace_log_path)
    node_id = 0
    with open(trace_log_path, 'r') as f:
        trace_txt = f.read()
        lines = trace_txt.split('\n')
        for line in lines:
            if not line.startswith('0x'):
                continue
            line = line.strip(',')
            addr = hex(int(line.split(':')[0].strip(), 16))
            addr_list = line.split(':')[1].split(',')  # type: list
            for i in range(len(addr_list)):
                if 'rdi' in addr_list[i]:
                    addr_list[i] = addr_list[i].replace('rdi ', '').strip()
                elif 'rsi' in addr_list[i]:
                    addr_list[i] = addr_list[i].replace('rsi ', '').strip()
                elif 'rdx' in addr_list[i]:
                    addr_list[i] = addr_list[i].replace('rdx ', '').strip()
                elif 'rcx' in addr_list[i]:
                    addr_list[i] = addr_list[i].replace('rcx ', '').strip()
            config_flag = False
            for key, param_list in func2param.items():
                if key in addr2label[addr]:
                    print('{}: {:>10} - {:<16}: {} {}, {} {}, {} {}, {} {}'.format(addr, addr2funcs[addr],
                                                                                   addr2label[addr],
                                                                                   param_list[0], addr_list[0],
                                                                                   param_list[1], addr_list[1],
                                                                                   param_list[2], addr_list[2],
                                                                                   param_list[3], addr_list[3]))
                    input_list = []
                    output_list = []
                    for i in range(len(param_list)):
                        p = param_list[i]
                        if 'in' in p:
                            input_list.append(addr_list[i])
                        if 'out' in p:
                            output_list.append(addr_list[i])
                    addr2param[node_id] = [addr, addr2funcs[addr], (input_list, output_list)]
                    config_flag = True  # found definition in the config.json
                    break
            if not config_flag:  # not defined in config.json
                # print('{}: {:<16}: param1 {}, param2 {}, param3 {}'.format(addr, addr2label[addr],
                #                                                           addr_list[0], addr_list[1], addr_list[2]))
                print('{}: {:>10} - {:<16}:'.format(addr, addr2funcs[addr], addr2label[addr]), end=' ')
                for i in range(len(addr_list)):
                    if i == len(addr_list) - 1:
                        end_str = '\n'
                    else:
                        end_str = ', '
                    print('param{} {}'.format(i + 1, addr_list[i]), end=end_str)
                addr2param[node_id] = [addr, addr2funcs[addr], (addr_list[0], addr_list[1])]  # TODO: not accurate
            node_id += 1
            """
            if 'reshape' != addr2label[addr]:
                if 'max-pool' in addr2label[addr] or 'avg-pool' in addr2label[addr]:
                    in_addr, out_addr = line.split(':')[1].split(',')
                    in_addr = in_addr.strip().replace('in ', 'out')
                    out_addr = out_addr.strip().replace('out', 'in ')
                    print('{}: {:<16}: {}, {}'.format(addr, addr2label[addr], out_addr, in_addr))
                    if addr not in addr2param.keys():
                        out_addr = out_addr.replace('in', '').strip()
                        in_addr = in_addr.replace('out', '').strip()
                        addr2param[addr] = (out_addr, in_addr)
                elif 'add relu' in addr2label[addr]:
                    in_addr, out_addr = line.split(':')[1].split(',')
                    in_addr = in_addr.replace('in', '').strip()
                    out_addr = out_addr.replace('out', '').strip()
                    print('{}: {:<16}: in  {},{}, out {}'.format(addr, addr2label[addr], in_addr, out_addr, out_addr))
                    if addr not in addr2param.keys():
                        addr2param[addr] = (out_addr, in_addr)
                else:
                    in_addr, out_addr = line.split(':')[1].split(',')
                    in_addr = in_addr.replace('in', '').strip()
                    out_addr = out_addr.replace('out', '').strip()
                    print('{}: {:<16}:{}'.format(addr, addr2label[addr], line.split(':')[1]))
                    if addr not in addr2param.keys():
                        addr2param[addr] = (in_addr, out_addr)
            """
    dict_to_json(addr2param, './addr2param.json')
    return copy.deepcopy(addr2param)


# ==============================================================
# Do lightweight Symbolic Execution
# ==============================================================

def get_func_range(func_asm_path: str):
    start_addr = ''
    end_addr = ''
    with open(func_asm_path, 'r') as f:
        asm_txt = f.read()
        lines = asm_txt.split('\n')
        for line in lines:
            if line.startswith(';'):
                continue
            start_addr = line.split(':')[0]
            break
        lines.reverse()
        for line in lines:
            if line.startswith(';') or len(line) < 1:
                continue
            end_addr = line.split(':')[0]
            break
    return start_addr.lower(), end_addr.lower()


def generate_inst_trace(func_name: str, log_path: str, prog_path, data_path: str, timeout=False):
    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)
    start_addr, end_addr = get_func_range(func_asm_path)

    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)

    inst_trace_log(log_path, start_addr, end_addr, prog_path, data_path, timeout)


def generate_symbolic_expression(func_name: str, inst_log_path: str, exp_log_path: str, max_inst=10000000):
    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)

    inst_log_path = os.path.abspath(inst_log_path)
    exp_log_path = os.path.abspath(exp_log_path)

    lightweight_SymEx(func_asm_path, inst_log_path, exp_log_path, max_inst_num=max_inst)  # TODO: max_inst_num


# ==============================================================
# Recover shapes using heuristics
# ==============================================================


def identify_fixed_insert_tensor(asm_path: str):
    """
        two patterns of insert tensor
        the third parameter, rdx, is used or not
        used: not fixed, return False
        not used: fixed, return True
    """
    with open(asm_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            if 'rdx' in l:
                input = l.split(',')[1]
                if 'rdx' not in input:
                    return True
                else:
                    return False
            elif 'edx' in l:
                if 'xor' in l:
                    return True
                input = l.split(',')[1]
                if 'edx' not in input:
                    return True
                else:
                    return False


def recover_shape(func_name: str, mem_exp_log: str,
                  mem_read_log_path: str, mem_write_log_path: str,
                  prog_path: str, data_path: str, func_type='conv2d', func_info=[], is2d=False):
    global addr2param
    addr2param = json_to_dict('./addr2param.json')

    mem_read_log_path = os.path.abspath(mem_read_log_path)
    mem_write_log_path = os.path.abspath(mem_write_log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)

    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)
    start_addr, end_addr = get_func_range(func_asm_path)

    # in_addr = addr2param[start_addr][0]
    param_list = [addr2param[k] if addr2param[k][0] == start_addr else None for k in addr2param.keys()]
    param_list = list(filter(lambda a: a != None, param_list))
    in_addr = param_list[0][2][0]  # use th first one, as we only log the first occurence
    assert 'conv' not in func_type or len(in_addr) == 1, 'the size of inputs of conv operator should be 1.'
    in_addr = int(in_addr[0], 16)
    # out_addr= addr2param[start_addr][1]
    # out_addr = int(out_addr, 16)

    mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, data_path)
    mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
    read_mem_regions = memory_slices(mem_read_log_path)
    write_mem_regions = memory_slices(mem_write_log_path)
    if 'conv' in func_type:
        # try with different stride
        filter_shape = (0, 0, 0, 0)
        input_shape = (0, 0, 0, 0)
        output_shape = (0, 0, 0, 0)
        with_relu = False
        for stride in range(1, 4):
            for padding in range(0, 4):
                # print('try with stride: {}, padding: {}'.format(stride, padding))
                if '0091' in func_name:
                    print('debug')
                tmp_filter_shape, tmp_input_shape, tmp_output_shape, tmp_with_relu = explain_glow_conv2d_result(
                    mem_exp_log,
                    read_mem_regions,
                    write_mem_regions,
                    in_addr=in_addr,
                    guess_stride=stride,
                    guess_padding=padding)
                if tmp_filter_shape[0] != 0:
                    filter_shape = tmp_filter_shape
                    input_shape = tmp_input_shape
                    output_shape = tmp_output_shape
                    with_relu = tmp_with_relu
                    if filter_shape[2] == filter_shape[3] == 1:
                        # print('stride: 2')  # not always
                        return filter_shape, input_shape, output_shape, with_relu  # no need to guess padding/stride
        print(filter_shape)
        if filter_shape[0] == 0:
            assert False, ("failed to predict the filter shape. \n"
                           "Sometimes this will happen, mainly because implicit padding in Glow binary.\n"
                           "In such case, you may delete the slice log file corresponding to current function, and try again.\n"
                           "the trace_filter will randomly pick an address again."
                           )
        return filter_shape, input_shape, output_shape, with_relu
    elif 'matmul' in func_type:  # dense in tvm
        input_size, output_size = explain_glow_dense_result(mem_exp_log, write_mem_regions)
        return output_size, input_size
    elif 'add' in func_type:
        if len(read_mem_regions) == 1:
            # split the read region
            input_addrs = [ int(x, 16) for x in func_info[3]]
            assert len(input_addrs) == 2, "what if the number of inputs for 'add' is {}".format(len(input_addrs))
            large_addr = max(input_addrs)
            addr1 = read_mem_regions[0][0]
            addr2 = read_mem_regions[0][1]
            assert addr1 < large_addr < addr2, 'what if the small_addr is not inside the read_mem_region'
            read_mem_regions = [[addr1, large_addr], [large_addr, addr2]]
        if read_mem_regions[0][1] - read_mem_regions[0][0] == read_mem_regions[1][1] - read_mem_regions[1][0]:
            # the case of add layer after dense/fully-connected layer
            return (read_mem_regions[0][1] - read_mem_regions[0][0]) / 4
        bias_length = explain_tvm_add_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return bias_length
    elif 'max_pool' in func_type:  # max_pool
        kernel_size, stride = explain_glow_maxpool_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return int(kernel_size), int(stride)
    elif 'avg_pool' in func_type:  # avg_pool
        kernel_size, stride = explain_glow_avgpool_result(mem_exp_log, write_mem_regions, read_mem_regions, is2d)
        return kernel_size, stride
    elif 'insert_tensor_param' in func_type:
        # each time insert_tensor_param is called with different offset
        # maybe we can get the offsets during parameter extraction?
        return None
    elif 'insert_tensor' in func_type:
        offset = explain.explain_glow_insert_tensor(mem_exp_log, write_mem_regions, read_mem_regions, func_info)
        return offset
    elif 'local_response_normalization' in func_type:
        size = explain.explain_glow_lrn(mem_exp_log, write_mem_regions, read_mem_regions)
        return size


def recover_shape_tvm(func_name: str, mem_exp_log: str,
                      mem_read_log_path: str, mem_write_log_path: str,
                      prog_path: str, data_path: str, func_type='conv2d', optimized=False, func_info=[]):
    mem_read_log_path = os.path.abspath(mem_read_log_path)
    mem_write_log_path = os.path.abspath(mem_write_log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)

    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)
    start_addr, end_addr = get_func_range(func_asm_path)

    mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, data_path)
    read_mem_regions = memory_slices(mem_read_log_path)
    mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
    write_mem_regions = memory_slices(mem_write_log_path)
    if 'conv2d' in func_type:
        filter_shape = (0, 0, 0, 0)
        input_shape = (0, 0, 0, 0)
        output_shape = (0, 0, 0, 0)
        layout_shape = (0, 0, 0, 0)
        for stride in range(1, 4):
            # print('try with stride: {}'.format(stride))
            tmp_filter_shape, tmp_input_shape, tmp_output_shape, tmp_layout_shape = \
                explain_tvm_conv2d_result(mem_exp_log, read_mem_regions,
                                          write_mem_regions, guess_stride=stride,
                                          optimized=optimized)
            if filter_shape[0] == 0 or tmp_layout_shape[0] != 0:
                filter_shape = tmp_filter_shape
                input_shape = tmp_input_shape
                output_shape = tmp_output_shape
                layout_shape = tmp_layout_shape
                if layout_shape[0] != 0:
                    break
        return filter_shape, input_shape, output_shape, layout_shape
    elif 'dense' in func_type or 'matmul' in func_type:
        input_size, output_size = explain_tvm_dense_result(mem_exp_log, read_mem_regions, write_mem_regions, func_info)
        # print('({}, {})'.format(input_size, output_size))
        return output_size, input_size
    elif 'add' in func_type:
        if len(read_mem_regions) > 1 and \
                read_mem_regions[0][1] - read_mem_regions[0][0] == read_mem_regions[1][1] - read_mem_regions[1][0]:
            # the case of add layer after dense/fully-connected layer
            return (read_mem_regions[0][1] - read_mem_regions[0][0]) / 4
        bias_length = explain_tvm_add_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return bias_length
    elif 'max' in func_type:
        kernel_size, stride = explain_tvm_maxpool_result(mem_exp_log, write_mem_regions)
        return kernel_size, stride
    elif 'avg' in func_type:
        kernel_size, stride = explain_tvm_avgpool_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return kernel_size, stride
    elif 'embedding' in func_type:
        vector_size = explain_tvm_embedding_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return vector_size
    elif 'lrn' in func_type:
        size = explain.explain_tvm_lrn_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return size


# ==============================================================
# Handle all conv2d functions
# ==============================================================


def handle_all_conv(prog_path: str, in_data: str, label_file_path: str,
                    func_trace_map=dict(), compiler='tvm', optimized=False, topo_list=[]):
    label_file_path = os.path.abspath(label_file_path)
    # --- get conv2d functions' name
    funcs_name_list = []
    func_types = dict()
    with open(label_file_path, 'r') as f:
        labels = f.read()
        lines = labels.split('\n')
        for line in lines:
            if ':' not in line:
                continue
            name, label = line.split(':')
            if len(label.strip()) > 0 and ('conv' in label or 'dense' in label or 'matmul' in label):  # and ('0163' in name or '0153' in name):
                name = name.strip()
                funcs_name_list.append(name)
                func_types[name] = label.strip()

    func_shape = dict()
    for func_name in funcs_name_list:
        print('\n' + func_name)
        func_info = []
        if len(topo_list) > 0:
            for node in topo_list:
                if node[1] == func_name:
                    func_info = node
                    break
        # --- recover the shape of each layer
        # tmp_log_path = './inst_trace.log'
        if func_name in func_trace_map:
            tmp_log_path = func_trace_map[func_name]
        else:
            tmp_log_path = './inst_trace.log'
            generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)
        exp_log_path = './mem_exp.log'  # tmp file
        generate_symbolic_expression(func_name, tmp_log_path, exp_log_path)

        # --- try to interpret the filter shape from symbolic expression log
        mem_read_log_path = 'mem_read.log'  # tmp file
        mem_write_log_path = 'mem_write.log'  # tmp file
        if compiler == 'tvm':
            all_shapes = recover_shape_tvm(func_name, exp_log_path,
                                           mem_read_log_path, mem_write_log_path,
                                           prog_path, in_data, func_type=func_types[func_name],
                                           optimized=optimized, func_info=func_info)
        else:
            all_shapes = recover_shape(func_name, exp_log_path,
                                       mem_read_log_path, mem_write_log_path,
                                       prog_path, in_data, func_type=func_types[func_name])
        func_shape[func_name] = all_shapes  # filter_shape, input_shape, output_shape, layout_shape
        print(all_shapes)  # for debug
    return func_shape


# ==============================================================
# Extract parameters
# ==============================================================


def extract_params_tvm(prog_path: str, in_data: str, w_shape: tuple, dump_point: str,
                       log_path: str, func_name: str, func_type='conv2d', data_idx=1, special_layout=()):
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    dwords_len = 1
    for w_dim in w_shape:
        dwords_len *= w_dim
    if os.path.exists(log_path):
        rm_log(log_path)
    dump_dwords_2(prog_path, in_data, dump_point, dwords_len, log_path, data_index=data_idx)

    # then convert dwords to floats
    with open(log_path, 'r') as f:
        dw_txt = f.read()
        f.close()
        end_count = dw_txt.count('end')
        dw_segs = dw_txt.split('end')[:end_count]
        for i in range(end_count):
            dw_txt = dw_segs[i].strip()
            dw_txt = dw_txt[dw_txt.find('\n') + 1:]
            float_array = convert_dwords2float(dw_txt, dwords_len)

            w = np.asarray(float_array)
            if len(special_layout) == 6:
                w = w.reshape(special_layout)
                w = np.transpose(w, (0, 5, 1, 4, 2, 3))
                w = w.reshape(w_shape)
            elif len(special_layout) == 3:
                w = w.reshape(special_layout)
                w1 = np.transpose(w, (0, 2, 1))
                w = w1.reshape(w_shape)
            else:
                w = w.reshape(w_shape)
            # print(type(w))
            lists = w.tolist()
            json_str = json.dumps(lists)
            # print(json_str)
            json_str = json_str.replace('],', '],\n')
            if 'conv2d' in func_type:
                json_name = func_name[:func_name.rfind('.')] + '.weights_{}.json'.format(i)
            elif 'dense' in func_type:
                json_name = func_name[:func_name.rfind('.')] + '.dense_weights_{}.json'.format(i)
            elif 'add' in func_type:
                json_name = func_name[:func_name.rfind('.')] + '.biases_{}.json'.format(i)
            else:
                json_name = func_name[:func_name.rfind('.')] + '.{}_{}.json'.format(func_type, i)
            with open(json_name, 'w') as wf:
                wf.write(json_str)
                wf.close()
    rm_log(log_path)


def extract_inserttensor_offset_glow(prog_path: str, in_data: str, log_path: str, topo_list: list):
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)

    # get all start addresses of insert_tensor operators
    insert_tensor_list = []
    addr_set = set()
    for node in topo_list:
        if 'insert_tensor_param' in node[2]:
            func_name = node[1]
            func_addr, _ = get_func_range(os.path.join(funcs_dir, func_name))
            insert_tensor_list.append([func_name, func_addr, None])  # name, address, offset
            addr_set.add(func_addr)

    # Log
    pin_tools.fun_call_rdx(prog_path, in_data, list(addr_set), log_path)

    # Read the log file
    with open(log_path, 'r') as f:
        lines = f.readlines()
        idx = 0
        for l in lines:
            if l.startswith('#eof'):
                break
            func_addr, offset = l.split(':')
            offset = int(offset.split(',')[-1])
            assert insert_tensor_list[idx][1] == func_addr and not insert_tensor_list[idx][2]
            insert_tensor_list[idx][2] = offset
            idx += 1
    return insert_tensor_list




def extract_params_glow(prog_path: str, in_data: str, w_shape: tuple, dump_point: str,
                        log_path: str, func_name: str, reg_num=1, func_type=''):
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    dwords_len = 1
    for w in w_shape:
        dwords_len *= w
    if os.path.exists(log_path):
        rm_log(log_path)
    dump_dwords(prog_path, in_data, dump_point, dwords_len, log_path, reg_num=reg_num)  # rdx

    # then convert dwords to floats
    with open(log_path, 'r') as f:
        dw_txt = f.read()
        f.close()
        end_count = dw_txt.count('end')
        dw_segs = dw_txt.split('end')[:end_count]
        for i in range(end_count):
            dw_txt = dw_segs[i].strip()
            dw_txt = dw_txt[dw_txt.find('\n') + 1:]
            float_array = convert_dwords2float(dw_txt, dwords_len)

            w = np.asarray(float_array)
            w = w.reshape(w_shape)
            if 'DKKC8' in func_type:  # Glow applies special layout alteration named DKKC8
                # w = np.swapaxes(w, 0, 3)
                w = np.transpose(w, (3, 1, 2, 0, 4))
                new_shape = (w_shape[3], w_shape[1], w_shape[2], w_shape[0] * w_shape[4])
                w = np.reshape(w, new_shape, order='C')
                w = np.transpose(w, (3, 0, 1, 2))
            if reg_num == 1 and len(w_shape) == 4 and func_type != 'matmul':
                w = np.transpose(w, (0, 3, 1, 2))  # store weights in (N,H,W,C) format
            elif reg_num == 1 and len(w_shape) == 4 and func_type == 'matmul':
                w = np.transpose(w, (3, 2, 0, 1))
                print('new shape', (w_shape[3], w_shape[0] * w_shape[1] * w_shape[2]))
                w = w.reshape(w_shape[3], w_shape[0] * w_shape[1] * w_shape[2])
            elif reg_num == 1 and len(w_shape) == 2 and w_shape[0] != 1:
                w = np.transpose(w, (1, 0))
            # print(type(w))
            lists = w.tolist()
            json_str = json.dumps(lists)
            # print(json_str)
            json_str = json_str.replace('],', '],\n')
            if len(w_shape) == 5 and 'DKKC8' in func_type:
                json_name = func_name[:func_name.rfind('.')] + '.weights_{}.json'.format(i)
            elif reg_num == 1 and len(w_shape) == 4:
                json_name = func_name[:func_name.rfind('.')] + '.weights_{}.json'.format(i)
            elif reg_num == 1 and len(w_shape) == 2:
                json_name = func_name[:func_name.rfind('.')] + '.params_{}.json'.format(i)
            elif reg_num == 2:
                json_name = func_name[:func_name.rfind('.')] + '.biases_{}.json'.format(i)

            with open(json_name, 'w') as wf:
                wf.write(json_str)
                wf.close()
    rm_log(log_path)


def extract_params_nnfusion(prog_path: str, in_data: str, w_shape: tuple, dump_point: str,
                            log_path: str, func_name: str, reg_num=1, func_type=''):
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    dwords_len = 1
    for w in w_shape:
        dwords_len *= w
    if os.path.exists(log_path):
        rm_log(log_path)
    dump_dwords(prog_path, in_data, dump_point, dwords_len, log_path, reg_num=reg_num)  # rdx

    # then convert dwords to floats
    with open(log_path, 'r') as f:
        dw_txt = f.read()
        f.close()
        end_count = dw_txt.count('end')
        dw_segs = dw_txt.split('end')[:end_count]
        for i in range(end_count):
            dw_txt = dw_segs[i].strip()
            dw_txt = dw_txt[dw_txt.find('\n') + 1:]
            float_array = convert_dwords2float(dw_txt, dwords_len)

            w = np.asarray(float_array)
            w = w.reshape(w_shape)
            lists = w.tolist()
            json_str = json.dumps(lists)
            # print(json_str)
            json_str = json_str.replace('],', '],\n')
            if reg_num == 1 and len(w_shape) == 4:
                json_name = func_name[:func_name.rfind('.')] + '.weights_{}.json'.format(i)
            elif reg_num == 1 and len(w_shape) == 2:
                json_name = func_name[:func_name.rfind('.')] + '.params_{}.json'.format(i)
            elif reg_num == 2:
                json_name = func_name[:func_name.rfind('.')] + '.biases_{}.json'.format(i)
            else:
                json_name = func_name[:func_name.rfind('.')] + '.param{}_{}.json'.format(reg_num, i)

            with open(json_name, 'w') as wf:
                wf.write(json_str)
                wf.close()
    rm_log(log_path)


def extract_params_general(prog_path: str, in_data: str, w_shape: tuple, dump_point: str,
                           log_path: str, func_name: str, dump_addr: str):
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    dwords_len = 1
    for w in w_shape:
        dwords_len *= w
    if os.path.exists(log_path):
        rm_log(log_path)
    dump_dwords_3(prog_path, in_data, dump_point, dwords_len, log_path, dump_addr=dump_addr)

    # then convert dwords to floats
    with open(log_path, 'r') as f:
        dw_txt = f.read()
        f.close()
        end_count = dw_txt.count('end')
        dw_segs = dw_txt.split('end')[:end_count]
        for i in range(end_count):
            dw_txt = dw_segs[i].strip()
            dw_txt = dw_txt[dw_txt.find('\n') + 1:]
            float_array = convert_dwords2float(dw_txt, dwords_len)

            w = np.asarray(float_array)
            w = w.reshape(w_shape)
            lists = w.tolist()
            json_str = json.dumps(lists)
            # print(json_str)
            json_str = json_str.replace('],', '],\n')
            json_name = func_name[:func_name.rfind('.')] + '.params_{}.json'.format(i)

            with open(json_name, 'w') as wf:
                wf.write(json_str)
                wf.close()
    rm_log(log_path)


if __name__ == '__main__':
    prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/vgg16_tvm_O3_strip"
    in_data = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/cat.bin"
    log_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/func_call.log"
    label_file = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/step1.txt"

    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = './mem_read.log'
    mem_write_log_path = './mem_write.log'
    mem_dump_log_path = 'mem_dump.log'
    w_shape = [64, 3, 3, 3]
    dump_point = "0x426f90"
    func_name = "0070.txt"
    func_type = 'conv2d'
    data_index = 1
    layout_shape = [2, 1, 3, 3, 3, 32]
    extract_params_tvm(prog_path, in_data, w_shape, dump_point, mem_dump_log_path, func_name,
                                 func_type=func_type, data_idx=data_index, special_layout=layout_shape)

    exit(0)
    def tmp_handle_func_call(func_call_trace: str):
        id_list = []
        idx = 0
        with open(func_call_trace) as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('0x'):
                    continue
                in_addrs = line.split(':')[2].strip()
                in_addrs = in_addrs[in_addrs.find('in ') + 3:]
                in_addrs = in_addrs[:in_addrs.find('out')]
                in_addrs = in_addrs.strip(' ,')
                in_addrs = in_addrs.split(',')
                out_addr = line[line.find('out') + 4:].strip()

                id_list.append((idx, in_addrs, out_addr))
                idx += 1

                if 'conv2d' in line:
                    id_list.append((idx, [out_addr], out_addr))
                    idx += 1
        new_id_list = []
        idx = len(id_list) - 1
        while idx >= 0:
            in_addrs = id_list[idx][1]
            tmp_list = []
            for in_addr in in_addrs:
                prev_idx = idx - 1
                while prev_idx >= 0:
                    if id_list[prev_idx][2] == in_addr:
                        tmp_list.append(prev_idx)
                        break
                    prev_idx -= 1
            new_id_list.insert(0, tmp_list)
            idx -= 1
        print(new_id_list)


    # tmp_handle_func_call('/home/lifter/Documents/tvm_output/func_call.txt')

    funcs_dir = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.8/resnet18_tvm_O0/resnet18_funcs/"
    generate_symbolic_expression('0207.txt', './resnet18_tvm_v08_O0/0207.log', './resnet18_tvm_v08_O0/mem_exp.log',
                                 max_inst=5000000)
