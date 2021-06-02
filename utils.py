#! /usr/bin/python3
import os
import random
import sys
import json
import numpy as np
from scripts.pin_tools import func_call_trace, inst_trace_log, mem_read_log, mem_write_log, dump_dwords, dump_dwords_2
from scripts.pin_tools import convert_dwords2float, rm_log, fun_call_rdi_rsi, compile_all_tools
from scripts.se_engine import lightweight_SymEx
from scripts.mem_slices import memory_slices
from scripts.explain import explain_tvm_conv2d_result, explain_tvm_dense_result
from scripts.explain import explain_tvm_add_result, explain_tvm_maxpool_result
from scripts.explain import explain_glow_conv2d_result, explain_glow_dense_result, explain_glow_maxpool_result


def dict_to_json(dict_obj: dict, output_path: str):
    j = json.dumps(dict_obj)
    with open(output_path, 'w') as f:
        f.write(j)


def json_to_dict(json_path: str):
    with open(json_path, 'r') as f:
        j_txt = f.read()
        dict_obj = json.loads(s=j_txt)
        return dict_obj


addr2label = dict()
addr2funcs = dict()
addr2param = dict()

funcs_dir = './vgg16_glow_funcs/'
# ==============================================================
# Generate the function call trace
# ==============================================================


def get_addr_list(label_path: str):
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
                addr = start_addr

                addr_list.append(addr)

                addr2label[addr] = label.strip()
                addr2funcs[addr] = name.strip()
    return addr_list


def get_funcs_trace(prog_path: str, in_data: str, log_path: str, label_file: str):

    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    label_file = os.path.abspath(label_file)

    addr_list = get_addr_list(label_file)

    # func_call_trace(prog_path, in_data, addr_list, log_path)
    fun_call_rdi_rsi(prog_path, in_data, addr_list, log_path)

    dict_to_json(addr2label, './addr2label.json')
    dict_to_json(addr2funcs, './addr2funcs.json')


def print_layer_label(trace_log_path: str):
    global addr2label, addr2funcs, addr2param
    addr2label = json_to_dict('./addr2label.json')
    addr2funcs = json_to_dict('./addr2funcs.json')

    trace_log_path = os.path.abspath(trace_log_path)
    with open(trace_log_path, 'r') as f:
        trace_txt = f.read()
        lines = trace_txt.split('\n')
        for line in lines:
            if not line.startswith('0x'):
                continue
            addr = hex(int(line.split(':')[0].strip(), 16))
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
    dict_to_json(addr2param, './addr2param.json')


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
    return start_addr, end_addr


def generate_inst_trace(func_name: str, log_path: str, prog_path, data_path: str):
    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)
    start_addr, end_addr = get_func_range(func_asm_path)

    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)

    inst_trace_log(log_path, start_addr, end_addr, prog_path, data_path)


def generate_symbolic_expression(func_name: str, inst_log_path: str, exp_log_path: str, max_inst=5000000):
    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)

    inst_log_path = os.path.abspath(inst_log_path)
    exp_log_path = os.path.abspath(exp_log_path)

    lightweight_SymEx(func_asm_path, inst_log_path, exp_log_path, max_inst_num=max_inst)  # TODO: max_inst_num


# ==============================================================
# Recover shapes using heuristics
# ==============================================================


def recover_shape(func_name: str, mem_exp_log: str,
                  mem_read_log_path: str, mem_write_log_path: str,
                  prog_path: str, data_path: str, func_type='conv2d'):
    global addr2param
    addr2param = json_to_dict('./addr2param.json')

    mem_read_log_path = os.path.abspath(mem_read_log_path)
    mem_write_log_path = os.path.abspath(mem_write_log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)

    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)
    start_addr, end_addr = get_func_range(func_asm_path)
    in_addr = addr2param[start_addr][0]
    out_addr= addr2param[start_addr][1]
    in_addr = int(in_addr, 16)
    out_addr = int(out_addr, 16)

    if func_type == 'conv2d':
        mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, data_path)
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        read_mem_regions = memory_slices(mem_read_log_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        # try with different stride
        filter_shape = (0, 0, 0, 0)
        input_shape = (0, 0, 0, 0)
        output_shape = (0, 0, 0, 0)
        with_relu = False
        for stride in range(1, 4):
            for padding in range(0, 4):
                print('try with stride: {}, padding: {}'.format(stride, padding))
                tmp_filter_shape, tmp_input_shape, tmp_output_shape, tmp_with_relu = explain_glow_conv2d_result(mem_exp_log,
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
                        print('stride: 2')
                        return filter_shape  # no need to guess padding/stride
        return filter_shape
    elif func_type == 'dense':
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        input_size, output_size = explain_glow_dense_result(mem_exp_log, write_mem_regions)
        return output_size, input_size
    elif func_type == 'add':
        mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, data_path)
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        read_mem_regions = memory_slices(mem_read_log_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        if read_mem_regions[0][1] - read_mem_regions[0][0] == read_mem_regions[1][1] - read_mem_regions[1][0]:
            # the case of add layer after dense/fully-connected layer
            return (read_mem_regions[0][1] - read_mem_regions[0][0]) / 4
        bias_length = explain_tvm_add_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return bias_length
    elif func_type.startswith('max'):
        mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, data_path)
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        read_mem_regions = memory_slices(mem_read_log_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        kernel_size, stride = explain_glow_maxpool_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return kernel_size, stride


# ==============================================================
# Handle all conv2d functions
# ==============================================================


def get_early_stop_point(func_asm_path: str):
    start_addr = ''
    end_addr = ''
    with open(func_asm_path, 'r') as f:
        asm_txt = f.read()
        lines = asm_txt.split('\n')
        lines.reverse()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            idx += 1
            if line.startswith(';') or len(line) < 1:
                continue
            if line.startswith('0x'):
                early_stop_addr = line.split(':')[0]
                opcode = line[44:].strip()
                opcode = opcode.split(' ')[0]
                if opcode.startswith('j'):  # the last jump
                    prev_line = lines[idx]
                    loop_size = prev_line.split(' ')[-1].strip()
                    if loop_size[0].isnumeric():
                        loop_size = int(loop_size, 16)
                        if loop_size < 64:
                            early_stop_addr = early_stop_addr + '_ignore'
                    else:
                        loop_size = 64
                        early_stop_addr = early_stop_addr+'_ignore'
                    break
    loop_size = max(loop_size, 64)
    return early_stop_addr, loop_size * 4  # 4 <- length of float


def find_rand_addr(prog_path: str, in_data: str, log_path: str, label_file_path: str):
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    label_file_path = os.path.abspath(label_file_path)
    # --- get conv2d functions' name
    funcs_name_list = []
    with open(label_file_path, 'r') as f:
        labels = f.read()
        lines = labels.split('\n')
        for line in lines:
            if ':' not in line:
                continue
            name, label = line.split(':')
            if len(label.strip()) > 0 and ('dense' in label or 'conv2d' in label):
                name = name.strip()
                funcs_name_list.append(name)

    func_addrs = dict()  # start_addr, end_addr, mid_output_addr
    for func_name in funcs_name_list:
        print(func_name)
        mem_write_log_path = log_path
        mem_write_log_path = os.path.abspath(mem_write_log_path)
        prog_path = prog_path
        data_path = in_data

        func_asm_path = os.path.join(funcs_dir, func_name)
        func_asm_path = os.path.abspath(func_asm_path)
        start_addr, end_addr = get_func_range(func_asm_path)
        early_stop_addr, loop_size = get_early_stop_point(func_asm_path)
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        out_mem = (0, 0)
        for mem_blk in write_mem_regions:
            if (mem_blk[1] - mem_blk[0]) > (out_mem[1] - out_mem[0]):
                out_mem = mem_blk
        rnd_addr = random.randrange(out_mem[0], out_mem[1], 4)
        #mid_addr = out_mem[0] + (out_mem[1] - out_mem[0])/2
        #mid_addr = int(mid_addr)
        #mid_addr = hex(mid_addr)
        rnd_addr = hex(rnd_addr)
        func_addrs[func_name] = (start_addr, end_addr, early_stop_addr, loop_size, rnd_addr, (hex(out_mem[0]), hex(out_mem[1])))
    return func_addrs


def handle_all_conv(prog_path: str, in_data: str, label_file_path: str, func_trace_map: dict):
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
            if len(label.strip()) > 0 and ('conv2d' in label or 'dense' in label):
                name = name.strip()
                funcs_name_list.append(name)
                func_types[name] = label.strip()

    func_shape = dict()
    for func_name in funcs_name_list:
        print(func_name)
        # --- recover the shape of each layer
        # tmp_log_path = './inst_trace.log'
        tmp_log_path = func_trace_map[func_name]  # generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)
        exp_log_path = './mem_exp.log'  # tmp file
        generate_symbolic_expression(func_name, tmp_log_path, exp_log_path)

        # --- try to interpret the filter shape from symbolic expression log
        mem_read_log_path = 'mem_read.log'  # tmp file
        mem_write_log_path = 'mem_write.log'  # tmp file
        filter_shape = recover_shape(func_name, exp_log_path,
                                     mem_read_log_path, mem_write_log_path,
                                     prog_path, in_data, func_type=func_types[func_name])
        func_shape[func_name] = filter_shape
    return func_shape


# ==============================================================
# Extract parameters
# ==============================================================


def extract_params(prog_path: str, in_data: str, w_shape: tuple, dump_point: str, log_path: str, func_name: str, func_type='conv2d'):
    """
    :param dump_point: the start address of layer function (before reshaping the parameters)
    """
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    if func_type == 'conv2d':
        dwords_len = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
    elif func_type == 'dense':
        dwords_len = w_shape[0] * w_shape[1]
    elif func_type == 'add':
        dwords_len = w_shape[0] * w_shape[1]
    rm_log(log_path)
    dump_dwords_2(prog_path, in_data, dump_point, dwords_len, log_path)

    # then convert dwords to floats
    with open(log_path, 'r') as f:
        dw_txt = f.read()
        f.close()
        end_count = dw_txt.count('end')
        dw_segs = dw_txt.split('end')[:end_count]
        for i in range(end_count):
            dw_txt = dw_segs[i].strip()
            dw_txt = dw_txt[dw_txt.find('\n')+1:]
            float_array = convert_dwords2float(dw_txt, dwords_len)

            w = np.asarray(float_array)
            w = w.reshape(w_shape)
            # print(type(w))
            lists = w.tolist()
            json_str = json.dumps(lists)
            # print(json_str)
            if func_type == 'conv2d':
                json_name = func_name[:func_name.rfind('.')] + '.weights_{}.json'.format(i)
            elif func_type == 'dense':
                json_name = func_name[:func_name.rfind('.')] + '.dense_weights_{}.json'.format(i)
            elif func_type == 'add':
                json_name = func_name[:func_name.rfind('.')] + '.biases_{}.json'.format(i)
            with open(json_name, 'w') as wf:
                wf.write(json_str)
                wf.close()
    rm_log(log_path)


def extract_params_glow_conv2d(prog_path: str, in_data: str, w_shape: tuple, dump_point: str, log_path: str, func_name: str, reg_num=1, func_type=''):
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    dwords_len = 1
    for w in w_shape:
        dwords_len *= w
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
            dw_txt = dw_txt[dw_txt.find('\n')+1:]
            float_array = convert_dwords2float(dw_txt, dwords_len)

            w = np.asarray(float_array)
            w = w.reshape(w_shape)
            if 'DKKC8' in func_name:
                # w = np.swapaxes(w, 0, 3)
                w = np.transpose(w, (3, 1, 2, 0, 4))
                new_shape = (w_shape[3], w_shape[1], w_shape[2], w_shape[0]*w_shape[4])
                w = np.reshape(w, new_shape, order='C')
                w = np.transpose(w, (3, 0, 1, 2))
            if reg_num == 1 and len(w_shape) == 4 and func_type != 'dense':
                w = np.transpose(w, (0, 3, 1, 2))  # store weights in (N,H,W,C) format
            elif reg_num == 1 and len(w_shape) == 4 and func_type == 'dense':
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
            if len(w_shape) == 5 and 'DKKC8' in func_name:
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


if __name__ == '__main__':
    def tmp_handle_func_call(func_call_trace: str):
        id_list = []
        idx = 0
        with open(func_call_trace) as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('0x'):
                    continue
                in_addrs = line.split(':')[2].strip()
                in_addrs = in_addrs[in_addrs.find('in ')+3:]
                in_addrs = in_addrs[:in_addrs.find('out')]
                in_addrs = in_addrs.strip(' ,')
                in_addrs = in_addrs.split(',')
                out_addr = line[line.find('out')+4:].strip()

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


    tmp_handle_func_call('/home/lifter/Documents/tvm_output/func_call.txt')
