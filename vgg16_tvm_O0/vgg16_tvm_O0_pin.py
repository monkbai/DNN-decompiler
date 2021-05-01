#! /usr/bin/python3
import os
import sys
import json
import numpy as np
from scripts.pin_tools import func_call_trace, inst_trace_log, mem_read_log, mem_write_log, dump_dwords_2, \
    convert_dwords2float, rm_log
from scripts.myUtils import lightweight_SymEx
from scripts.mem_slices import memory_slices
from scripts.explain import explain_tvm_conv2d_result, explain_tvm_dense_result


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

funcs_dir = './vgg16_strip_funcs/'
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
                addr = name.strip()
                addr = addr.split('_')[1]
                addr = addr.split('.')[0]
                addr = '0x' + addr

                addr_list.append(addr)

                addr2label[addr] = label.strip()
                addr2funcs[addr] = name.strip()
    return addr_list


def get_funcs_trace(prog_path: str, in_data: str, log_path: str, label_file: str):

    # prog_path = './vgg16_strip'
    prog_path = os.path.abspath(prog_path)
    # in_data = './cat.bin'
    in_data = os.path.abspath(in_data)
    # log_path = './vgg16_strip_func_call.log'
    log_path = os.path.abspath(log_path)
    # label_file = './step1.txt'
    label_file = os.path.abspath(label_file)

    addr_list = get_addr_list(label_file)
    # addr_list = '0x40d89a,0x42a324,0x42dcc0,0x417b3e,0x4221f5,0x420ee0,0x421b40,0x4213f5,0x42065a,0x40e400,0x41c11e,0x42953e,0x406bca,0x428d9a,0x418edf,0x42d740,0x42286a,0x41979a,0x41826a,0x41688e,0x420296,0x401e4e,0x4250f0,0x401720,0x42e224,0x41cd7a,0x420ac0,0x416330,0x402b4e,0x424d80,0x413f9a,0x40eafe,0x40383a,0x42de41,0x40f75a,0x429a40,0x429160,0x40746a,0x42dad2,0x429ad9,0x4127ea,0x42867a,0x425dba,0x417100,0x41f9ce,0x429c62,0x4011f4,0x40dd9e,0x42da02,0x42dbb0,0x4257f4,0x429cc0,0x417533,0x41346e,0x40a8ba'

    func_call_trace(prog_path, in_data, addr_list, log_path)

    dict_to_json(addr2label, './addr2label.json')
    dict_to_json(addr2funcs, './addr2funcs.json')


def print_layer_label(trace_log_path: str):
    global addr2label, addr2funcs
    addr2label = json_to_dict('./addr2label.json')
    addr2funcs = json_to_dict('./addr2funcs.json')

    trace_log_path = os.path.abspath(trace_log_path)
    with open(trace_log_path, 'r') as f:
        trace_txt = f.read()
        lines = trace_txt.split('\n')
        for line in lines:
            if not line.startswith('0x'):
                continue
            addr = hex(int(line.strip(), 16))
            if 'reshape' != addr2label[addr]:
                print('{}: {}'.format(addr, addr2label[addr]))


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


def generate_symbolic_expression(func_name: str, inst_log_path: str, exp_log_path: str):
    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)

    inst_log_path = os.path.abspath(inst_log_path)
    exp_log_path = os.path.abspath(exp_log_path)

    lightweight_SymEx(func_asm_path, inst_log_path, exp_log_path, max_inst_num=5000000)  # TODO: max_inst_num


# ==============================================================
# Recover ses shapes using heuristics
# ==============================================================


def recover_shape(func_name: str, mem_exp_log: str,
                  mem_read_log_path: str, mem_write_log_path: str,
                  prog_path: str, data_path: str, func_type='conv2d'):
    mem_read_log_path = os.path.abspath(mem_read_log_path)
    mem_write_log_path = os.path.abspath(mem_write_log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)

    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)
    start_addr, end_addr = get_func_range(func_asm_path)

    if func_type == 'conv2d':
        mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, data_path)
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        read_mem_regions = memory_slices(mem_read_log_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        filter_shape, input_shape, output_shape = explain_tvm_conv2d_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return filter_shape
    elif func_type == 'dense':
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        input_size, output_size = explain_tvm_dense_result(mem_exp_log, write_mem_regions)
        return output_size, input_size


# ==============================================================
# Handle all conv2d functions
# ==============================================================


def handle_all_conv(label_file_path: str):
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
            if len(label.strip()) > 0 and 'conv2d' in label and not name.startswith('0126'):
                name = name.strip()
                funcs_name_list.append(name)

    func_shape = dict()
    for func_name in funcs_name_list:
        print(func_name)
        # --- recover the shape of each layer
        tmp_log_path = './inst_trace.log'
        generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)
        exp_log_path = './mem_exp.log'
        generate_symbolic_expression(func_name, tmp_log_path, exp_log_path)

        # --- try to interpret the filter shape from symbolic expression log
        mem_read_log_path = 'mem_read.log'
        mem_write_log_path = 'mem_write.log'
        filter_shape = recover_shape(func_name, exp_log_path, mem_read_log_path, mem_write_log_path, prog_path, in_data)
        func_shape[func_name] = filter_shape
    return func_shape


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
            with open(json_name, 'w') as wf:
                wf.write(json_str)
                wf.close()
    rm_log(log_path)


if __name__ == '__main__':
    prog_path = './vgg16_strip'
    in_data = './cat.bin'
    log_path = './vgg16_strip_func_call.log'
    label_file = './step1.txt'

    # ==============================================================
    # Step 1 --- Get the Sequence of Layers ---
    # ==============================================================
    # get_funcs_trace(prog_path, in_data, log_path, label_file)
    # print_layer_label(log_path)
    """
    func_shape = handle_all_conv(label_file)
    for name, result in func_shape.items():
        print(name)
        print(result)
    """

    # ==============================================================
    # Step 2 --- Recover the Shape of each Layer
    # ==============================================================
    #func_name = '0139.function_429530.txt'  # 4096 * 25088
    #func_name = '0091.function_417b30.txt'  # 4096 * 4096
    func_name = '0067.function_40dd90.txt'  # 1000 * 4096
    tmp_log_path = './inst_trace.log'
    # generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)
    exp_log_path = './mem_exp.log'
    # generate_symbolic_expression(func_name, tmp_log_path, exp_log_path)

    # --- try to interpret the filter shape from symbolic expression log
    mem_read_log_path = 'mem_read.log'
    mem_write_log_path = 'mem_write.log'
    # shape = recover_shape(func_name, exp_log_path, mem_read_log_path, mem_write_log_path, prog_path, in_data, func_type='dense')
    # print(shape)

    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================
    mem_dump_log_path = 'mem_dump.log'
    func_meta_data = [('0122.function_422860.txt', (64, 3, 3, 3), '0x421c20', 'conv2d'),
                      ('0098.function_419790.txt', (512, 512, 3, 3), '0x4183e0', 'conv2d'),
                      ('0103.function_41cd70.txt', (512, 256, 3, 3), '0x41ba90', 'conv2d'),
                      ('0081.function_413f90.txt', (128, 64, 3, 3), '0x412c50', 'conv2d'),
                      ('0053.function_403830.txt', (256, 128, 3, 3), '0x402320', 'conv2d'),
                      ('0074.function_40f750.txt', (256, 256, 3, 3), '0x40e470', 'conv2d'),
                      ('0058.function_407460.txt', (128, 128, 3, 3), '0x406540', 'conv2d'),
                      ('0131.function_425db0.txt', (512, 512, 3, 3), '0x425170', 'conv2d'),
                      ('0063.function_40a8b0.txt', (64, 64, 3, 3), '0x409680', 'conv2d'),
                      ('0139.function_429530.txt', (4096, 25088), '0x4291c0', 'dense'),
                      ('0091.function_417b30.txt', (4096, 4096), '0x4177c0', 'dense'),
                      ('0067.function_40dd90.txt', (1000, 4096), '0x40da20', 'dense'), ]
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        func_type = fun_data[3]
        if func_type != 'dense':
            continue
        extract_params(prog_path, in_data, w_shape, dump_point, mem_dump_log_path, func_name, func_type)
