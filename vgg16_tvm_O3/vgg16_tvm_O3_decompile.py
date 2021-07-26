#! /usr/bin/python3
import os
import sys
import sys
sys.path.append("../..")
import json
import numpy as np
from scripts import utils
"""
from scripts.pin_tools import func_call_trace, inst_trace_log, mem_read_log, mem_write_log, dump_dwords_2, \
    convert_dwords2float, rm_log
from scripts.se_engine import lightweight_SymEx
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
"""


if __name__ == '__main__':
    utils.funcs_dir = '/export/d1/zliudc/TVM/vgg16_tvm_O3_funcs/'

    prog_path = '/export/d1/zliudc/TVM/scripts/vgg16_tvm_O3/vgg16_tvm_O3'
    in_data = '/export/d1/zliudc/TVM/scripts/vgg16_tvm_O3/cat.bin'
    log_path = '/home/lifter/Documents/tvm_output/vgg16_tvm_O3_func_call.log'
    label_file = '/home/lifter/Documents/tvm_output/step1.txt'

    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = 'mem_read.log'
    mem_write_log_path = 'mem_write.log'
    # ==============================================================
    # Step 1 --- Get the Sequence of Layers ---
    # ==============================================================
    """
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm')
    utils.print_layer_label_tvm(log_path)
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm', only_fused=True)
    utils.print_layer_label_tvm(log_path, only_fused=True)
    exit(0)
    """
    """
    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, optimized=True)
    for name, result in func_shape.items():
        print(name)
        print(result)
    exit(0)
    """
    # ==============================================================
    # Step 2 --- Recover the Shape of each Layer
    # ==============================================================
    """
    # conv2d layers
    func_type = 'conv2d'
    # func_name = '0046.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2_2'  # 512, 256, 3, 3
    # func_name = '0049.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4_2'  # 256, 128, 3, 3
    # func_name = '0052.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_8_2'  # 64, 3, 3, 3  [2.0, 1, 3.0, 3.0, 3.0, 32.0]
    # func_name = '0004.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_3_2'  # 256, 256, 3, 3
    func_name = '0007.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_6_2'  # 128, 64, 3, 3
    # func_name = '0012.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1_2'  # 512, 512, 3, 3
    # func_name = '0025.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_5_2'  # 128, 128, 3, 3
    # func_name = '0028.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2'  # 512, 512, 3, 3
    # func_name = '0041.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_7_2'  # 64, 64, 3, 3  [2, 4, 3, 3, 16, 32]
    # dense/fully-connected layers
    # func_type = 'dense'
    # func_name = '0058.txt.fused_nn_dense_add_nn_relu_1'  # 4096, 4096
    # func_name = '0060.txt.fused_nn_dense_add_nn_relu_1_1'  # 4096, 25088
    # func_name = '0030.txt.fused_nn_dense_add_1'  # 1000, 4096
    # bias add layer
    # func_type = 'add'
    # add is merged into conv2d, no more independent add function
    # max-poll layers
    # func_type = 'max'
    # func_name = '0056.txt.fused_nn_max_pool2d_3_1'  # max 2, 2, kernel, stride
    # func_name = '0009.txt.fused_nn_max_pool2d_2_1'  # 2, 2
    # func_name = '0016.txt.fused_nn_max_pool2d_1'  # 2, 2
    # func_name = '0020.txt.fused_nn_max_pool2d_1_1'  # 2, 2
    # func_name = '0043.txt.fused_nn_max_pool2d_4_1'  # 2, 2

    #utils.generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)

    #utils.generate_symbolic_expression(func_name, tmp_log_path, exp_log_path, max_inst=5000000)

    # --- try to interpret the filter shape from symbolic expression log
    filter_shape, layout_shape = utils.recover_shape_tvm(func_name, exp_log_path,
                                    mem_read_log_path, mem_write_log_path,
                                    prog_path, in_data, func_type=func_type, optimized=True)
    print(filter_shape)
    print(layout_shape)
    exit(0)
    """
    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================
    mem_dump_log_path = 'mem_dump.log'
    func_meta_data = [('0046.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2_2', (512, 256, 3, 3), '0x41E0B0', 'conv2d',
                       [16, 1, 3, 3, 256, 32]),
                      ('0049.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4_2', (256, 128, 3, 3), '0x4225E0', 'conv2d',
                       [8, 4, 3, 3, 32, 32]),
                      ('0052.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_8_2', (64, 3, 3, 3), '0x426F90', 'conv2d',
                       [2, 1, 3, 3, 3, 32]),
                      ('0004.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_3_2', (256, 256, 3, 3), '0x401FA0', 'conv2d',
                       [8, 1, 3, 3, 256, 32]),
                      ('0007.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_6_2', (128, 64, 3, 3), '0x4069D0', 'conv2d',
                       [2, 2, 3, 3, 32, 64]),
                      ('0012.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1_2', (512, 512, 3, 3), '0x40B0E0', 'conv2d',
                       [32, 1, 3, 3, 512, 16]),
                      ('0025.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_5_2', (128, 128, 3, 3), '0x40FB20', 'conv2d',
                       [2, 2, 3, 3, 64, 64]),
                      ('0028.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2', (512, 512, 3, 3), '0x413A80', 'conv2d',
                       [32, 1, 3, 3, 512, 16]),
                      ('0041.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_7_2', (64, 64, 3, 3), '0x4192E0', 'conv2d',
                       [2, 4, 3, 3, 16, 32]),

                      ('0046.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2_2', (1, 512), '0x41E0B0', 'add'),
                      ('0049.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4_2', (1, 256), '0x4225E0', 'add'),
                      ('0052.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_8_2', (1, 64), '0x426F90', 'add'),
                      ('0004.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_3_2', (1, 256), '0x401FA0', 'add'),
                      ('0007.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_6_2', (1, 128), '0x4069D0', 'add'),
                      ('0012.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1_2', (1, 512), '0x40B0E0', 'add'),
                      ('0025.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_5_2', (1, 128), '0x40FB20', 'add'),
                      ('0028.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2', (1, 512), '0x413A80', 'add'),
                      ('0041.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_7_2', (1, 64), '0x4192E0', 'add'),

                      ('0058.txt.fused_nn_dense_add_nn_relu_1', (4096, 4096), '0x42BD40', 'dense'),
                      ('0060.txt.fused_nn_dense_add_nn_relu_1_1', (4096, 25088), '0x42C6E0', 'dense'),
                      ('0030.txt.fused_nn_dense_add_1', (1000, 4096), '0x416D40', 'dense'),

                      ('0058.txt.fused_nn_dense_add_nn_relu_1', (1, 4096), '0x42BD40', 'add'),
                      ('0060.txt.fused_nn_dense_add_nn_relu_1_1', (1, 4096), '0x42C6E0', 'add'),
                      ('0030.txt.fused_nn_dense_add_1', (1, 1000), '0x416D40', 'add'),
                      ]
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        func_type = fun_data[3]
        data_index = 1  # conv2d weights
        layout_shape = ()
        if func_type == 'add':
            data_index = 2  # conv2d biases
        elif func_type == 'conv2d':
            layout_shape = tuple(fun_data[4])

        #if func_type != 'conv2d' or w_shape[0] != 64:  # for debug
        #    continue

        utils.extract_params_tvm(prog_path, in_data, w_shape, dump_point, mem_dump_log_path, func_name,
                                 func_type=func_type, data_idx=data_index, special_layout=layout_shape)
