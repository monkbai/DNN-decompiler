#! /usr/bin/python3
import os
import sys
import json
from scripts import utils
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from scripts.pin_tools import func_call_trace, inst_trace_log, mem_read_log, mem_write_log, dump_dwords_2
from scripts.pin_tools import fused_rdi
from scripts.pin_tools import convert_dwords2float, rm_log
from scripts.se_engine import lightweight_SymEx
from scripts.mem_slices import memory_slices
from scripts.explain import explain_tvm_conv2d_result, explain_tvm_dense_result
from scripts.explain import explain_tvm_add_result, explain_tvm_maxpool_result

"""
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

funcs_dir = './resnet_O0_funcs/'
# ==============================================================
# Generate the function call trace
# ==============================================================


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
                addr = name.strip()
                addr = addr.split('_')[1]
                addr = addr.split('.')[0]
                addr = '0x' + addr

                addr2label[addr] = label.strip()
                addr2funcs[addr] = name.strip()
                if fused and 'fused' not in label:
                    continue
                elif not fused and 'fused' in label:
                    continue

                addr_list.append(addr)
    return addr_list


def get_funcs_trace(prog_path: str, in_data: str, log_path: str, label_file: str, only_fused=False):

    # prog_path = './vgg16_strip'
    prog_path = os.path.abspath(prog_path)
    # in_data = './cat.bin'
    in_data = os.path.abspath(in_data)
    # log_path = './vgg16_strip_func_call.log'
    log_path = os.path.abspath(log_path)
    # label_file = './step1.txt'
    label_file = os.path.abspath(label_file)

    addr_list = get_addr_list(label_file, fused=only_fused)
    # addr_list = '0x40d89a,0x42a324,0x42dcc0,0x417b3e,0x4221f5,0x420ee0,0x421b40,0x4213f5,0x42065a,0x40e400,0x41c11e,0x42953e,0x406bca,0x428d9a,0x418edf,0x42d740,0x42286a,0x41979a,0x41826a,0x41688e,0x420296,0x401e4e,0x4250f0,0x401720,0x42e224,0x41cd7a,0x420ac0,0x416330,0x402b4e,0x424d80,0x413f9a,0x40eafe,0x40383a,0x42de41,0x40f75a,0x429a40,0x429160,0x40746a,0x42dad2,0x429ad9,0x4127ea,0x42867a,0x425dba,0x417100,0x41f9ce,0x429c62,0x4011f4,0x40dd9e,0x42da02,0x42dbb0,0x4257f4,0x429cc0,0x417533,0x41346e,0x40a8ba'

    # tmp_log_path = os.path.abspath('./fused_data.log')
    fused_rdi(prog_path, in_data, addr_list, log_path)
    # func_call_trace(prog_path, in_data, addr_list, log_path)

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


def print_fused_trace(trace_log_path: str, new_log_path: str):
    global addr2label, addr2funcs
    addr2label = json_to_dict('./addr2label.json')
    addr2funcs = json_to_dict('./addr2funcs.json')

    addr2label_list = list(addr2label.items())
    addr2label_list = sorted(addr2label_list, key=lambda x: x[0])

    trace_log_path = os.path.abspath(trace_log_path)
    new_log_path = os.path.abspath(new_log_path)
    new_log = open(new_log_path, 'w')
    with open(trace_log_path, 'r') as f:
        trace_txt = f.read()
        lines = trace_txt.split('\n')
        for line in lines:
            if not line.startswith('0x'):
                continue
            addr = line.split(':')[0]
            addr = hex(int(addr.strip(), 16))
            if 'reshape' in addr2label[addr]:
                continue
            elif 'fused' in addr2label[addr]:
                idx = addr2label_list.index((addr, addr2label[addr]))
                new_log.write('{}, {: <18}'.format(addr2label_list[idx+1][0], addr2label_list[idx+1][1]) + ' - ')
                new_log.write(line + '\n')


def get_call_graph_list(trace_log_path: str):
    global addr2label, addr2funcs

    trace_log_path = os.path.abspath(trace_log_path)
    call_graph_list = []
    with open(trace_log_path, 'r') as f:
        trace_txt = f.read()
        lines = trace_txt.split('\n')
        for line in lines:
            if not line.startswith('0x'):
                continue
            addr_label = line.split('-')[0].strip()
            addr = addr_label.split(',')[0]
            label = addr_label.split(',')[1].strip()
            args = line.split(':')[1].strip()
            args = args.strip(',')
            args = args.split(',')
            call_graph_list.append(((addr, label), args))
        return call_graph_list


def show_graph(call_graph: list):
    dg = nx.DiGraph()
    for i in range(len(call_graph)):
        node, args = call_graph[i]
        dg.add_node(str((i, node[0], node[1])))

    for i in range(len(call_graph)):
        node, args = call_graph[i]
        addr = node[0]
        label = node[1]
        j = i + 1
        while j < len(call_graph):
            c_node, c_args = call_graph[j]
            in_c_args = c_args[:-1]
            if args[-1] in in_c_args:  # or j == i+1:
                dg.add_edge(str((i, addr, label)), str((j, c_node[0], c_node[1])))
            if args[-1] == c_args[-1]:
                break
            j += 1
    net = Network(notebook=True)
    net.height = 1000
    net.width = 1000
    net.from_nx(dg)
    net.show('tmp.html')
    # nx.draw_spectral(dg, with_labels=True)
    # plt.savefig("tmp.png")


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
    elif func_type == 'add':
        mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, data_path)
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        read_mem_regions = memory_slices(mem_read_log_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        if len(read_mem_regions) > 1 and \
                read_mem_regions[0][1] - read_mem_regions[0][0] == read_mem_regions[1][1] - read_mem_regions[1][0]:
            # the case of add layer after dense/fully-connected layer
            return (read_mem_regions[0][1] - read_mem_regions[0][0]) / 4
        bias_length = explain_tvm_add_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return bias_length
    elif func_type.startswith('max'):
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        kernel_size, stride = explain_tvm_maxpool_result(mem_exp_log, write_mem_regions)
        return kernel_size, stride


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
            if len(label.strip()) > 0 and 'conv2d' in label:
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


def extract_params(prog_path: str, in_data: str, w_shape: tuple, dump_point: str, log_path: str, func_name: str, func_type='conv2d', data_index=1):

    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    dwords_len = 0
    if func_type == 'conv2d':
        dwords_len = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
    elif func_type == 'dense' or func_type == 'add':
        dwords_len = w_shape[0] * w_shape[1]
    elif func_type == 'var' or func_type == 'gamma' or func_type == 'mean' or func_type == 'beta':
        dwords_len = w_shape[0] * w_shape[1]
    else:
        assert 'the func_type {} is not defined'.format(func_type) and False

    # dump the memory
    rm_log(log_path)
    dump_dwords_2(prog_path, in_data, dump_point, dwords_len, log_path, data_index)

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
            elif func_type == 'var':
                json_name = func_name[:func_name.rfind('.')] + '.var_{}.json'.format(i)
            elif func_type == 'gamma':
                json_name = func_name[:func_name.rfind('.')] + '.gamma_{}.json'.format(i)
            elif func_type == 'mean':
                json_name = func_name[:func_name.rfind('.')] + '.mean_{}.json'.format(i)
            elif func_type == 'beta':
                json_name = func_name[:func_name.rfind('.')] + '.beta_{}.json'.format(i)
            else:
                assert 'func_type {} is not supported'.format(func_type) and False
            with open(json_name, 'w') as wf:
                wf.write(json_str)
                wf.close()
    rm_log(log_path)
"""

if __name__ == '__main__':
    utils.funcs_dir = '/home/lifter/Documents/tvm_output/resnet18_tvm_O3_funcs/'

    prog_path = '/home/lifter/Documents/tvm_output/resnet18_tvm_O3'
    in_data = '/home/lifter/Documents/tvm_output/cat.bin'
    log_path = './resnet18_tvm_O3_func_call_trace.log'
    label_file = './step1.txt'

    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = 'mem_read.log'
    mem_write_log_path = 'mem_write.log'
    # ==============================================================
    # Step 1 --- Get the Sequence of Layers ---
    # ==============================================================
    # get_funcs_trace(prog_path, in_data, log_path, label_file, only_fused=False)
    """
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm')
    utils.print_layer_label_tvm(log_path)
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm', only_fused=True)
    utils.print_layer_label_tvm(log_path, only_fused=True)
    utils.print_input_id(log_path)
    exit(0)
    """
    # ==============================================================
    # Step 2 --- Recover the Shape of each Layer
    # ==============================================================
    """
    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, compiler='tvm', optimized=True)
    for name, result in func_shape.items():
        print(name)
        print(result)
    exit(0)
    """
    """
    # conv2d layers
    # func_type = 'conv2d'
    # func_name = '0008.txt.fused_nn_contrib_conv2d_NCHWc_add_2_2'  # ([512.0, 512.0, 3, 3], [16.0, 1.0, 3, 3, 512.0, 32.0])
    # func_name = '0011.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_7_2'  # ([512.0, 512.0, 3, 3], [16.0, 1.0, 3, 3, 512.0, 32.0])
    # func_name = '0014.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4_2'  # ([256.0, 128.0, 3, 3], [8.0, 16.0, 3, 3, 8.0, 32.0])
    # func_name = '0017.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_4_2'  # ([128.0, 128.0, 3, 3], [4.0, 1.0, 3, 3, 128.0, 32.0])
    # func_name = '0020.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2_2'  # <-- this one is harder
    # ([128.0, 64.0, 3.0, 3.0], [2.0, 2.0, 3.0, 3.0, 32.0, 64.0])
    # func_name = '0027.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_6_2'  # ([512.0, 256.0, 3, 3], [16.0, 32.0, 3, 3, 8.0, 32.0])
    # func_name = '0032.txt.fused_nn_contrib_conv2d_NCHWc_add_1_2'  # ([256.0, 256.0, 3, 3], [16.0, 2.0, 3, 3, 128.0, 16.0])
    # func_name = '0035.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_2' # ([512.0, 512.0, 3, 3], [16.0, 1.0, 3, 3, 512.0, 32.0])
    # func_name = '0037.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_3_1'  # 1, 1 kernel is harder
    # ([256.0, 128, 1, 1], [16.0, 2.0, 1, 1, 64, 16.0])
    # func_name = '0055.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_5_1'  # ([128.0, 64, 1, 1], [4.0, 8.0, 1, 1, 8, 32.0])
    # func_name = '0058.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_3_2'  # ([128.0, 128.0, 3, 3], [4.0, 1.0, 3, 3, 128.0, 32.0])
    # func_name = '0061.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_6_2'  # ([64.0, 64.0, 3, 3], [2.0, 1.0, 3, 3, 64.0, 32.0])
    # func_name = '0064.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_5_2'  # ([256.0, 256.0, 3, 3], [16.0, 2.0, 3, 3, 128.0, 16.0])
    # func_name = '0069.txt.fused_nn_contrib_conv2d_NCHWc_add_2'  # ([128.0, 128.0, 3, 3], [4.0, 1.0, 3, 3, 128.0, 32.0])
    # func_name = '0072.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1_2'  # ([64.0, 64.0, 3, 3], [2.0, 1.0, 3, 3, 64.0, 32.0])
    # func_name = '0075.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_2_2'  # ([256.0, 256.0, 3, 3], [16.0, 2.0, 3, 3, 128.0, 16.0])
    # func_name = '0081.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2'  # <-- wrong shape prediction
    # ([64.0, 3.0, 7.0, 7.0], [2.0, 2.0, 7.0, 7.0, 1.5, 32.0])
    # func_name = '0001.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_1_1'  # ([512.0, 256, 1, 1], [16.0, 2.0, 1, 1, 128, 32.0])
    # dense/fully-connected layers
    # func_type = 'dense'
    # func_name = '0029.txt.fused_nn_dense_add_1'  # (1000.0, 512)
    # bias add layers are fused into the conv2d layers
    # max-poll layers
    # func_type = 'max'
    # func_name = '0047.txt.fused_nn_max_pool2d_1'  # max 3, 2, kernel, stride
    func_type = 'avg'
    func_name = '0077.txt.fused_nn_global_avg_pool2d_1'  # 7, 1
    #func_name = '0078.txt.fused_nn_global_avg_pool2d_2'  # which one?

    utils.generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)
    
    utils.generate_symbolic_expression(func_name, tmp_log_path, exp_log_path, max_inst=5000000)

    # --- try to interpret the filter shape from symbolic expression log
    shape = utils.recover_shape_tvm(func_name, exp_log_path,
                                    mem_read_log_path, mem_write_log_path,
                                    prog_path, in_data, func_type=func_type, optimized=True)
    print(shape)
    exit(0)
    """
    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================
    mem_dump_log_path = 'mem_dump.log'
    func_meta_data = [('0008.txt.fused_nn_contrib_conv2d_NCHWc_add_2_2', (512, 512, 3, 3), '0x404810', 'conv2d',
                       (16, 1, 3, 3, 512, 32), 1),
                      ('0011.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_7_2', (512, 512, 3, 3), '0x407D60', 'conv2d',
                       (16, 1, 3, 3, 512, 32), 1),
                      ('0014.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4_2', (256, 128, 3, 3), '0x40B360', 'conv2d',
                       (8, 16, 3, 3, 8, 32), 1),
                      ('0017.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_4_2', (128, 128, 3, 3), '0x40F1E0', 'conv2d',
                       (4, 1, 3, 3, 128, 32), 2),  # 'extra_add'
                      ('0020.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2_2', (128, 64, 3, 3), '0x413860', 'conv2d',
                       (2, 2, 3, 3, 32, 64), 1),
                      ('0027.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_6_2', (512, 256, 3, 3), '0x418A00', 'conv2d',
                       (16, 32, 3, 3, 8, 32), 1),
                      ('0032.txt.fused_nn_contrib_conv2d_NCHWc_add_1_2', (256, 256, 3, 3), '0x41C830', 'conv2d',
                       (16, 2, 3, 3, 128, 16), 1),
                      ('0035.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_2', (512, 512, 3, 3), '0x41F8E0', 'conv2d',
                       (16, 1, 3, 3, 512, 32), 2),  # 'extra_add'
                      ('0037.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_3_1', (256, 128, 1, 1), '0x423330', 'conv2d',
                       (16, 2, 1, 1, 64, 16), 1),  # 'extra_add'
                      ('0055.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_5_1', (128, 64, 1, 1), '0x42A470', 'conv2d',
                       (4, 8, 1, 1, 8, 32), 1),  # 'extra_add'
                      ('0058.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_3_2', (128, 128, 3, 3), '0x42C660', 'conv2d',
                       (4, 1, 3, 3, 128, 32), 1),
                      ('0061.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_6_2', (64, 64, 3, 3), '0x430940', 'conv2d',
                       (2, 1, 3, 3, 64, 32), 2),  # 'extra_add'
                      ('0064.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_5_2', (256, 256, 3, 3), '0x434E70', 'conv2d',
                       (16, 2, 3, 3, 128, 16), 1),
                      ('0069.txt.fused_nn_contrib_conv2d_NCHWc_add_2', (128, 128, 3, 3), '0x438930', 'conv2d',
                       (4, 1, 3, 3, 128, 32), 1),
                      ('0072.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1_2', (64, 64, 3, 3), '0x43CB60', 'conv2d',
                       (2, 1, 3, 3, 64, 32), 1),
                      ('0075.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_2_2', (256, 256, 3, 3), '0x440CF0', 'conv2d',
                       (16, 2, 3, 3, 128, 16), 2),  # 'extra_add'
                      # ('0081.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2', (64, 3, 7, 7), '0x444E30', 'conv2d', (2, 2, 7, 7, 1.5, 32)),
                      ('0081.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2', (64, 3, 7, 7), '0x444E30', 'conv2d',
                       (2, 1, 7, 7, 3, 32), 1),
                      ('0001.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_1_1', (512, 256, 1, 1), '0x401550', 'conv2d',
                       (16, 2, 1, 1, 128, 32), 1),  # 'extra_add'

                      ('0008.txt.fused_nn_contrib_conv2d_NCHWc_add_2_2', (1, 512), '0x404810', 'add', 2),
                      ('0011.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_7_2', (1, 512), '0x407D60', 'add', 2),
                      ('0014.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_4_2', (1, 256), '0x40B360', 'add', 2),
                      ('0017.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_4_2', (1, 128), '0x40F1E0', 'add', 3),  # 'extra_add'
                      ('0020.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2_2', (1, 128), '0x413860', 'add', 2),
                      ('0027.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_6_2', (1, 512), '0x418A00', 'add', 2),
                      ('0032.txt.fused_nn_contrib_conv2d_NCHWc_add_1_2', (1, 256), '0x41C830', 'add', 2),
                      ('0035.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_2', (1, 512), '0x41F8E0', 'add', 3),  # 'extra_add'
                      ('0037.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_3_1', (1, 256), '0x423330', 'add', 2),  # 'extra_add'
                      ('0055.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_5_1', (1, 128), '0x42A470', 'add', 2),  # 'extra_add'
                      ('0058.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_3_2', (1, 128), '0x42C660', 'add', 2),
                      ('0061.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_6_2', (1, 64), '0x430940', 'add', 3),  # 'extra_add'
                      ('0064.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_5_2', (1, 256), '0x434E70', 'add', 2),
                      ('0069.txt.fused_nn_contrib_conv2d_NCHWc_add_2', (1, 128), '0x438930', 'add', 2),
                      ('0072.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1_2', (1, 64), '0x43CB60', 'add', 2),
                      ('0075.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_2_2', (1, 256), '0x440CF0', 'add', 3),  # 'extra_add'
                      ('0081.txt.fused_nn_contrib_conv2d_NCHWc_add_nn_relu_2', (1, 64), '0x444E30', 'add', 2),
                      ('0001.txt.fused_nn_contrib_conv2d_NCHWc_add_add_nn_relu_1_1', (1, 512), '0x401550', 'add', 2),  # 'extra_add'

                      ('0029.txt.fused_nn_dense_add_1', (1000, 512), '0x41BEA0', 'dense', 1),
                      ('0029.txt.fused_nn_dense_add_1', (1, 1000), '0x41BEA0', 'add', 2),
                      ]

    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        func_type = fun_data[3]
        data_index = fun_data[-1]
        layout_shape = ()
        if func_type == 'conv2d':
            layout_shape = tuple(fun_data[4])


        #if func_type != 'conv2d' or w_shape[0] != 64 or w_shape[1] != 3:  # for debug
        #    continue

        utils.extract_params_tvm(prog_path, in_data, w_shape, dump_point, mem_dump_log_path, func_name,
                                 func_type=func_type, data_idx=data_index, special_layout=layout_shape)
