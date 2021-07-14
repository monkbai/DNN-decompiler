#! /usr/bin/python3
import os
import sys
import math
from scripts import utils
from scripts import pin_tools


if __name__ == '__main__':
    utils.funcs_dir = '/home/lifter/Documents/tvm_output/scripts/ida/embedding/embedding_tvm_O0_funcs'

    # prepared in advance
    prog_path = '/home/lifter/Documents/tvm_output/embedding_tvm_O0'
    in_data = ''  # no input needed
    label_file = './step1.txt'
    config_file = './config.json'

    # (tmp files) generated during analysis
    log_path = './embedding_tvm_O0_func_call.log'
    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = './mem_read.log'
    mem_write_log_path = './mem_write.log'
    mem_dump_log_path = './mem_dump.log'

    # ===============================
    # Recover the Computational Graph
    # ===============================
    # with parameters from entry functions
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm', only_fused=True)
    param_list = utils.print_layer_label_tvm(log_path, only_fused=True)

    # without parameters, only start addresses
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm', only_fused=False)
    utils.print_layer_label_tvm(log_path, only_fused=False)

    # ===============================
    # Recover other layers
    # ===============================
    func_data = {
                 'embedding': '0037.sub_405A30.txt',  # take
                 'avg_pool': '0025.sub_403330.txt',
                 'matmul': '0031.sub_4046E0.txt',
                 'add': '0019.sub_401D70.txt'
                 }

    for func_type, func_name in func_data.items():
        utils.generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)

        utils.generate_symbolic_expression(func_name, tmp_log_path, exp_log_path, max_inst=5000000)

        shape = utils.recover_shape_tvm(func_name, exp_log_path,
                                        mem_read_log_path, mem_write_log_path,
                                        prog_path, in_data, func_type=func_type)
        print(shape)
        if 'embedding' in func_type:
            embedding_start = int('0x41ec40', 16)
            for param in param_list:
                if param > embedding_start:
                    dict_size = (param - embedding_start)/4/shape
                    dict_size = math.floor(dict_size)
                    print('embedding dict size: {}'.format(dict_size))
                    break

    # ===============================
    # Extract Parameters
    # ===============================
    # func_meta_data is collected from previous output
    func_meta_data = [
                      ('0037.sub_405A30.txt', (25006, 100), '0x405a30', '0x41ec40', 'embedding'),
                      ('0031.sub_4046E0.txt', (1, 100), '0x4046e0', '0xdacc40', 'matmul'),
                      ('0019.sub_401D70.txt', (1, 1), '0x401d70', '0x415440', 'add'),
                      ]
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        dump_addr = fun_data[3]
        func_type = fun_data[4]
        utils.extract_params_general(prog_path, in_data, w_shape, dump_point,
                                     mem_dump_log_path, func_name, dump_addr)
