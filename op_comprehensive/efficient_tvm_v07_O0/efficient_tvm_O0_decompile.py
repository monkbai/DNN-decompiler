#! /usr/bin/python3
import os
import sys
import json
import math
sys.path.append("../..")
import trace_filter
import utils
from utils import list_to_json, dict_to_json, json_to_list, json_to_dict
import se_engine
import logging
from fused_trace import fuse_batchnorm
print('get logger: {}'.format('decompiler.'+__name__))
logger = logging.getLogger('decompiler.'+__name__)


if __name__ == '__main__':
    utils.funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.7/efficientnet_tvm_O0/efficientnet_funcs/"
    prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.7/efficientnet_tvm_O0/efficientnet_lite4_tvm_O0_strip"
    in_data = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.7/efficientnet_tvm_O0/cat.bin"
    log_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.7/efficientnet_tvm_O0/func_call.log"
    label_file = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.7/efficientnet_tvm_O0/ground_truth.txt"

    # utils.funcs_dir = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/efficientnet_tvm_O0/efficientnet_funcs/"
    # prog_path = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/efficientnet_tvm_O0/efficientnet_lite4_tvm_O0_strip"
    # in_data = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/efficientnet_tvm_O0/cat.bin"
    # log_path = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/efficientnet_tvm_O0/func_call.log"
    # label_file = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/efficientnet_tvm_O0/ground_truth.txt"

    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = './mem_read.log'
    mem_write_log_path = './mem_write.log'
    mem_dump_log_path = 'mem_dump.log'

    # ==============================================================
    # Step 1 --- Get the Sequence of Layers ---
    # ==============================================================

    # get_funcs_trace(prog_path, in_data, log_path, label_file, only_fused=False)
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm')
    utils.print_layer_label_tvm(log_path)
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm', only_fused=True)
    utils.print_layer_label_tvm(log_path, config_path='config.json', only_fused=True)
    func_meta_data, topo_list = utils.print_input_id(log_path)  # to reconstruct the conputational graph
    # exit(0)

    # ==============================================================
    # Step 2 --- Recover the Shape of each Layer
    # ==============================================================
    
    # Step 2.1 Generate and Filter Trace
    
    logger.info('START')
    func_trace_map = {}
    func_rndaddr_map = {}
    asm_files = os.listdir(utils.funcs_dir)
    for asm_file in asm_files:
        if 'labels' not in asm_file and asm_file.endswith('.txt'):
            asm_path = os.path.join(utils.funcs_dir, asm_file)
            start_addr, _ = utils.get_func_range(asm_path)
            if start_addr in utils.addr2label.keys():
                if 'dense' in utils.addr2label[start_addr] or 'conv' in utils.addr2label[start_addr]:
                    trace_path = os.path.join(os.path.dirname(log_path), asm_file.replace('.txt', '.log'))
                    slice_log, rnd_addr, loop_size, start_addr, end_addr = \
                        trace_filter.get_trace(asm_path, prog_path, in_data, trace_path, compiler='tvm', func_type=utils.addr2label[start_addr])
                    func_trace_map[asm_file] = slice_log
                    func_rndaddr_map[asm_file] = (rnd_addr, loop_size, start_addr, end_addr)
                    
    print(func_trace_map)
    print(func_rndaddr_map)
    logger.info('END')
    #exit(0)

    # ==============================================================

    # Step 2.2 Recover Shape with Symbolic Execution
    # Step 2.2.1 Conv and Matmul layers

    # We have to pass the external function address to SE engine
    # This can be done automatically, but we do it manually for simplicity
    se_engine.extern_functions = {'0x400cd0': 'memset', '0x400c20': 'expf'}  # address in .plt, name
    # handle all conv layer. Also, all dense/matmul

    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, func_trace_map, compiler='tvm', topo_list=topo_list)
    print('all conv and dense done.')
    for name, result in func_shape.items():
        print(name)
        for i in range(len(func_meta_data)):
            if func_meta_data[i][0] == name:
                func_meta_data[i][1] = result
                break
        if len(result) == 4:
            print('filter_shape', result[0])
            print('input_shape', result[1])
            print('output_shape', result[2])
            # for O0 binary we do not need layout shape
        else:
            print(result)
    # exit(0)
    
    # ==============================================================
    
    # Step 2.2.2 Other layers
    
    asm_files = os.listdir(utils.funcs_dir)
    se_engine.extern_functions = {'0x400cd0': 'memset', '0x400c20': 'expf'}  # address in .plt, name
    results_dict = dict()
    for asm_file in asm_files:
        if 'labels' not in asm_file and asm_file.endswith('.txt'):
            asm_path = os.path.join(utils.funcs_dir, asm_file)
            start_addr, _ = utils.get_func_range(asm_path)
            if start_addr in utils.addr2label.keys():
                func_type = utils.addr2label[start_addr]
                if 'clip' in func_type or 'bias_add' in func_type or 'add' in func_type or 'avg_pool2d' in func_type:

                    print('\nSE for {}, {}'.format(asm_file, func_type))
                    tmp_log_path = os.path.basename(asm_file)[:-4] + '.log'
                    # gnereate tmp trace file, it should be fast
                    utils.generate_inst_trace(asm_file, tmp_log_path, prog_path, in_data, timeout=True)
                    # symbolic execution, also should be fast
                    utils.generate_symbolic_expression(asm_file, tmp_log_path, exp_log_path, max_inst=5000000)
                    # --- try to interpret the filter shape from symbolic expression log
                    shape = utils.recover_shape_tvm(asm_file, exp_log_path,
                                                mem_read_log_path, mem_write_log_path,
                                                prog_path, in_data, func_type=func_type, is2d=True)
                    print('name', asm_file, 'shape:', shape)
                    results_dict[asm_file] = shape
    for name, result in results_dict.items():
        for i in range(len(func_meta_data)):
            if func_meta_data[i][0] == name:
                if isinstance(result, float):
                    func_meta_data[i][1] = [1, result]
                else:
                    func_meta_data[i][1] = result
                break
        print(name)
        print(result)
    # exit(0)
    list_to_json(topo_list, './topo_list.json')
    dict_to_json(func_meta_data, './meta_data.json')
    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================
    func_meta_data = fuse_batchnorm(topo_list, func_meta_data)
    new_meta_data = []
    logged_func = []
    for i in range(len(func_meta_data)):
        meta_data = func_meta_data[i]
        if meta_data[0] in logged_func:
            continue
        else:
            logged_func.append(meta_data[0])
        if meta_data[3] == 'conv2d':
            meta_data[6] = 1
            meta_data[5] = int(meta_data[1][1][3] / meta_data[1][2][3])
            meta_data[4] = math.ceil((meta_data[1][1][3] - meta_data[1][2][3] * meta_data[5]) / 2)
            new_meta_data.append(meta_data)
        elif meta_data[3] == 'dense' or meta_data[3] == 'bias_add' or meta_data[3] == 'add' or meta_data[3] == 'gamma' or meta_data[3] == 'beta':
            meta_data[6] = 1
            new_meta_data.append(meta_data)
        elif meta_data[3] == 'mean' or meta_data[3] == 'var':
            meta_data[6] = 0
            new_meta_data.append(meta_data)
        
    func_meta_data = new_meta_data
    for meta_data in func_meta_data:
        if meta_data[6]:
            print(meta_data)
    list_to_json(func_meta_data, './new_meta_data.json')

    logged_func = []
    for meta_data in func_meta_data:
        func_name = meta_data[0]
        if func_name not in logged_func:
            logged_func.append(func_name)
        else:
            continue
        w_shape = list(meta_data[1])
        dump_point = meta_data[2]
        func_type = meta_data[3]
        data_index = meta_data[6]
        if func_type == 'conv2d':
            w_shape = w_shape[0]
        for i in range(len(w_shape)):
            if isinstance(w_shape[i], float):
                w_shape[i] = int(w_shape[i])
        
        utils.extract_params_tvm(prog_path, in_data, w_shape, dump_point, mem_dump_log_path, func_name, func_type, data_index)
