#! /usr/bin/python3
import os
import sys
import json
import math
sys.path.append("../..")
import trace_filter
import utils
import se_engine
from utils import list_to_json, dict_to_json, json_to_dict, json_to_list
import logging
from fused_trace import fuse_batchnorm
print('get logger: {}'.format('decompiler.'+__name__))
logger = logging.getLogger('decompiler.'+__name__)


if __name__ == '__main__':
    utils.funcs_dir = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/resnet18_funcs/"
    prog_path = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/resnet18_tvm_O0_strip"
    in_data = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/cat.bin"
    log_path = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/func_call.log"
    label_file = "/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/label.txt"

    if len(sys.argv) == 6:
        utils.funcs_dir = sys.argv[1]
        prog_path = sys.argv[2]
        in_data = sys.argv[3]
        log_path = sys.argv[4]
        label_file = sys.argv[5]

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
    
    """ # to be removed
    log_path = './resnet18_strip_func_call_fused.log'
    get_funcs_trace(prog_path, in_data, log_path, label_file, only_fused=True)
    new_log_path = './resnet18_strip_func_call_fused_2.log'
    print_fused_trace(log_path, new_log_path)
    call_graph_list = get_call_graph_list('./resnet18_strip_func_call_fused_3.log')
    show_graph(call_graph_list)
    # print_layer_label(log_path)
    """

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
                    
    # print(func_trace_map)
    # print(func_rndaddr_map)
    logger.info('END')
    #exit(0)

    # ==============================================================

    # Step 2.2 Recover Shape with Symbolic Execution
    # Step 2.2.1 Conv and Matmul layers

    # generated in previous stage, can be removed, just for debugging
    # func_trace_map = {'0170.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0170_slice.log',
    #                   '0160.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0160_slice.log',
    #                   '0153.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0153_slice.log',
    #                   '0142.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0142_slice.log',
    #                   '0122.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0122_slice.log',
    #                   '0115.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0115_slice.log',
    #                   '0090.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0090_slice.log',
    #                   '0063.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0063_slice.log',
    #                   '0059.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0059_slice.log',
    #                   '0051.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0051_slice.log',
    #                   '0028.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0028_slice.log',  # conv
    # 
    #                   '0092.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0092_slice.log',  # dense
    #                   }
    # not used
    # func_rndaddr_map = {'0170.txt': ('0x46917c8', 64, '0x432d50', '0x434875'),
    #                     '0160.txt': ('0x377d7c8', 64, '0x42ec00', '0x430F8F'),
    #                     '0153.txt': ('0x377d82c', 64, '0x42aec0', '0x42D6D0'),
    #                     '0142.txt': ('0x46acbd4', 64, '0x4262e0', '0x428869'),
    #                     '0122.txt': ('0x377d7d4', 64, '0x420600', '0x422F30'),
    #                     '0115.txt': ('0x377d7d4', 64, '0x41c230', '0x41EB80'),
    #                     '0090.txt': ('0x46acbd8', 64, '0x415810', '0x417E09'),
    #                     '0063.txt': ('0x3469024', 64, '0x40ff40', '0x411274'),
    #                     '0059.txt': ('0x3468fc8', 64, '0x40d190', '0x40EB4B'),
    #                     '0051.txt': ('0x377d850', 64, '0x409280', '0x40BBC0'),
    #                     '0028.txt': ('0x346913c', 64, '0x402ea0', '0x404FB7'),  # conv
    #
    #                     '0092.txt': ('0x377bbc0', 64, '0x4181a0', '0x4185f1'),  # dense
    #                     }
    # Sometimes we may want to re-log/symex/analyze the trace for specific operator
    #se_engine.extern_functions = {'0x400c10': 'memset'}
    # utils.generate_inst_trace('0153.txt', tmp_log_path, prog_path, in_data, timeout=True)
    #utils.generate_symbolic_expression('0153.txt', '/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.7/resnet18_tvm_O0/0153_slice.log', exp_log_path)
    #shape = utils.recover_shape_tvm('0153.txt', exp_log_path, mem_read_log_path, mem_write_log_path, prog_path, in_data, func_type='conv2d', optimized=True)
    #print(shape)
    #exit(0)

    # We have to pass the external function address to SE engine
    # This can be done automatically, but we do it manually for simplicity
    se_engine.extern_functions = {'0x400c10': 'memset'}  # address in .plt, name
    # handle all conv layer. Also, all dense/matmul
    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, func_trace_map, compiler='tvm')
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
    #exit(0)
    
    # ==============================================================
    
    # Step 2.2.2 Other layers
    # the BatchNorm2d is implemented with a special sequence
    # [add, sqrt, divide, multiply, expand_dims, multiply, negative, multiply, add, expand_dims, add]
    # For this special sequence, we try to merge it back into BatchNorm in fused_trace.py
    
    asm_files = os.listdir(utils.funcs_dir)
    se_engine.extern_functions = {'0x400c10': 'memset'}  # address in .plt, name
    results_dict = dict()
    for asm_file in asm_files:
        if 'labels' not in asm_file and asm_file.endswith('.txt'):
            asm_path = os.path.join(utils.funcs_dir, asm_file)
            start_addr, _ = utils.get_func_range(asm_path)
            if start_addr in utils.addr2label.keys():
                func_type = utils.addr2label[start_addr]
                if 'pool' in func_type or 'bias_add' in func_type or 'add' in func_type:
                    # transpose, expand_dims and relu could be ignored, batchnormalization always follow after a conv layer
                    print('SE for {}, {}'.format(asm_file, func_type))
                    tmp_log_path = os.path.basename(asm_file)[:-4] + '.log'
                    # gnereate tmp trace file, it should be fast
                    utils.generate_inst_trace(asm_file, tmp_log_path, prog_path, in_data, timeout=True)
                    # symbolic execution, also should be fast
                    utils.generate_symbolic_expression(asm_file, tmp_log_path, exp_log_path, max_inst=5000000)
                    # --- try to interpret the filter shape from symbolic expression log
                    shape = utils.recover_shape_tvm(asm_file, exp_log_path,
                                                mem_read_log_path, mem_write_log_path,
                                                prog_path, in_data, func_type=func_type)
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
            meta_data[4] = math.ceil((meta_data[1][1][3] - meta_data[1][2][3]*meta_data[5]) / 2)
            new_meta_data.append(meta_data)
        elif meta_data[3] == 'dense' or meta_data[3] == 'bias_add' or meta_data[3] == 'gamma' or meta_data[3] == 'beta':
            meta_data[6] = 1
            new_meta_data.append(meta_data)
        elif meta_data[3] == 'mean' or meta_data[3] == 'var':
            meta_data[6] = 0
            new_meta_data.append(meta_data)
    func_meta_data = new_meta_data
    for meta_data in func_meta_data:
        if meta_data[0] == '0153.txt':
            new_shape = list(meta_data[1])
            new_shape[0] = [128, 64, 3, 3]
            meta_data[1] = tuple(new_shape)  # manually fix the wrong shape, can be inferred from previous/succeed op
            meta_data[4] = 1
            meta_data[5] = 2
            meta_data[6] = 1
        if meta_data[6]:
            print(meta_data)
    dict_to_json(func_meta_data, './new_meta_data.json')
    # (name, shape, fused_func, type, padding, stride, param_index)
    # func_meta_data = [('0028.txt', (64, 3, 7, 7), '0x4022b0', 'conv2d', 3, 2, 1),
    #                   ('0051.txt', (64, 64, 3, 3), '0x408330', 'conv2d', 1, 1, 1),
    #                   ('0059.txt', (256, 128, 1, 1), '0x40c450', 'conv2d', 0, 2, 1),
    #                   ('0063.txt', (128, 64, 1, 1), '0x40eb70', 'conv2d', 0, 2, 1),
    #                   ('0090.txt', (512, 512, 3, 3), '0x414820', 'conv2d', 1, 1, 1),
    #                   ('0115.txt', (128, 128, 3, 3), '0x41b190', 'conv2d', 1, 1, 1),
    #                   ('0122.txt', (256, 128, 3, 3), '0x41ef20', 'conv2d', 1, 2, 1),
    #                   ('0142.txt', (512, 256, 3, 3), '0x425450', 'conv2d', 1, 2, 1),
    #                   # ('0153.txt', (*, 16, 6, 6), '0x429b40', 'conv2d', 1, 2, 1),  # <-- wrong shape
    #                   ('0153.txt', (128, 64, 3, 3), '0x429b40', 'conv2d', 1, 2, 1),
    #                   ('0160.txt', (256, 256, 3, 3), '0x42db40', 'conv2d', 1, 1, 1),
    #                   ('0170.txt', (512, 256, 1, 1), '0x431950', 'conv2d', 0, 2, 1),
    #
    #                   ('0092.txt', (1000, 512), '0x417e30', 'dense', 0, 0, 1),
    #                   ('0155.txt', (1, 1000), '0x42d700', 'bias_add', 0, 0, 1),  # bias add
    #
    #                   ('0083.txt', (1, 512), '0x4140c0', 'var', 0, 0, 0),  # norm var - used in add
    #                   ('0085.txt', (1, 512), '0x414440', 'gamma', 0, 0, 1),  # norm gamma - used in multiply
    #                   ('0129.txt', (1, 512), '0x423e00', 'mean', 0, 0, 0),  # norm mean - used in negative
    #                   ('0096.txt', (1, 512), '0x418a00', 'beta', 0, 0, 1),  # norm beta - used in add
    #
    #                   ('0100.txt', (1, 256), '0x419380', 'var', 0, 0, 0),  # norm var - used in add
    #                   ('0081.txt', (1, 256), '0x413ce0', 'gamma', 0, 0, 1),  # norm gamma - used in multiply
    #                   ('0067.txt', (1, 256), '0x411680', 'mean', 0, 0, 0),  # norm mean - used in negative
    #                   ('0065.txt', (1, 256), '0x4112a0', 'beta', 0, 0, 1),  # norm beta - used in add
    #
    #                   ('0172.txt', (1, 128), '0x4348a0', 'var', 0, 0, 0),  # norm var - used in add
    #                   ('0131.txt', (1, 128), '0x4240d0', 'gamma', 0, 0, 1),  # norm gamma - used in multiply
    #                   ('0023.txt', (1, 128), '0x401fe0', 'mean', 0, 0, 0),  # norm mean - used in negative
    #                   ('0166.txt', (1, 128), '0x431570', 'beta', 0, 0, 1),  # norm beta - used in add
    #
    #                   ('0117.txt', (1, 64), '0x41eba0', 'var', 0, 0, 0),  # norm var - used in add
    #                   ('0176.txt', (1, 64), '0x435180', 'gamma', 0, 0, 1),  # norm gamma - used in multiply
    #                   ('0162.txt', (1, 64), '0x430fb0', 'mean', 0, 0, 0),  # norm mean - used in negative
    #                   ('0077.txt', (1, 64), '0x413290', 'beta', 0, 0, 1),  # norm beta - used in add
    #
    #                   ]

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
        if len(w_shape) == 2 and w_shape[1] == 257:
            w_shape[1] = 256
        utils.extract_params_tvm(prog_path, in_data, w_shape, dump_point, mem_dump_log_path, func_name, func_type, data_index)
