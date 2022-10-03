#! /usr/bin/python3
import os
import sys
sys.path.append("../..")
import utils
import trace_filter
from utils import list_to_json, dict_to_json, json_to_list, json_to_dict
import logging
import math
import copy
print('get logger: {}'.format('decompiler.' + __name__))
logger = logging.getLogger('decompiler.' + __name__)

if __name__ == '__main__':
    utils.funcs_dir = "/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/resnet18_glow_ida/"

    prog_path = "/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/resnet18_strip.out"
    in_data = "/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/cat.bin"
    log_path = "/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/func_call.log"
    label_file = "/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/step1.txt"

    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = './mem_read.log'
    mem_write_log_path = './mem_write.log'
    mem_dump_log_path = './mem_dump.log'

    # ==============================================================
    # Step 1 --- Get the Sequence of Layers ---
    # ==============================================================

    utils.get_funcs_trace(prog_path, in_data, log_path, label_file)
    addr2param = utils.print_layer_label(log_path, 'config.json')  # set addr2label in utils
    func_meta_data, topo_list = utils.print_input_id(log_path, compiler='glow', addr2param=addr2param)
    # ==============================================================
    # Step 2 --- Recover the Shape of each Layer
    # ==============================================================

    # Step 2.1 Generate and Filter Trace
    
    func_trace_map = {}
    func_rndaddr_map = {}
    asm_files = os.listdir(utils.funcs_dir)
    for asm_file in asm_files:
        if 'labels' not in asm_file and asm_file.endswith('.txt'):
            asm_path = os.path.join(utils.funcs_dir, asm_file)
            start_addr, _ = utils.get_func_range(asm_path)
            if start_addr in utils.addr2label.keys():
                if 'matmul' in utils.addr2label[start_addr] or 'conv' in utils.addr2label[start_addr]:
                    trace_path = os.path.join(os.path.dirname(log_path), asm_file.replace('.txt', '.log'))
                    slice_log, rnd_addr, loop_size, start_addr, end_addr = \
                        trace_filter.get_trace(asm_path, prog_path, in_data, trace_path, func_type=utils.addr2label[start_addr])
                    func_trace_map[asm_file] = slice_log
                    func_rndaddr_map[asm_file] = (rnd_addr, loop_size, start_addr, end_addr)
    print(func_trace_map)
    print(func_rndaddr_map)
    #exit(0)
    
    # ==============================================================

    # Step 2.2 Recover Shape with Symbolic Execution

    # Step 2.2.0 Choose Another Random Target Address (if needed)
    #func_name = '0030.txt'
    #asm_path = os.path.join(utils.funcs_dir, func_name)
    #slice_log, rnd_addr, loop_size = trace_filter.filt_trace(asm_path, prog_path, in_data, '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0030_rev.log')
    #print(' slice_log {}\n rnd_addr {}\n loop_size {}\n'.format(slice_log, rnd_addr, loop_size))
    #utils.generate_symbolic_expression(func_name, '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0030_slice.log', exp_log_path, max_inst=5000000)
    #exit(0)

    # Step 2.2.1 Conv and Matmul layers
    
    # func_trace_map = {'0013.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0013_slice.log',
    #                   '0016.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0016_slice.log',
    #                   '0029.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0029_slice.log',
    #                   '0020.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0020_slice.log',
    #                   '0017.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0017_slice.log',
    #                   '0032.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0032_slice.log',
    #                   '0022.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0022_slice.log',
    #                   '0028.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0028_slice.log',
    #                   '0023.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0023_slice.log',
    #                   '0026.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0026_slice.log',
    #                   '0030.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0030_slice.log',
    #                   '0024.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0024_slice.log',
    #                   '0010.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0010_slice.log',
    #                   '0018.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0018_slice.log',
    #                   '0012.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0012_slice.log',  # conv
    #
    #                   '0035.txt': '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2020/resnet18_glow/0035_slice.log'  # matmul
    #                   }

    # func_rndaddr_map = {'0013.txt': ('0x3200c10', 64, '0x4029c0', '0x4030f9'),
    #                     '0016.txt': ('0x313b2cc', 64, '0x403320', '0x40382F'),
    #                     '0029.txt': ('0x315cd6c', 512, '0x409300', '0x40B8AE'),
    #                     '0020.txt': ('0x319d0ac', 128, '0x4045f0', '0x404C6B'),
    #                     '0017.txt': ('0x31b0010', 128, '0x403850', '0x403E8C'),
    #                     '0032.txt': ('0x3163440', 512, '0x40e000', '0x410631'),
    #                     '0022.txt': ('0x31ac1d0', 64, '0x404da0', '0x405442'),
    #                     '0028.txt': ('0x313aed8', 512, '0x406d70', '0x4092D8'),
    #                     '0023.txt': ('0x31edc1c', 256, '0x405450', '0x405BAC'),
    #                     '0026.txt': ('0x313b14c', 256, '0x4064b0', '0x406C32'),
    #                     '0030.txt': ('0x31703b4', 512, '0x40b8d0', '0x40DED4'),
    #                     '0024.txt': ('0x3162358', 256, '0x405bd0', '0x40637C'),
    #                     '0010.txt': ('0x31d24c0', 64, '0x401850', '0x401F4A'),
    #                     '0018.txt': ('0x323d73c', 64, '0x403eb0', '0x4044dc'),
    #                     '0012.txt': ('0x313c544', 64, '0x402200', '0x40299B'),  # conv
    #
    #                     '0035.txt': ('0x313ad58', 64, '0x410b60', '0x41118d')  # matmul
    #                     }


    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, func_trace_map, compiler='glow') # also matmul layer
    print('all conv and matmul done.')
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
            print('with relu', result[3])
        else:
            print(result)
    #exit(0)
    

    # Step 2.2.2 Other layers
    results_dict = dict()
    asm_files = os.listdir(utils.funcs_dir)
    for asm_file in asm_files:
        if 'labels' not in asm_file and asm_file.endswith('.txt'):
            asm_path = os.path.join(utils.funcs_dir, asm_file)
            start_addr, _ = utils.get_func_range(asm_path)
            if start_addr in utils.addr2label.keys():
                func_type = utils.addr2label[start_addr]
                if 'matmul' not in func_type and 'conv' not in func_type and \
                   'trans' not in func_type and 'add, relu' not in func_type:  # transpose could be ignored, (add, relu) layer is known
                    print('SE for {}, {}'.format(asm_file, func_type))
                    tmp_log_path = os.path.basename(asm_file)[:-4] + '.log'
                    # gnereate tmp trace file, it should be fast
                    utils.generate_inst_trace(asm_file, tmp_log_path, prog_path, in_data, timeout=True)
                    # symbolic execution, also should be fast
                    utils.generate_symbolic_expression(asm_file, tmp_log_path, exp_log_path, max_inst=5000000)
                    # --- try to interpret the filter shape from symbolic expression log
                    shape = utils.recover_shape(asm_file, exp_log_path,
                                                mem_read_log_path, mem_write_log_path,
                                                prog_path, in_data, func_type=func_type)
                    print('shape:', shape)
                    results_dict[asm_file] = shape
    for name, result in results_dict.items():
        for i in range(len(func_meta_data)):
            if func_meta_data[i][0] == name:
                if isinstance(result, float):
                    func_meta_data[i][1] = [1, int(result)]
                else:
                    func_meta_data[i][1] = result
                break
        print(name)
        print(result)
    #exit(0)

    list_to_json(topo_list, './topo_list.json')
    dict_to_json(func_meta_data, './meta_data.json')
    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================
    # with all above information, it is feasible to extract paramters

    func_meta_data = list(func_meta_data.values())
    new_meta_data = []
    logged_func = []
    for i in range(len(func_meta_data)):
        meta_data = func_meta_data[i]
        if meta_data[0] in logged_func:
            continue
        else:
            logged_func.append(meta_data[0])

        # identify the type of conv
        func_name = meta_data[0]

        if 'conv' in meta_data[3]:
            meta_data[6] = 1
            meta_data[5] = int(meta_data[1][1][3] / meta_data[1][2][3])  # stride
            meta_data[4] = math.ceil((meta_data[1][1][3] - meta_data[1][2][3] * meta_data[5]) / 2)  # padding
            new_meta_data.append(meta_data)  # weights of conv
            meta_data = copy.deepcopy(meta_data)
            meta_data[6] = 2
            meta_data[5] = meta_data[4] = None
            meta_data[3] = 'add'
            meta_data[1] = [1, int(meta_data[1][0][0])]
            new_meta_data.append(meta_data)  # biases of conv
        elif 'matmul' in meta_data[3]:
            meta_data[6] = 1
            new_meta_data.append(meta_data)  # weights of dense
        elif 'batchedadd' in meta_data[3]:
            meta_data[6] = 1
            meta_data[1] = [1, int(meta_data[1][1])]
            new_meta_data.append(meta_data)  # biases of dense
        else:
            new_meta_data.append(meta_data)

    # print for debug
    func_meta_data = new_meta_data
    for meta_data in func_meta_data:
        if meta_data[6]:  # has parameters to be extracted
            print(meta_data)
    dict_to_json(func_meta_data, './new_meta_data.json')
    # (name, shape, fused_func, type, padding, stride, param_index)
    # func_meta_data = [('0010.txt', (64, 3, 7, 7), '0x401850', 'conv2d'),
    #                   ('0010. txt', (1, 64), '0x401850', 'add'),
    #                   ('0012.txt', (64, 64, 3, 3), '0x402200', 'conv2d'),
    #                   ('0012.txt', (1, 64), '0x402200', 'add'),
    #                   ('0013.txt', (64, 64, 3, 3), '0x4029c0', 'convDKKC8'),
    #                   ('0013.txt', (1, 64), '0x4029c0', 'add'),
    #                   ('0016.txt', (128, 64, 1, 1), '0x403320', 'convDKKC8'),
    #                   ('0016.txt', (1, 128), '0x403320', 'add'),
    #                   ('0017.txt', (128, 64, 3, 3), '0x403850', 'conv2d'),
    #                   ('0017.txt', (1, 128), '0x403850', 'add'),
    #                   ('0018.txt', (128, 128, 3, 3), '0x403eb0', 'convDKKC8'),
    #                   ('0018.txt', (1, 128), '0x403eb0', 'add'),
    #                   ('0020.txt', (128, 128, 3, 3), '0x4045f0', 'conv2d'),
    #                   ('0020.txt', (1, 128), '0x4045f0', 'add'),
    #                   ('0022.txt', (256, 128, 1, 1), '0x404da0', 'convDKKC8'),
    #                   ('0022.txt', (1, 256), '0x404da0', 'add'),
    #                   ('0023.txt', (256, 128, 3, 3), '0x405450', 'conv2d'),
    #                   ('0023.txt', (1, 256), '0x405450', 'add'),
    #                   ('0024.txt', (256, 256, 3, 3), '0x405bd0', 'convDKKC8'),
    #                   ('0024.txt', (1, 256), '0x405bd0', 'add'),
    #                   ('0026.txt', (256, 256, 3, 3), '0x4064b0', 'conv2d'),
    #                   ('0026.txt', (1, 256), '0x4064b0', 'add'),
    #                   ('0028.txt', (512, 256, 1, 1), '0x406d70', 'convDKKC8'),
    #                   ('0028.txt', (1, 512), '0x406d70', 'add'),
    #                   ('0029.txt', (512, 256, 3, 3), '0x409300', 'conv2d'),
    #                   ('0029.txt', (1, 512), '0x409300', 'add'),
    #                   ('0030.txt', (512, 512, 3, 3), '0x40b8d0', 'convDKKC8'),
    #                   ('0030.txt', (1, 512), '0x40b8d0', 'add'),
    #                   ('0032.txt', (512, 512, 3, 3), '0x40e000', 'conv2d'),
    #                   ('0032.txt', (1, 512), '0x40e000', 'add'),
    #
    #                   ('0035.txt', (512, 1000), '0x410b60', 'matmul'),
    #                   ('0036.txt', (1, 1000), '0x411190', 'batchedadd'),
    #
    #                   ]
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        func_type = fun_data[3]
        data_index = fun_data[-1]
        if not data_index:  # no data to extract
            continue
        if 'conv' in func_type:
            w_shape = w_shape[0]
        logger.info('Extract Parameter for {}'.format(func_name))
        if 'conv' in func_type and 'DKKC8' not in func_type:
            w_shape = (w_shape[0], w_shape[2], w_shape[3], w_shape[1])
            w_shape = [int(w_shape[i]) for i in range(len(w_shape))]
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, data_index, func_type)
        elif 'conv' in func_type and 'DKKC8' in func_type:
            w_shape = (int(w_shape[0]/8), w_shape[2], w_shape[3], w_shape[1], 8)
            print(w_shape)
            w_shape = [int(w_shape[i]) for i in range(len(w_shape))]
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, data_index, func_type)
        elif 'matmul' in func_type:
            w_shape = [w_shape[1], w_shape[0]]
            w_shape = [int(w_shape[i]) for i in range(len(w_shape))]
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, data_index, func_type)
        else:
            w_shape = [int(w_shape[i]) for i in range(len(w_shape))]
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, data_index, func_type)
