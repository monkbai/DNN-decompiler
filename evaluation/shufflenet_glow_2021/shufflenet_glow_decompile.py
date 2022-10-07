#! /usr/bin/python3
import os
import sys
sys.path.append("../..")
import utils
import trace_filter
from utils import list_to_json, dict_to_json, json_to_list, json_to_dict
import se_engine
import logging
import math
import copy
print('get logger: {}'.format('decompiler.' + __name__))
logger = logging.getLogger('decompiler.' + __name__)

if __name__ == '__main__':
    utils.funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2021/shufflenet_v2/shufflenet_funcs"

    prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2021/shufflenet_v2/shufflenet_v2_strip.out"
    in_data = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2021/shufflenet_v2/cat.bin"
    log_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2021/shufflenet_v2/func_call.log"
    label_file = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2021/shufflenet_v2/label.txt"

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
    mem_dump_log_path = './mem_dump.log'

    # ==============================================================
    # Step 1 --- Get the Sequence of Layers ---
    # ==============================================================

    utils.get_funcs_trace(prog_path, in_data, log_path, label_file)
    addr2param = utils.print_layer_label(log_path, 'config.json')  # set addr2label in utils
    func_meta_data, topo_list = utils.print_input_id(log_path, compiler='glow', addr2param=addr2param)

    list_to_json(topo_list, './topo_list.json')

    # ==============================================================
    # Step 2 --- Recover the Shape of each Layer
    # ==============================================================

    # Step 2.1 Generate and Filter Trace
    # Warning: 0051.txt may need to pick rand target addr several times
    trace_filter.all_trace_list = json_to_list('./all_trace_list.json')
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
                        trace_filter.get_trace(asm_path, prog_path, in_data, trace_path,
                                               func_type=utils.addr2label[start_addr])
                    func_trace_map[asm_file] = slice_log
                    func_rndaddr_map[asm_file] = (rnd_addr, loop_size, start_addr, end_addr)
    print(func_trace_map)
    print(func_rndaddr_map)
    # exit(0)

    # ==============================================================

    # Step 2.2 Recover Shape with Symbolic Execution

    # Step 2.2.0 Choose Another Random Target Address (if needed)
    # func_name = '0030.txt'
    # asm_path = os.path.join(utils.funcs_dir, func_name)
    # slice_log, rnd_addr, loop_size = trace_filter.filt_trace(asm_path, prog_path, in_data, '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2022/resnet18_glow/0030_rev.log')
    # print(' slice_log {}\n rnd_addr {}\n loop_size {}\n'.format(slice_log, rnd_addr, loop_size))
    # utils.generate_symbolic_expression(func_name, '/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2022/resnet18_glow/0030_slice.log', exp_log_path, max_inst=5000000)
    # exit(0)

    # Step 2.2.1 Conv and Matmul layers
    se_engine.extern_functions = {}  # no external function used
    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, func_trace_map,
                                       compiler='glow')  # also matmul layer
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
    # exit(0)

    # Step 2.2.2 Other layers
    se_engine.extern_functions = {}
    results_dict = dict()
    asm_files = os.listdir(utils.funcs_dir)
    for asm_file in asm_files:
        if 'labels' not in asm_file and asm_file.endswith('.txt'):
            asm_path = os.path.join(utils.funcs_dir, asm_file)
            start_addr, _ = utils.get_func_range(asm_path)
            if start_addr in utils.addr2label.keys():
                # get func type and func_info
                func_type = utils.addr2label[start_addr]
                func_info = []
                if len(topo_list) > 0:
                    for node in topo_list:
                        if node[1] == asm_file:
                            func_info = node
                            break

                if 'matmul' not in func_type and 'conv' not in func_type and \
                        'relu' not in func_type and \
                        'softmax' not in func_type:
                    # softmax has no parameter
                    if 'insert_tensor' in func_type:  # there are two type of insert_tensor
                        ins_ten_fixed_flag = utils.identify_fixed_insert_tensor(asm_path)
                        if not ins_ten_fixed_flag:
                            func_type += '_param'
                            utils.addr2label[start_addr] = func_type
                            for node in topo_list:
                                if node[1] == asm_file:
                                    node[2] = func_type
                    print('SE for {}, {}'.format(asm_file, func_type))

                    tmp_log_path = os.path.basename(asm_file)[:-4] + '.log'
                    # gnereate tmp trace file, it should be fast
                    utils.generate_inst_trace(asm_file, tmp_log_path, prog_path, in_data, timeout=True)
                    # symbolic execution, also should be fast
                    if 'transpose' not in func_type:
                        utils.generate_symbolic_expression(asm_file, tmp_log_path, exp_log_path, max_inst=5000000)
                    # --- try to interpret the filter shape from symbolic expression log
                    shape = utils.recover_shape(asm_file, exp_log_path,
                                                mem_read_log_path, mem_write_log_path,
                                                prog_path, in_data, func_type=func_type, func_info=func_info, is2d=True)
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
    # exit(0)

    insert_tensor_list = utils.extract_inserttensor_offset_glow(prog_path, in_data, './inserttensor_param.log',
                                                                topo_list)
    print(insert_tensor_list)

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
            meta_data = copy.deepcopy(meta_data)
            meta_data[6] = 2
            meta_data[5] = meta_data[4] = None
            meta_data[3] = 'add'
            meta_data[1] = [1, int(meta_data[1][0])]
            new_meta_data.append(meta_data)  # biases of conv
        else:
            new_meta_data.append(meta_data)

    # print for debug
    func_meta_data = new_meta_data
    for meta_data in func_meta_data:
        if meta_data[6]:  # has parameters to be extracted
            print(meta_data)
    dict_to_json(func_meta_data, './new_meta_data.json')

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
            w_shape = (int(w_shape[0] / 8), w_shape[2], w_shape[3], w_shape[1], 8)
            print(w_shape)
            w_shape = [int(w_shape[i]) for i in range(len(w_shape))]
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, data_index, func_type)
        elif 'matmul' in func_type:
            w_shape = [w_shape[1], w_shape[0]]
            w_shape = [int(w_shape[i]) for i in range(len(w_shape))]
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, data_index, func_type)
        else:  # 'add' type, biases
            w_shape = [int(w_shape[i]) for i in range(len(w_shape))]
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, data_index, func_type)
