#! /usr/bin/python3
import os
import sys
sys.path.append("../")
import math
import utils
from utils import list_to_json, dict_to_json, json_to_list, json_to_dict
import pin_tools


def get_embedding_size(memcpy_plt: str, log_path: str):  # 0x401050 -- embedding is implemented with memcpy
    log_path = os.path.abspath(log_path)
    addr_list = [memcpy_plt]
    pin_tools.fun_call_rdi_rsi(prog_path, in_data, addr_list, log_path)
    with open(log_path, 'r') as f:
        lines = f.readlines()
        line = lines[0]
        size = line[line.find('rdx'):].split(' ')[1].strip()
        size = size.strip(', ')
        size = int(size, 16) / 4
        src_list = []
        for line in lines:
            if not line.startswith('#'):
                src_addr = line[line.find('rsi')+3:].split(',')[0].strip()
                src_addr = int(src_addr, 16)
                src_list.append(src_addr)
    return size, src_list


def get_embedding_region(entry_func: str, next_func: str, src_list: list):
    entry_path = os.path.join(utils.funcs_dir, entry_func)
    next_func_path = os.path.join(utils.funcs_dir, next_func)

    start_addr, end_addr = utils.get_func_range(entry_path)
    addr_list = [start_addr]
    pin_tools.fun_call_rdi_rsi(prog_path, in_data, addr_list, log_path)
    with open(log_path, 'r') as f:
        lines = f.readlines()
        line = lines[0]
        region_start = line[line.find('rdi')+3:].split(',')[0].strip()
        region_start = int(region_start, 16)

    start_addr, end_addr = utils.get_func_range(next_func_path)
    utils.mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, in_data)
    read_mem_regions = utils.memory_slices(mem_read_log_path)
    read_mem_regions = sorted(read_mem_regions, key=lambda x: x[0])
    for mem_blk in read_mem_regions:
        embedding_flag = False
        for src_addr in src_list:
            if mem_blk[0] <= src_addr <= mem_blk[1]:
                embedding_flag = True
                break
        if not embedding_flag:
            region_end = mem_blk[0]
            break
    return region_start, region_end


if __name__ == '__main__':
    utils.funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/embedding/embedding_glow_funcs/"

    # prepared in advance
    prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/embedding/embedding_glow"
    in_data = ''  # no input needed, hard coded in source code
    label_file = './step1.txt'
    config_file = './config.json'

    if len(sys.argv) == 4:
        utils.funcs_dir = sys.argv[1]
        prog_path = sys.argv[2]
        label_file = sys.argv[3]

    # (tmp files) generated during analysis
    log_path = './embedding_glow_func_call.log'
    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = './mem_read.log'
    mem_write_log_path = './mem_write.log'
    mem_dump_log_path = './mem_dump.log'

    # ===============================
    # Recover the Computational Graph
    # ===============================
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file)
    addr2param = utils.print_layer_label(log_path, config_file)
    func_meta_data, topo_list = utils.print_input_id(log_path, compiler='glow', addr2param=addr2param)
    
    # ===============================
    # Recover the embedding layer, which does not have a specific function call
    # ===============================
    size, src_list = get_embedding_size('0x401050', log_path)  # we can get the address of memcpy
    emb_start, emb_end = get_embedding_region('0008.embedding_2.txt', '0010.libjit_matmul_f.txt', src_list)  # we can also know next layer with parameters (from funcs call trace)
    print('embedding dict: start {}, end {}, vector size {}, vocabulary size {}'.
          format(hex(emb_start), hex(emb_end), size, math.floor((emb_end-emb_start)/4/size)))

    # ===============================
    # Recover other layers
    # ===============================
    func_data = {
                 'avg_pool': '0009.libjit_avg_pool_f.txt',
                 'matmul': '0010.libjit_matmul_f.txt',
                 # 'add': '0011.libjit_stacked_kernel' # no need
                 }

    for func_type, func_name in func_data.items():
        utils.generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)

        utils.generate_symbolic_expression(func_name, tmp_log_path, exp_log_path, max_inst=5000000)

        shape = utils.recover_shape(func_name, exp_log_path,
                                    mem_read_log_path, mem_write_log_path,
                                    prog_path, in_data, func_type=func_type)
        for i in range(len(func_meta_data)):
            if func_meta_data[i][0] == func_name:
                func_meta_data[i][1] = shape
                break
        print(shape)
    
    list_to_json(topo_list, './topo_list.json')
    dict_to_json(func_meta_data, './meta_data.json')
    # ===============================
    # Extract Parameters
    # ===============================
    # func_meta_data is collected from previous output
    func_meta_data = [
                      ('0008.embedding_2.txt', (25002, 100), '0x401270', '0x404080', 'embedding'),  # embedding need to be manually set
                      ('0010.libjit_matmul_f.txt', (1, 100), '0x401610', '0xd8da80', 'matmul'),
                      ('0011.libjit_stacked_kernel.txt', (1, 1), '0x401690', '0xd8da40', 'add'),
                      ]
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        dump_addr = fun_data[3]
        func_type = fun_data[4]
        utils.extract_params_general(prog_path, in_data, w_shape, dump_point,
                                     mem_dump_log_path, func_name, dump_addr)
