#! /usr/bin/python3
import os
import sys
sys.path.append("../..")
from scripts import utils
from scripts import trace_filter

if __name__ == '__main__':
    utils.funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_glow/vgg16_glow_ida/"
    prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_glow/vgg16_strip.out"
    in_data = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_glow/cat.bin"
    log_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_glow/func_call.log"
    label_file = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_glow/step1.txt"

    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = './mem_read.log'
    mem_write_log_path = './mem_write.log'

    # compile_all_tools()
    # ==============================================================
    # Step 1 --- Get the Sequence of Layers ---
    # ==============================================================

    utils.get_funcs_trace(prog_path, in_data, log_path, label_file)
    utils.print_layer_label(log_path, 'config.json')

    # ==============================================================
    # Step 2 --- Recover the Shape of each Layer
    # ==============================================================

    # Generate and Filter Trace
    func_trace_map = {}
    func_rndaddr_map = {}
    asm_files = os.listdir(utils.funcs_dir)
    for asm_file in asm_files:
        if 'labels' not in asm_file and asm_file.endswith('.txt'):
            asm_path = os.path.join(utils.funcs_dir, asm_file)
            start_addr, _ = utils.get_func_range(asm_path)
            if start_addr in utils.addr2label.keys():
                if 'conv' in utils.addr2label[start_addr]:
                    trace_path = os.path.join(os.path.dirname(log_path), asm_file.replace('.txt', '.log'))
                    slice_log, rnd_addr, loop_size, start_addr, end_addr = \
                        trace_filter.get_trace(asm_path, prog_path, in_data, trace_path)
                    func_trace_map[asm_path] = slice_log
                    func_rndaddr_map[asm_path] = (rnd_addr, loop_size, start_addr, end_addr)
    print(func_trace_map)
    print(func_rndaddr_map)
    exit(0)

    """
    func_addrs = utils.find_rand_addr(label_file)
    print('start_addr, end_addr, early_stop, loop_size, rand_addr')
    for name, result in func_addrs.items():
        print(name)
        print(result)
    exit(0)
    """
    """
    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, func_trace_map)
    for name, result in func_shape.items():
        print(name)
        print(result)
    exit(0)
    """
    """
    # convolution layers
    #func_type = 'conv2d'
    #func_name = '0017.libjit_conv2d_f.txt'
    #tmp_log_path = './0x4017b0-0x401e91_slice.log'
    #func_name = '0018.libjit_conv2d_f.txt'
    #tmp_log_path = './0x401ea0-0x40265c_slice.log'
    #func_name = '0020.libjit_conv2d_f.txt'
    #tmp_log_path = './0x402990-0x403019_slice.log'
    #func_name = '0021.libjit_conv2d_f.txt'
    #tmp_log_path = './0x403020-0x4036a9_slice.log'
    #func_name = '0023.libjit_conv2d_f.txt'
    #tmp_log_path = './0x403900-0x404089_slice.log'
    #func_name = '0024.libjit_conv2d_f.txt'
    #tmp_log_path = './0x404090-0x404819_slice.log'
    #func_name = '0026.libjit_conv2d_f.txt'
    #tmp_log_path = './0x404a70-0x405409_slice.log'
    #func_name = '0027.libjit_conv2d_f.txt'
    #tmp_log_path = './0x405410-0x405da9_slice.log'
    #func_name = '0029.libjit_conv2d_f.txt'
    #tmp_log_path = './0x406000-0x4069a0_slice.log'
    # dense/fully-connected layers
    func_type = 'dense'
    #func_name = '0031.libjit_matmul_f.txt'  # 4096, 25088
    #tmp_log_path = './0x406c00-0x4071f6_slice.log'
    #func_name = '0034.libjit_matmul_f.txt'  # 4096, 4096
    #tmp_log_path = './0x407490-0x407a83_slice.log'
    func_name = '0036.libjit_matmul_f.txt'  # 1000, 4096
    tmp_log_path = './0x407b50-0x40817d_slice.log'
    # bias add layer
    #func_type = 'add'
    #func_name = '0032.libjit_batchedadd_f.txt'  # 4096
    #func_name = '0037.libjit_batchedadd_f.txt'  # 1000
    # max-poll layers
    #func_type = 'max-pool2d'
    #func_name = '0019.libjit_max_pool_f.txt'  # max 2,2
    #func_name = '0022.libjit_max_pool_f.txt'  # max 2,2
    #func_name = '0025.libjit_max_pool_f.txt'  # max 2,2
    #func_name = '0028.libjit_max_pool_f.txt'  # max 2,2
    #func_name = '0030.libjit_max_pool_f.txt'  # max 2,2

    # tmp_log_path = './inst_trace.log'  # move to begin
    if 'conv2d' not in func_type and 'dense' not in func_type:  # computing-intensive layer need extra preprocessing
        utils.generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)
    # exp_log_path = './mem_exp.log'  # move to begin
    utils.generate_symbolic_expression(func_name, tmp_log_path, exp_log_path, max_inst=5000000)

    # --- try to interpret the filter shape from symbolic expression log
    # mem_read_log_path = 'mem_read.log'  # move to begin
    # mem_write_log_path = 'mem_write.log'  # move to begin
    shape = utils.recover_shape(func_name, exp_log_path,
                                mem_read_log_path, mem_write_log_path,
                                prog_path, in_data, func_type=func_type)
    print(shape)
    exit(0)
    """
    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================
    mem_dump_log_path = './vgg16_glow_O0/mem_dump.log'
    func_meta_data = [('0017.libjit_conv2d_f.txt', (64, 3, 3, 3), '0x4017b0', 'conv2d'),
                      ('0017.libjit_conv2d_f.txt', (1, 64), '0x4017b0', 'add'),
                      ('0018.libjit_conv2d_f.txt', (64, 64, 3, 3), '0x401ea0', 'conv2d'),
                      ('0018.libjit_conv2d_f.txt', (1, 64), '0x401ea0', 'add'),
                      ('0020.libjit_conv2d_f.txt', (128, 64, 3, 3), '0x402990', 'conv2d'),
                      ('0020.libjit_conv2d_f.txt', (1, 128), '0x402990', 'add'),
                      ('0021.libjit_conv2d_f.txt', (128, 128, 3, 3), '0x403020', 'conv2d'),
                      ('0021.libjit_conv2d_f.txt', (1, 128), '0x403020', 'add'),
                      ('0023.libjit_conv2d_f.txt', (256, 128, 3, 3), '0x403900', 'conv2d'),
                      ('0023.libjit_conv2d_f.txt', (1, 256), '0x403900', 'add'),
                      ('0024.libjit_conv2d_f.txt', (256, 256, 3, 3), '0x404090', 'conv2d'),
                      ('0024.libjit_conv2d_f.txt', (1, 256), '0x404090', 'add'),
                      ('0026.libjit_conv2d_f.txt', (512, 256, 3, 3), '0x404a70', 'conv2d'),
                      ('0026.libjit_conv2d_f.txt', (1, 512), '0x404a70', 'add'),
                      ('0027.libjit_conv2d_f.txt', (512, 512, 3, 3), '0x405410', 'conv2d'),
                      ('0027.libjit_conv2d_f.txt', (1, 512), '0x405410', 'add'),
                      ('0029.libjit_conv2d_f.txt', (512, 512, 3, 3), '0x406000', 'conv2d'),
                      ('0029.libjit_conv2d_f.txt', (1, 512), '0x406000', 'add'),

                      ('0031.libjit_matmul_f.txt', (7, 7, 512, 4096), '0x406c00', 'dense'),
                      ('0034.libjit_matmul_f.txt', (4096, 4096), '0x407490', 'dense'),
                      ('0036.libjit_matmul_f.txt', (4096, 1000), '0x407b50', 'dense'),

                      ('0032.libjit_batchedadd_f.txt', (1, 4096), '0x407200', 'dense add'),
                      ('0037.libjit_batchedadd_f.txt', (1, 1000), '0x408180', 'dense add'),
                      ]
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        func_type = fun_data[3]
        if func_type == 'conv2d':
            continue
            w_shape = (w_shape[0], w_shape[2], w_shape[3], w_shape[1])
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1)
        elif func_type == 'add':
            continue
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 2)
        elif func_type == 'dense':
            # continue
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, reg_num=1, func_type='dense')
        elif func_type == 'dense add':
            continue
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1)