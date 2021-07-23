#! /usr/bin/python3
import os
import sys
sys.path.append("../..")
from scripts import utils
from scripts import trace_filter


if __name__ == '__main__':
    utils.funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/resnet18_glow_ida/"

    prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/resnet18_strip.out"
    in_data = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/cat.bin"
    log_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/func_call.log"
    label_file = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/step1.txt"

    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = './mem_read.log'
    mem_write_log_path = './mem_write.log'

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
    tmp_write_log = '../../tmp_mem_write.log'
    func_addrs = utils.find_rand_addr(prog_path, in_data, tmp_write_log, label_file)
    print('start_addr, end_addr, early_stop, loop_size, rand_addr')
    for name, result in func_addrs.items():
        print(name)
        print(result)
    exit(0)
    """

    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, func_trace_map)
    for name, result in func_shape.items():
        print(name)
        print(result)
    exit(0)

    """
    # conv2d layers
    func_type = 'conv2d'
    func_name = '0017.libjit_conv2d_f.txt'  # 64, 3, 7, 7
    tmp_log_path = '../../0x401850-0x401f61_slice.log'
    #func_name = '0019.libjit_conv2d_f.txt'  # 64, 64, 3, 3
    #tmp_log_path = '../../0x402200-0x4029b6_slice.log'
    #func_name = '0020.libjit_convDKKC8_f.txt'  # 64, 64, 3, 3
    #tmp_log_path = '../../0x4029c0-0x4030f9_slice.log'
    #func_name = '0023.libjit_convDKKC8_f.txt'  # 128, 64, 1, 1
    #tmp_log_path = '../../0x403320-0x403846_slice.log'
    #func_name = '0024.libjit_conv2d_f.txt'  # 128, 64, 3, 3
    #tmp_log_path = '../../0x403850-0x403eaa_slice.log'
    #func_name = '0025.libjit_convDKKC8_f.txt'  # 128, 128, 3, 3
    #tmp_log_path = '../../0x403eb0-0x4044dc_slice.log'
    #func_name = '0027.libjit_conv2d_f.txt'  # 128, 128, 3, 3
    #tmp_log_path = '../../0x4045f0-0x404c89_slice.log'
    #func_name = '0029.libjit_convDKKC8_f.txt'  # 256, 128, 1, 1
    #tmp_log_path = '../../0x404da0-0x405442_slice.log'
    #func_name = '0030.libjit_conv2d_f.txt'  # 256, 128, 3, 3
    #tmp_log_path = '../../0x405450-0x405bca_slice.log'
    #func_name = '0031.libjit_convDKKC8_f.txt'  # 256, 256, 3, 3
    #tmp_log_path = '../../0x405bd0-0x40639a_slice.log'
    #func_name = '0033.libjit_conv2d_f.txt'  # 256, 256, 3, 3
    #tmp_log_path = '../../0x4064b0-0x406c50_slice.log'
    #func_name = '0035.libjit_convDKKC8_f.txt'  #  512, 256, 1, 1
    #tmp_log_path = '../../0x406d70-0x4092f2_slice.log'
    #func_name = '0036.libjit_conv2d_f.txt'  # 512, 256, 3, 3
    #tmp_log_path = '../../0x409300-0x40b8cc_slice.log'
    #func_name = '0037.libjit_convDKKC8_f.txt'  # 512, 512, 3, 3
    #tmp_log_path = '../../0x40b8d0-0x40deee_slice.log'
    #func_name = '0039.libjit_conv2d_f.txt'  # 512, 512, 3, 3
    #tmp_log_path = '../../0x40e000-0x41064f_slice.log'
    # dense/fully-connected layers
    #func_type = 'dense'
    #func_name = '0042.libjit_matmul_f.txt'  # 1000, 512
    #tmp_log_path = '../../0x410b60-0x41118d_slice.log'
    # bias add layer
    #func_type = 'add'
    #func_name = '0043.libjit_batchedadd_f.txt'  # dense add 1000
    # max-poll layers
    #func_type = 'max'
    #func_name = '0018.libjit_max_pool_f.txt'  # max 3, 2, kernel, stride

    # utils.generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)

    utils.generate_symbolic_expression(func_name, tmp_log_path, exp_log_path, max_inst=5000000)

    # --- try to interpret the filter shape from symbolic expression log
    shape = utils.recover_shape(func_name, exp_log_path,
                                mem_read_log_path, mem_write_log_path,
                                prog_path, in_data, func_type=func_type)
    print(shape)
    exit(0)
    """
    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================

    mem_dump_log_path = 'mem_dump.log'
    # (name, shape, fused_func, type, padding, stride, param_index)
    func_meta_data = [('0017.libjit_conv2d_f.txt', (64, 3, 7, 7), '0x401850', 'conv2d'),
                      ('0017.libjit_conv2d_f.txt', (1, 64), '0x401850', 'add'),
                      ('0019.libjit_conv2d_f.txt', (64, 64, 3, 3), '0x402200', 'conv2d'),
                      ('0019.libjit_conv2d_f.txt', (1, 64), '0x402200', 'add'),
                      ('0020.libjit_convDKKC8_f.txt', (64, 64, 3, 3), '0x4029c0', 'conv2d'),
                      ('0020.libjit_convDKKC8_f.txt', (1, 64), '0x4029c0', 'add'),
                      ('0023.libjit_convDKKC8_f.txt', (128, 64, 1, 1), '0x403320', 'conv2d'),
                      ('0023.libjit_convDKKC8_f.txt', (1, 128), '0x403320', 'add'),
                      ('0024.libjit_conv2d_f.txt', (128, 64, 3, 3), '0x403850', 'conv2d'),
                      ('0024.libjit_conv2d_f.txt', (1, 128), '0x403850', 'add'),
                      ('0025.libjit_convDKKC8_f.txt', (128, 128, 3, 3), '0x403eb0', 'conv2d'),
                      ('0025.libjit_convDKKC8_f.txt', (1, 128), '0x403eb0', 'add'),
                      ('0027.libjit_conv2d_f.txt', (128, 128, 3, 3), '0x4045f0', 'conv2d'),
                      ('0027.libjit_conv2d_f.txt', (1, 128), '0x4045f0', 'add'),
                      ('0029.libjit_convDKKC8_f.txt', (256, 128, 1, 1), '0x404da0', 'conv2d'),
                      ('0029.libjit_convDKKC8_f.txt', (1, 256), '0x404da0', 'add'),
                      ('0030.libjit_conv2d_f.txt', (256, 128, 3, 3), '0x405450', 'conv2d'),
                      ('0030.libjit_conv2d_f.txt', (1, 256), '0x405450', 'add'),
                      ('0031.libjit_convDKKC8_f.txt', (256, 256, 3, 3), '0x405bd0', 'conv2d'),
                      ('0031.libjit_convDKKC8_f.txt', (1, 256), '0x405bd0', 'add'),
                      ('0033.libjit_conv2d_f.txt', (256, 256, 3, 3), '0x4064b0', 'conv2d'),
                      ('0033.libjit_conv2d_f.txt', (1, 256), '0x4064b0', 'add'),
                      ('0035.libjit_convDKKC8_f.txt', (512, 256, 1, 1), '0x406d70', 'conv2d'),
                      ('0035.libjit_convDKKC8_f.txt', (1, 512), '0x406d70', 'add'),
                      ('0036.libjit_conv2d_f.txt', (512, 256, 3, 3), '0x409300', 'conv2d'),
                      ('0036.libjit_conv2d_f.txt', (1, 512), '0x409300', 'add'),
                      ('0037.libjit_convDKKC8_f.txt', (512, 512, 3, 3), '0x40b8d0', 'conv2d'),
                      ('0037.libjit_convDKKC8_f.txt', (1, 512), '0x40b8d0', 'add'),
                      ('0039.libjit_conv2d_f.txt', (512, 512, 3, 3), '0x40e000', 'conv2d'),
                      ('0039.libjit_conv2d_f.txt', (1, 512), '0x40e000', 'add'),

                      ('0042.libjit_matmul_f.txt', (512, 1000), '0x410b60', 'dense'),
                      ('0043.libjit_batchedadd_f.txt', (1, 1000), '0x411190', 'dense add'),  # bias add

                      # ('0112.function_414370.txt', (1, 512), '0x4140c0', 'var', 0, 0, 0),  # norm var
                      # ('0114.function_414740.txt', (1, 512), '0x414440', 'gamma', 0, 0, 1),  # norm gamma
                      # ('0158.function_424000.txt', (1, 512), '0x423e00', 'mean', 0, 0, 0),  # norm mean
                      # ('0125.function_418d00.txt', (1, 512), '0x418a00', 'beta', 0, 0, 1),  # norm beta
                      #
                      # ('0129.function_419630.txt', (1, 256), '0x419380', 'var', 0, 0, 0),  # norm var
                      # ('0110.function_413fe0.txt', (1, 256), '0x413ce0', 'gamma', 0, 0, 1),  # norm gamma
                      # ('0096.function_411880.txt', (1, 256), '0x411680', 'mean', 0, 0, 0),  # norm mean
                      # ('0094.function_4115a0.txt', (1, 256), '0x4112a0', 'beta', 0, 0, 1),  # norm beta
                      #
                      # ('0201.function_434b50.txt', (1, 128), '0x4348a0', 'var', 0, 0, 0),  # norm var
                      # ('0160.function_4243d0.txt', (1, 128), '0x4240d0', 'gamma', 0, 0, 1),  # norm gamma
                      # ('0052.function_4021e0.txt', (1, 128), '0x401fe0', 'mean', 0, 0, 0),  # norm mean
                      # ('0195.function_431870.txt', (1, 128), '0x431570', 'beta', 0, 0, 1),  # norm beta
                      #
                      # ('0146.function_41ee50.txt', (1, 64), '0x41eba0', 'var', 0, 0, 0),  # norm var
                      # ('0205.function_435470.txt', (1, 64), '0x435180', 'gamma', 0, 0, 1),  # norm gamma
                      # ('0191.function_4311a0.txt', (1, 64), '0x430fb0', 'mean', 0, 0, 0),  # norm mean
                      # ('0106.function_413580.txt', (1, 64), '0x413290', 'beta', 0, 0, 1),  # norm beta

                      ]
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        func_type = fun_data[3]
        if func_type == 'conv2d' and 'DKKC8' not in func_name:
            continue
            w_shape = (w_shape[0], w_shape[2], w_shape[3], w_shape[1])
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1)
        elif func_type == 'conv2d' and 'DKKC8' in func_name:
            # continue
            w_shape = (int(w_shape[0]/8), w_shape[2], w_shape[3], w_shape[1], 8)
            print(w_shape)
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1)
        elif func_type == 'add':
            continue
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 2)
        elif func_type == 'dense':
            continue
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, reg_num=1, func_type='dense')
        elif func_type == 'dense add':
            continue
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1)
