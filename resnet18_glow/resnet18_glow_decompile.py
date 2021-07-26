#! /usr/bin/python3
import os
import sys
sys.path.append("../..")
from scripts import utils
from scripts import trace_filter
import logging

print('get logger: {}'.format('decompiler.' + __name__))
logger = logging.getLogger('decompiler.' + __name__)

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
    mem_dump_log_path = './mem_dump.log'

    # ==============================================================
    # Step 1 --- Get the Sequence of Layers ---
    # ==============================================================

    utils.get_funcs_trace(prog_path, in_data, log_path, label_file)
    utils.print_layer_label(log_path, 'config.json')  # set addr2label in utils
    
    # ==============================================================
    # Step 2 --- Recover the Shape of each Layer
    # ==============================================================

    # Step 2.1 Generate and Filter Trace
    '''
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
                    func_trace_map[asm_path] = slice_log
                    func_rndaddr_map[asm_path] = (rnd_addr, loop_size, start_addr, end_addr)
    print(func_trace_map)
    print(func_rndaddr_map)
    exit(0)
    '''
    # ==============================================================

    # Step 2.2 Recover Shape with Symbolic Execution

    # Step 2.2.0 Choose Another Random Target Address (if needed)
    #func_name = '0030.txt'
    #asm_path = os.path.join(utils.funcs_dir, func_name)
    #slice_log, rnd_addr, loop_size = trace_filter.filt_trace(asm_path, prog_path, in_data, '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0030_rev.log')
    #print(' slice_log {}\n rnd_addr {}\n loop_size {}\n'.format(slice_log, rnd_addr, loop_size))
    #utils.generate_symbolic_expression(func_name, '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0030_slice.log', exp_log_path, max_inst=5000000)
    #exit(0)

    # Step 2.2.1 Conv and Matmul layers
    '''
    func_trace_map = {'0013.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0013_slice.log', 
                      '0016.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0016_slice.log', 
                      '0029.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0029_slice.log', 
                      '0020.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0020_slice.log', 
                      '0017.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0017_slice.log', 
                      '0032.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0032_slice.log', 
                      '0022.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0022_slice.log', 
                      '0028.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0028_slice.log', 
                      '0023.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0023_slice.log', 
                      '0026.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0026_slice.log', 
                      '0030.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0030_slice.log', 
                      '0024.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0024_slice.log', 
                      '0010.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0010_slice.log', 
                      '0018.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0018_slice.log', 
                      '0012.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0012_slice.log',  # conv

                      '0035.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/0035_slice.log'  # matmul
                      }

    func_rndaddr_map = {'0013.txt': ('0x3200c10', 64, '0x4029c0', '0x4030f9'), 
                        '0016.txt': ('0x313b2cc', 64, '0x403320', '0x40382F'), 
                        '0029.txt': ('0x315cd6c', 512, '0x409300', '0x40B8AE'), 
                        '0020.txt': ('0x319d0ac', 128, '0x4045f0', '0x404C6B'), 
                        '0017.txt': ('0x31b0010', 128, '0x403850', '0x403E8C'), 
                        '0032.txt': ('0x3163440', 512, '0x40e000', '0x410631'), 
                        '0022.txt': ('0x31ac1d0', 64, '0x404da0', '0x405442'), 
                        '0028.txt': ('0x313aed8', 512, '0x406d70', '0x4092D8'), 
                        '0023.txt': ('0x31edc1c', 256, '0x405450', '0x405BAC'), 
                        '0026.txt': ('0x313b14c', 256, '0x4064b0', '0x406C32'), 
                        '0030.txt': ('0x31703b4', 512, '0x40b8d0', '0x40DED4'), 
                        '0024.txt': ('0x3162358', 256, '0x405bd0', '0x40637C'), 
                        '0010.txt': ('0x31d24c0', 64, '0x401850', '0x401F4A'), 
                        '0018.txt': ('0x323d73c', 64, '0x403eb0', '0x4044dc'), 
                        '0012.txt': ('0x313c544', 64, '0x402200', '0x40299B'),  # conv

                        '0035.txt': ('0x313ad58', 64, '0x410b60', '0x41118d')  # matmul
                        }


    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, func_trace_map, compiler='glow') # also matmul layer
    print('all conv and matmul done.')
    for name, result in func_shape.items():
        print(name)
        print(result)
    #exit(0)
    

    # Step 2.2.2 Other layers
    
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
                    
                    # gnereate tmp trace file, it should be fast
                    utils.generate_inst_trace(asm_file, tmp_log_path, prog_path, in_data, timeout=True)
                    # symbolic execution, also should be fast
                    utils.generate_symbolic_expression(asm_file, tmp_log_path, exp_log_path, max_inst=5000000)
                    # --- try to interpret the filter shape from symbolic expression log
                    shape = utils.recover_shape(asm_file, exp_log_path,
                                                mem_read_log_path, mem_write_log_path,
                                                prog_path, in_data, func_type=func_type)
                    print('shape:', shape)
    exit(0)
    '''
    
    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================
    # with all above information, it is feasible to extract paramters
    
    # (name, shape, fused_func, type, padding, stride, param_index)
    func_meta_data = [('0010.txt', (64, 3, 7, 7), '0x401850', 'conv2d'),
                      ('0010.txt', (1, 64), '0x401850', 'add'),
                      ('0012.txt', (64, 64, 3, 3), '0x402200', 'conv2d'),
                      ('0012.txt', (1, 64), '0x402200', 'add'),
                      ('0013.txt', (64, 64, 3, 3), '0x4029c0', 'convDKKC8'),
                      ('0013.txt', (1, 64), '0x4029c0', 'add'),
                      ('0016.txt', (128, 64, 1, 1), '0x403320', 'convDKKC8'),
                      ('0016.txt', (1, 128), '0x403320', 'add'),
                      ('0017.txt', (128, 64, 3, 3), '0x403850', 'conv2d'),
                      ('0017.txt', (1, 128), '0x403850', 'add'),
                      ('0018.txt', (128, 128, 3, 3), '0x403eb0', 'convDKKC8'),
                      ('0018.txt', (1, 128), '0x403eb0', 'add'),
                      ('0020.txt', (128, 128, 3, 3), '0x4045f0', 'conv2d'),
                      ('0020.txt', (1, 128), '0x4045f0', 'add'),
                      ('0022.txt', (256, 128, 1, 1), '0x404da0', 'convDKKC8'),
                      ('0022.txt', (1, 256), '0x404da0', 'add'),
                      ('0023.txt', (256, 128, 3, 3), '0x405450', 'conv2d'),
                      ('0023.txt', (1, 256), '0x405450', 'add'),
                      ('0024.txt', (256, 256, 3, 3), '0x405bd0', 'convDKKC8'),
                      ('0024.txt', (1, 256), '0x405bd0', 'add'),
                      ('0026.txt', (256, 256, 3, 3), '0x4064b0', 'conv2d'),
                      ('0026.txt', (1, 256), '0x4064b0', 'add'),
                      ('0028.txt', (512, 256, 1, 1), '0x406d70', 'convDKKC8'),
                      ('0028.txt', (1, 512), '0x406d70', 'add'),
                      ('0029.txt', (512, 256, 3, 3), '0x409300', 'conv2d'),
                      ('0029.txt', (1, 512), '0x409300', 'add'),
                      ('0030.txt', (512, 512, 3, 3), '0x40b8d0', 'convDKKC8'),
                      ('0030.txt', (1, 512), '0x40b8d0', 'add'),
                      ('0032.txt', (512, 512, 3, 3), '0x40e000', 'conv2d'),
                      ('0032.txt', (1, 512), '0x40e000', 'add'),

                      ('0035.txt', (512, 1000), '0x410b60', 'matmul'),
                      ('0036.txt', (1, 1000), '0x411190', 'batchedadd'), 

                      ]
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        func_type = fun_data[3]
        logger.info('Extract Parameter for {}'.format(func_name))
        if 'conv' in func_type and 'DKKC8' not in func_type:
            w_shape = (w_shape[0], w_shape[2], w_shape[3], w_shape[1])
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1, func_type)
        elif 'conv' in func_type and 'DKKC8' in func_type:
            w_shape = (int(w_shape[0]/8), w_shape[2], w_shape[3], w_shape[1], 8)
            print(w_shape)
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1, func_type)
        elif 'batchedadd' in func_type:
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1, func_type)
        elif 'add' in func_type:
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 2, func_type)
        elif 'matmul' in func_type:
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1, func_type)
        
