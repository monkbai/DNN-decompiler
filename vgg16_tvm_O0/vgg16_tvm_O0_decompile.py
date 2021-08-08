#! /usr/bin/python3
import os
import sys
sys.path.append("../..")
import time
from scripts import utils
from scripts import trace_filter
from scripts import se_engine
import logging
print('get logger: {}'.format('decompiler.'+__name__))
logger = logging.getLogger('decompiler.'+__name__)


if __name__ == '__main__':
    utils.funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/vgg16_funcs/"
    prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/vgg16_tvm_O0_strip"
    in_data = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/cat.bin"
    log_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/func_call.log"
    label_file = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/step1.txt"

    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = './mem_read.log'
    mem_write_log_path = './mem_write.log'
    mem_dump_log_path = 'mem_dump.log'

    # ==============================================================
    # Step 1 --- Get the Sequence of Layers ---
    # ==============================================================
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm')
    utils.print_layer_label_tvm(log_path)
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm', only_fused=True)
    utils.print_layer_label_tvm(log_path, config_path='config.json', only_fused=True)
    
    # get_funcs_trace(prog_path, in_data, log_path, label_file)
    # print_layer_label(log_path)


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
                    func_trace_map[asm_path] = slice_log
                    func_rndaddr_map[asm_path] = (rnd_addr, loop_size, start_addr, end_addr)
                    
    print(func_trace_map)
    print(func_rndaddr_map)
    logger.info('END')
    #exit(0)
    
    # ==============================================================

    # Step 2.2 Recover Shape with Symbolic Execution
    
    # Step 2.2.0 Rerun the Trace Logging (if needed)
    #func_name = '0081.txt'
    #asm_path = os.path.join(utils.funcs_dir, func_name)
    #slice_log, rnd_addr, loop_size, start_addr, end_addr = \
    #    trace_filter.get_trace(asm_path, prog_path, in_data, '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0081.log', compiler='tvm')
    #print(' slice_log {}\n rnd_addr {}\n loop_size {}\n'.format(slice_log, rnd_addr, loop_size))
    #se_engine.extern_functions = {'0x400c50': 'memset'}
    #utils.generate_symbolic_expression(func_name, '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0081_slice.log', exp_log_path, max_inst=5000000)
    #shape = utils.recover_shape_tvm('0081.txt', exp_log_path, mem_read_log_path, mem_write_log_path, prog_path, in_data, func_type='conv2d')
    #print(shape)
    #exit(0)
    
    # Step 2.2.1 Conv and Matmul layers
    func_trace_map = {'0110.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0110_slice.log', 
                      '0103.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0103_slice.log', 
                      '0094.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0094_slice.log', 
                      '0081.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0081_slice.log', 
                      '0074.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0074_slice.log', 
                      '0069.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0069_slice.log', 
                      '0064.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0064_slice.log', 
                      '0057.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0057_slice.log', 
                      '0030.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0030_slice.log',   # conv
                      
                      '0040.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0040_slice.log',   # dense
                      '0038.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0038_slice.log',
                      '0042.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/0042_slice.log', 
                      }

    func_rndaddr_map = {'0110.txt': ('0x702d00', 64, '0x4267b0', '0x42949A'), 
                        '0103.txt': ('0x702e0c', 64, '0x4227d0', '0x4249C1'), 
                        '0094.txt': ('0x13438ac', 64, '0x41e000', '0x420C95'), 
                        '0081.txt': ('0x1343a4c', 64, '0x419f60', '0x41C001'), 
                        '0074.txt': ('0x4609ca8', 64, '0x4163d0', '0x4186A2'), 
                        '0069.txt': ('0x1343b24', 64, '0x412310', '0x414FFA'), 
                        '0064.txt': ('0x702e14', 64, '0x40edd0', '0x4110B6'), 
                        '0057.txt': ('0x13438c4', 64, '0x40b3d0', '0x40D998'), 
                        '0030.txt': ('0x702cdc', 64, '0x403c70', '0x4065D0'),  # conv

                        '0042.txt': ('0x1e06f0a0', 64, '0x4089d0', '0x408E06'), # dense
                        '0040.txt': ('0x1e06f0a0', 64, '0x408200', '0x408636'), 
                        '0038.txt': ('0x1e0778a0', 64, '0x407a30', '0x407E66'), 
                        }

    #shape = utils.recover_shape_tvm('0103.txt', exp_log_path, mem_read_log_path, mem_write_log_path, prog_path, in_data, func_type='conv2d')
    #print(shape)
    #exit(0)
    # We have to pass the external function address to SE engine
    # This can be done automatically, but we do it manually for simplicity
    
    se_engine.extern_functions = {'0x400c50': 'memset'}  # address in .plt, name
    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, func_trace_map, compiler='tvm')  # also all dense
    print('all conv and dense done.')
    for name, result in func_shape.items():
        print(name)
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
    asm_files = os.listdir(utils.funcs_dir)
    se_engine.extern_functions = {'0x400c50': 'memset'}  # address in .plt, name
    results_dict = dict()
    for asm_file in asm_files:
        if 'labels' not in asm_file and asm_file.endswith('.txt'):
            asm_path = os.path.join(utils.funcs_dir, asm_file)
            start_addr, _ = utils.get_func_range(asm_path)
            if start_addr in utils.addr2label.keys():
                func_type = utils.addr2label[start_addr]
                if ('pool' in func_type or 'add' in func_type) and 'conv' not in func_type:
                    # transpose, expand_dims and relu could be ignored, batchnormalization always follow after a conv layer
                    print('SE for {}, {}'.format(asm_file, func_type))
                    tmp_log_path = os.path.basename(asm_file)[:-4] + '.log'
                    # gnereate tmp trace file, it should be fast
                    utils.generate_inst_trace(asm_file, tmp_log_path, prog_path, in_data, timeout=True)
                    # symbolic execution, also should be fast
                    #utils.generate_symbolic_expression(asm_file, tmp_log_path, exp_log_path, max_inst=5000000)
                    # --- try to interpret the filter shape from symbolic expression log
                    #shape = utils.recover_shape_tvm(asm_file, exp_log_path,
                    #                            mem_read_log_path, mem_write_log_path,
                    #                            prog_path, in_data, func_type=func_type)
                    #print('shape:', shape)
                    #results_dict[asm_file] = shape
    for name, result in results_dict.items():
        print(name)
        print(result)
    #exit(0)
    

    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================
    
    func_meta_data = [('0064.txt', (64, 3, 3, 3), '0x40e190', 'conv2d', 1),
                      ('0057.txt', (512, 512, 3, 3), '0x40a790', 'conv2d', 1),
                      ('0030.txt', (512, 256, 3, 3), '0x402990', 'conv2d', 1),
                      ('0081.txt', (128, 64, 3, 3), '0x418c20', 'conv2d', 1),
                      ('0110.txt', (256, 128, 3, 3), '0x4252a0', 'conv2d', 1),
                      ('0094.txt', (256, 256, 3, 3), '0x41cd20', 'conv2d', 1),
                      ('0103.txt', (128, 128, 3, 3), '0x4218b0', 'conv2d', 1),
                      ('0074.txt', (512, 512, 3, 3), '0x415020', 'conv2d', 1),
                      ('0069.txt', (64, 64, 3, 3), '0x4110e0', 'conv2d', 1),
                      
                      ('0019.txt', (1, 64), '0x4015a0', 'add', 1),
                      ('0050.txt', (1, 128), '0x409d40', 'add', 1),
                      ('0105.txt', (1, 256), '0x4249f0', 'add', 1),
                      ('0025.txt', (1, 512), '0x4024d0', 'add', 1),
                      ('0036.txt', (1, 512), '0x406e80', 'add', 1),

                      ('0040.txt', (4096, 25088), '0x407e90', 'dense', 1),
                      ('0038.txt', (4096, 4096), '0x4076c0', 'dense', 1),
                      ('0042.txt', (1000, 4096), '0x408660', 'dense', 1),
                      
                      ('0046.txt', (1, 4096), '0x409290', 'add', 1),
                      ('0034.txt', (1, 1000), '0x406a40', 'add', 1), 
                     ]
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        func_type = fun_data[3]
        data_index = fun_data[4]
        utils.extract_params_tvm(prog_path, in_data, w_shape, dump_point, mem_dump_log_path, func_name, func_type, data_idx=data_index)
        