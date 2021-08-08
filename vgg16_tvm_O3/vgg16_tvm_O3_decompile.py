#! /usr/bin/python3
import os
import sys
sys.path.append("../..")
from scripts import trace_filter
from scripts import utils
from scripts import se_engine
import logging
print('get logger: {}'.format('decompiler.'+__name__))
logger = logging.getLogger('decompiler.'+__name__)



if __name__ == '__main__':
    utils.funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/vgg16_funcs/"
    prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/vgg16_tvm_O3_strip"
    in_data = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/cat.bin"
    log_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/func_call.log"
    label_file = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/step1.txt"

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
    #exit(0)
    
    """
    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, optimized=True)
    for name, result in func_shape.items():
        print(name)
        print(result)
    exit(0)
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
                        trace_filter.get_trace(asm_path, prog_path, in_data, trace_path, compiler='tvm')
                    func_trace_map[asm_path] = slice_log
                    func_rndaddr_map[asm_path] = (rnd_addr, loop_size, start_addr, end_addr)
                    
    print(func_trace_map)
    print(func_rndaddr_map)
    logger.info('END')
    #exit(0)
    

    # ==============================================================
    # Step 2.2 Recover Shape with Symbolic Execution
    
    # Step 2.2.0 Rerun the Trace Logging (if needed)
    #func_name = '0022.txt'
    #asm_path = os.path.join(utils.funcs_dir, func_name)
    #slice_log, rnd_addr, loop_size, start_addr, end_addr = \
    #    trace_filter.get_trace(asm_path, prog_path, in_data, '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0022.log', compiler='tvm')
    #print(' slice_log {}\n rnd_addr {}\n loop_size {}\n'.format(slice_log, rnd_addr, loop_size))
    #se_engine.extern_functions = {'0x400c50': 'memset'}
    #utils.generate_symbolic_expression(func_name, '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0022_slice.log', exp_log_path, max_inst=5000000)
    #shape = utils.recover_shape_tvm('0022.txt', exp_log_path, mem_read_log_path, mem_write_log_path, prog_path, in_data, func_type='conv2d', optimized=True)
    #print(shape)
    #exit(0)
    
    # Step 2.2.1 Conv and Matmul layers
    func_trace_map = { 
                      '0070.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0070_slice.log', 
                      '0067.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0067_slice.log', 
                      '0064.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0064_slice.log', 
                      '0059.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0059_slice.log', 
                      '0046.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0046_slice.log', 
                      '0043.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0043_slice.log', 
                      '0030.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0030_slice.log', 
                      '0025.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0025_slice.log', 
                      '0022.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0022_slice.log',  # conv

                      '0078.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0078_slice.log',  # dense
                      '0076.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0076_slice.log',
                      '0048.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0048_slice.log',
                      }


    func_rndaddr_map = { 
                        '0070.txt': ('0x23102b44', 64, '0x427860', '0x42A810'), 
                        '0067.txt': ('0x230fe694', 64, '0x4235a0', '0x426F67'), 
                        '0064.txt': ('0x230fe42c', 64, '0x41f140', '0x4225C2'), 
                        '0059.txt': ('0x230fef60', 64, '0x41a060', '0x41DA22'), 
                        '0046.txt': ('0x6fbfe8', 64, '0x414630', '0x416D18'), 
                        '0043.txt': ('0x23101efc', 64, '0x410850', '0x413A59'), 
                        '0030.txt': ('0x13d2538', 64, '0x40b980', '0x40E430'), 
                        '0025.txt': ('0x230ffb0c', 64, '0x407990', '0x40AA4E'), 
                        '0022.txt': ('0x230fed68', 64, '0x403030', '0x40699B'),

                        '0078.txt': ('0x230fd760', 64, '0x42cc20', '0x42D056'), 
                        '0076.txt': ('0x648b60', 64, '0x42c280', '0x42C6B6'), 
                        '0048.txt': ('0x23104760', 64, '0x417270', '0x4176A6'),
                        }


    #se_engine.extern_functions = {'0x400c50': 'memset'}
    #utils.generate_symbolic_expression('0070.txt', '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/0070.log', exp_log_path, max_inst=5000000)
    #shape = utils.recover_shape_tvm('0070.txt', exp_log_path, mem_read_log_path, mem_write_log_path, prog_path, in_data, func_type='conv2d', optimized=True)
    #print(shape)
    #exit(0)
    # We have to pass the external function address to SE engine
    # This can be done automatically, but we do it manually for simplicity
    
    se_engine.extern_functions = {'0x400c50': 'memset'}  # address in .plt, name
    func_shape = utils.handle_all_conv(prog_path, in_data, label_file, func_trace_map, compiler='tvm', optimized=True)  # also all dense
    print('all conv and dense done.')
    for name, result in func_shape.items():
        print(name)
        if len(result) == 4:
            print('filter_shape', result[0])
            print('input_shape', result[1])
            print('output_shape', result[2])
            print('layout_shape', result[3])
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
                if ('pool' in func_type or 'add' in func_type) and 'conv' not in func_type and 'dense' not in func_type:
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

    
    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================
    func_meta_data = [('0064.txt', (512, 256, 3, 3), '0x41E0B0', 'conv2d', [16, 1, 3, 3, 256, 32], 1),
                      ('0067.txt', (256, 128, 3, 3), '0x4225E0', 'conv2d', [8, 4, 3, 3, 32, 32], 1),
                      ('0070.txt', (64, 3, 3, 3), '0x426F90', 'conv2d', [2, 1, 3, 3, 3, 32], 1),
                      ('0022.txt', (256, 256, 3, 3), '0x401FA0', 'conv2d', [8, 1, 3, 3, 256, 32], 1),
                      ('0025.txt', (128, 64, 3, 3), '0x4069D0', 'conv2d', [2, 2, 3, 3, 32, 64], 1),
                      ('0030.txt', (512, 512, 3, 3), '0x40B0E0', 'conv2d', [32, 1, 3, 3, 512, 16], 1),
                      ('0043.txt', (128, 128, 3, 3), '0x40FB20', 'conv2d', [2, 2, 3, 3, 64, 64], 1),
                      ('0046.txt', (512, 512, 3, 3), '0x413A80', 'conv2d', [32, 1, 3, 3, 512, 16], 1),
                      ('0059.txt', (64, 64, 3, 3), '0x4192E0', 'conv2d', [2, 4, 3, 3, 16, 32], 1),

                      ('0064.txt', (1, 512), '0x41E0B0', 'add', 2),
                      ('0067.txt', (1, 256), '0x4225E0', 'add', 2),
                      ('0070.txt', (1, 64), '0x426F90', 'add', 2),
                      ('0022.txt', (1, 256), '0x401FA0', 'add', 2),
                      ('0025.txt', (1, 128), '0x4069D0', 'add', 2),
                      ('0030.txt', (1, 512), '0x40B0E0', 'add', 2),
                      ('0043.txt', (1, 128), '0x40FB20', 'add', 2),
                      ('0046.txt', (1, 512), '0x413A80', 'add', 2),
                      ('0059.txt', (1, 64), '0x4192E0', 'add', 2),

                      ('0076.txt', (4096, 4096), '0x42BD40', 'dense', 1),
                      ('0078.txt', (4096, 25088), '0x42C6E0', 'dense', 1),
                      ('0048.txt', (1000, 4096), '0x416D40', 'dense', 1),

                      ('0076.txt', (1, 4096), '0x42BD40', 'add', 2),
                      ('0078.txt', (1, 4096), '0x42C6E0', 'add', 2),
                      ('0048.txt', (1, 1000), '0x416D40', 'add', 2),
                      ]
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        func_type = fun_data[3]
        data_index = fun_data[-1]  # conv2d weights
        layout_shape = ()
        if func_type == 'conv2d':
            layout_shape = tuple(fun_data[4])
        logger.info('Extract Parameter for {}'.format(func_name))
        utils.extract_params_tvm(prog_path, in_data, w_shape, dump_point, mem_dump_log_path, func_name,
                                 func_type=func_type, data_idx=data_index, special_layout=layout_shape)
