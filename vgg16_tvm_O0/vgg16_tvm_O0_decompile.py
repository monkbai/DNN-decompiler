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


"""
def dict_to_json(dict_obj: dict, output_path: str):
    j = json.dumps(dict_obj)
    with open(output_path, 'w') as f:
        f.write(j)


def json_to_dict(json_path: str):
    with open(json_path, 'r') as f:
        j_txt = f.read()
        dict_obj = json.loads(s=j_txt)
        return dict_obj


addr2label = dict()
addr2funcs = dict()

funcs_dir = './vgg16_strip_funcs/'
# ==============================================================
# Generate the function call trace
# ==============================================================


def get_addr_list(label_path: str):
    addr_list = []
    with open(label_path, 'r') as f:
        label_txt = f.read()
        lines = label_txt.split('\n')
        for line in lines:
            if ':' not in line:
                continue
            name, label = line.split(':')
            if len(label.strip()) > 0:
                addr = name.strip()
                addr = addr.split('_')[1]
                addr = addr.split('.')[0]
                addr = '0x' + addr

                addr_list.append(addr)

                addr2label[addr] = label.strip()
                addr2funcs[addr] = name.strip()
    return addr_list


def get_funcs_trace(prog_path: str, in_data: str, log_path: str, label_file: str):

    # prog_path = './vgg16_strip'
    prog_path = os.path.abspath(prog_path)
    # in_data = './cat.bin'
    in_data = os.path.abspath(in_data)
    # log_path = './vgg16_strip_func_call.log'
    log_path = os.path.abspath(log_path)
    # label_file = './step1.txt'
    label_file = os.path.abspath(label_file)

    addr_list = get_addr_list(label_file)
    # addr_list = '0x40d89a,0x42a324,0x42dcc0,0x417b3e,0x4221f5,0x420ee0,0x421b40,0x4213f5,0x42065a,0x40e400,0x41c11e,0x42953e,0x406bca,0x428d9a,0x418edf,0x42d740,0x42286a,0x41979a,0x41826a,0x41688e,0x420296,0x401e4e,0x4250f0,0x401720,0x42e224,0x41cd7a,0x420ac0,0x416330,0x402b4e,0x424d80,0x413f9a,0x40eafe,0x40383a,0x42de41,0x40f75a,0x429a40,0x429160,0x40746a,0x42dad2,0x429ad9,0x4127ea,0x42867a,0x425dba,0x417100,0x41f9ce,0x429c62,0x4011f4,0x40dd9e,0x42da02,0x42dbb0,0x4257f4,0x429cc0,0x417533,0x41346e,0x40a8ba'

    func_call_trace(prog_path, in_data, addr_list, log_path)

    dict_to_json(addr2label, './addr2label.json')
    dict_to_json(addr2funcs, './addr2funcs.json')


def print_layer_label(trace_log_path: str):
    global addr2label, addr2funcs
    addr2label = json_to_dict('./addr2label.json')
    addr2funcs = json_to_dict('./addr2funcs.json')

    trace_log_path = os.path.abspath(trace_log_path)
    with open(trace_log_path, 'r') as f:
        trace_txt = f.read()
        lines = trace_txt.split('\n')
        for line in lines:
            if not line.startswith('0x'):
                continue
            addr = hex(int(line.strip(), 16))
            if 'reshape' != addr2label[addr]:
                print('{}: {}'.format(addr, addr2label[addr]))


# ==============================================================
# Do lightweight Symbolic Execution
# ==============================================================

def get_func_range(func_asm_path: str):
    start_addr = ''
    end_addr = ''
    with open(func_asm_path, 'r') as f:
        asm_txt = f.read()
        lines = asm_txt.split('\n')
        for line in lines:
            if line.startswith(';'):
                continue
            start_addr = line.split(':')[0]
            break
        lines.reverse()
        for line in lines:
            if line.startswith(';') or len(line) < 1:
                continue
            end_addr = line.split(':')[0]
            break
    return start_addr, end_addr


def generate_inst_trace(func_name: str, log_path: str, prog_path, data_path: str):
    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)
    start_addr, end_addr = get_func_range(func_asm_path)

    log_path = os.path.abspath(log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)

    inst_trace_log(log_path, start_addr, end_addr, prog_path, data_path)


def generate_symbolic_expression(func_name: str, inst_log_path: str, exp_log_path: str):
    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)

    inst_log_path = os.path.abspath(inst_log_path)
    exp_log_path = os.path.abspath(exp_log_path)

    lightweight_SymEx(func_asm_path, inst_log_path, exp_log_path, max_inst_num=5000000)  # TODO: max_inst_num


# ==============================================================
# Recover ses shapes using heuristics
# ==============================================================


def recover_shape(func_name: str, mem_exp_log: str,
                  mem_read_log_path: str, mem_write_log_path: str,
                  prog_path: str, data_path: str, func_type='conv2d'):
    mem_read_log_path = os.path.abspath(mem_read_log_path)
    mem_write_log_path = os.path.abspath(mem_write_log_path)
    prog_path = os.path.abspath(prog_path)
    data_path = os.path.abspath(data_path)

    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)
    start_addr, end_addr = get_func_range(func_asm_path)

    if func_type == 'conv2d':
        mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, data_path)
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        read_mem_regions = memory_slices(mem_read_log_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        filter_shape, input_shape, output_shape = explain_tvm_conv2d_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return filter_shape
    elif func_type == 'dense':
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        input_size, output_size = explain_tvm_dense_result(mem_exp_log, write_mem_regions)
        return output_size, input_size


# ==============================================================
# Handle all conv2d functions
# ==============================================================


def handle_all_conv(label_file_path: str):
    label_file_path = os.path.abspath(label_file_path)
    # --- get conv2d functions' name
    funcs_name_list = []
    with open(label_file_path, 'r') as f:
        labels = f.read()
        lines = labels.split('\n')
        for line in lines:
            if ':' not in line:
                continue
            name, label = line.split(':')
            if len(label.strip()) > 0 and 'conv2d' in label and not name.startswith('0126'):
                name = name.strip()
                funcs_name_list.append(name)

    func_shape = dict()
    for func_name in funcs_name_list:
        print(func_name)
        # --- recover the shape of each layer
        tmp_log_path = './inst_trace.log'
        generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)
        exp_log_path = './mem_exp.log'
        generate_symbolic_expression(func_name, tmp_log_path, exp_log_path)

        # --- try to interpret the filter shape from symbolic expression log
        mem_read_log_path = 'mem_read.log'
        mem_write_log_path = 'mem_write.log'
        filter_shape = recover_shape(func_name, exp_log_path, mem_read_log_path, mem_write_log_path, prog_path, in_data)
        func_shape[func_name] = filter_shape
    return func_shape


def extract_params(prog_path: str, in_data: str, w_shape: tuple, dump_point: str, log_path: str, func_name: str, func_type='conv2d'):
    '''
    :param dump_point: the start address of layer function (before reshaping the parameters)
    '''
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    if func_type == 'conv2d':
        dwords_len = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
    elif func_type == 'dense':
        dwords_len = w_shape[0] * w_shape[1]
    rm_log(log_path)
    dump_dwords_2(prog_path, in_data, dump_point, dwords_len, log_path)

    # then convert dwords to floats
    with open(log_path, 'r') as f:
        dw_txt = f.read()
        f.close()
        end_count = dw_txt.count('end')
        dw_segs = dw_txt.split('end')[:end_count]
        for i in range(end_count):
            dw_txt = dw_segs[i].strip()
            dw_txt = dw_txt[dw_txt.find('\n')+1:]
            float_array = convert_dwords2float(dw_txt, dwords_len)

            w = np.asarray(float_array)
            w = w.reshape(w_shape)
            # print(type(w))
            lists = w.tolist()
            json_str = json.dumps(lists)
            # print(json_str)
            if func_type == 'conv2d':
                json_name = func_name[:func_name.rfind('.')] + '.weights_{}.json'.format(i)
            elif func_type == 'dense':
                json_name = func_name[:func_name.rfind('.')] + '.dense_weights_{}.json'.format(i)
            with open(json_name, 'w') as wf:
                wf.write(json_str)
                wf.close()
    rm_log(log_path)
"""

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
    '''
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
    exit(0)
    '''
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
    '''
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
    logger.info('# ' + '='*20)

    # Step 2.2.2 Other layers
    # the BatchNorm2d is implemented with a special sequence (add, sqrt, divide, multiply, expand_dims, multiply, negative, multiply, add, expand_dims, add)
    
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
                    
                    # gnereate tmp trace file, it should be fast
                    utils.generate_inst_trace(asm_file, tmp_log_path, prog_path, in_data, timeout=True)
                    # symbolic execution, also should be fast
                    utils.generate_symbolic_expression(asm_file, tmp_log_path, exp_log_path, max_inst=5000000)
                    # --- try to interpret the filter shape from symbolic expression log
                    shape = utils.recover_shape_tvm(asm_file, exp_log_path,
                                                mem_read_log_path, mem_write_log_path,
                                                prog_path, in_data, func_type=func_type)
                    print('shape:', shape)
                    results_dict[asm_file] = shape
    for name, result in results_dict.items():
        print(name)
        print(result)
    exit(0)
    '''

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
        
