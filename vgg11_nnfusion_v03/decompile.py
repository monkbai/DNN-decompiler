#!/usr/bin/python3
import os
import sys
sys.path.append("../")
from pin_tools import nnfusion_conv, nnfusion_gemm, nnfusion_pool, nnfusion_trace, convert_dwords2float
import utils
import numpy as np
import json
import re
import logging
print('get logger: {}'.format('decompiler.'+__name__))
logger = logging.getLogger('decompiler.'+__name__)

#
# Find some special function addresses
# Automatically or manually
addr_dict = {
    #
    'base_addr': '0x0',  # recompiled with -no-pie
    'kernel_entry_addr': '0x40CD00',
    'MlasConvPrepare_addr': '0x416650',
    'MlasConv_addr': '0x415e40',
    'MlasGemm_addr': '0x414030',
    'MlasPool_addr': '0x416b80',
    'concurrency_addr': '0x430100',  # concurrency::ThreadPool::ParallelFor
    #
    'Broadcast': '0x411ee0',  # cpu_reference_broadcast
    'Reshape': '0x410430',  # cpu_reference_reshape
}

prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg11_nnfusion_v03/vgg11_nnfusion_strip"
data_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg11_nnfusion_v03/cat.bin"


funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg11_nnfusion_v03/vgg11_nnfusion_strip_funcs/"


def runtime_addr(addr: str):
    return hex(int(addr_dict['base_addr'], 16) + int(addr, 16))


def get_dict_name(func_offset: str):
    for name, offset in addr_dict.items():
        if int(offset, 16) == int(func_offset, 16):
            return name


# def get_func_name(func_addr: str):
#     if func_addr.startswith('0x'):
#         func_addr = func_addr[2:]
#     files = os.listdir(funcs_dir)
#     func_addr = func_addr.upper()
#     func_name = '_{}.txt'.format(func_addr)
#     for file in files:
#         if file.endswith(func_name):
#             return os.path.join(funcs_dir, file)
def get_func_name(func_addr: str):
    if not func_addr.startswith('0x'):
        func_addr = '0x' + func_addr
    func_addr = func_addr.lower()
    files = os.listdir(funcs_dir)
    for f in files:
        f_path = os.path.join(funcs_dir, f)
        start_addr, end_addr = utils.get_func_range(f_path)
        if start_addr == func_addr:
            return f_path


def get_all_operator():
    operator_list = []
    kernel_entry_path = get_func_name(addr_dict['kernel_entry_addr'])
    with open(kernel_entry_path, 'r') as f:
        func_txt = f.read()
        lines = func_txt.split('\n')
        for line in lines:
            if line.startswith('0x'):
                if 'call    sub_' in line:
                    callee_addr = line[line.find('call ') + 5:]
                    callee_addr = callee_addr.strip()
                    callee_addr = callee_addr.split('_')[1]
                    callee_name = get_func_name(callee_addr)
                    if not callee_name:
                        print('maybe disassemble error: function border')
                        print(callee_addr)
                    operator_list.append(callee_name)
        return operator_list


def identify_operator(func_name: str):
    with open(func_name, 'r') as f:
        lines = f.read()
        lines = lines.split('\n')
        callee_addr_list = []
        callee_type = ''
        for line in lines:
            if line.startswith('0x'):
                if 'call    sub_' in line:
                    callee_addr = line[line.find('call ') + 5:]
                    callee_addr = callee_addr.strip()
                    callee_addr = callee_addr.split('_')[1]
                    callee_addr_list.append('sub_' + callee_addr)
                    callee_name = get_dict_name(callee_addr)
                    if not callee_name:
                        pass
                        # callee_type = 'unknown'
                    elif len(callee_type) > 0:
                        pass
                    elif 'MlasConvPrepare' in callee_name:
                        callee_type = 'Conv'
                    elif 'MlasConv' in callee_name:
                        callee_type = 'Conv'
                    elif 'MlasGemm' in callee_name:
                        callee_type = 'Dense'
                    elif 'MlasPool' in callee_name:
                        callee_type = 'Pool'
                    elif 'concurrency' in callee_name:
                        callee_type = 'concurrency'
                    elif 'Broadcast' in callee_name:
                        callee_type = 'Broadcast'
                    elif 'Reshape' in callee_name:
                        callee_type = 'Reshape'
                    else:
                        pass
                        # callee_type = 'unknown'
        if len(callee_type) == 0:
            callee_type = 'ignore'  # callee_type = 'unknown'
        if 'concurrency' in callee_type:
            func_ptr_list, asm_path_list = get_concurrency_func_ptr(func_name)
            print(func_ptr_list)
            callee_type += ' --> ' + simple_label(asm_path_list[0])
        # final label
        print('predict the operator type: {}'.format(callee_type))
        print(callee_addr_list)
        if callee_type == 'Broadcast' or callee_type == 'Reshape' or callee_type == 'ignore':
            return 
        return [func_name[func_name.rfind('/')+1:], callee_type]


def get_concurrency_func_ptr(op_file_path: str):
    func_ptr_list = []
    asm_path_list = []
    with open(op_file_path, 'r') as f:
        asm_txt = f.read()
        asm_lines = asm_txt.split('\n')
        for line in asm_lines:
            if line.startswith(';'):
                continue
            elif 'lea     rax' in line:
                target_addr = line[line.find('rax,') + 4:]
                target_addr = target_addr.strip()
                func_ptr_list.append(target_addr)

                target_addr = target_addr.split('_')[1].strip()
                target_func = get_func_name(target_addr)
                asm_path_list.append(target_func)  # TODO
        return func_ptr_list, asm_path_list


def simple_label(asm_path: str):
    label = ''
    with open(asm_path, 'r') as f:
        asm_txt = f.read()
        if asm_txt.find('vaddps') != -1 or asm_txt.find('vaddss') != -1:
            label += 'Add '
        if asm_txt.find('vmaxps') != -1 or asm_txt.find('vmaxss') != -1:
            label += 'ReLU '
        if asm_txt.find('vsqrtss') != -1 and asm_txt.find('vdivss') != -1:
            label += 'BatchNormalization '
    if len(label) == 0:
        label = 'unknown'
    return label


# ----------------------------------------------------------
pool_list = []
conv_list = []
gemm_list = []

def get_shape_info():
    global pool_list, conv_list, gemm_list
    def get_pool_list(log_path: str):
        pool_list = []
        with open(log_path, 'r') as f:
            log = f.read()
            blks = log.split('\n\n')
            for b in blks:
                if 'MlasPool' in b:
                    mat = re.search('KernelShape: ([0-9]+), ([0-9]+)', b)
                    kernel = int(mat.group(1))
                    mat = re.search('StrideShape: ([0-9]+), ([0-9]+)', b)
                    stride = int(mat.group(1))
                    pool_list.append([kernel, stride])
        return pool_list
    def get_conv_list(log_path: str):
        conv_list = []
        with open(log_path, 'r') as f:
            log = f.read()
            blks = log.split('\n\n')
            for b in blks:
                if 'MlasConv' in b:
                    mat = re.search('output channels: ([0-9]+)', b)
                    out_channels = int(mat.group(1))
                    mat = re.search('dimensions: ([0-9]+)\ninput', b)
                    dims = int(mat.group(1))
                    mat = re.search('kernel shape: ([0-9]+), ([0-9]+)', b)
                    ker_shape = int(mat.group(1))

                    mat = re.search('padding: ([0-9]+), ([0-9]+)', b)
                    padding = int(mat.group(1))
                    mat = re.search('stride shape: ([0-9]+), ([0-9]+)', b)
                    stride = int(mat.group(1))
                    conv_list.append([(out_channels, dims, ker_shape, ker_shape), padding, stride])
        return conv_list
    def get_gemm_list(log_path: str):
        gemm_list = []
        with open(log_path, 'r') as f:
            log = f.read()
            blks = log.split('\n\n')
            for b in blks:
                if 'MlasGemm' in b:
                    mat = re.search('output: ([0-9]+)', b)
                    out_shape = int(mat.group(1))
                    mat = re.search('input: ([0-9]+)', b)
                    in_shape = int(mat.group(1))
                    gemm_list.append([in_shape, out_shape])
        return gemm_list
    pool_addr_list = [runtime_addr(addr_dict['MlasPool_addr'])]
    pool_log_path = './pool_info.log'
    pool_log_path = os.path.abspath(pool_log_path)
    nnfusion_pool(prog_path, data_path, pool_addr_list, pool_log_path)
    pool_list = get_pool_list(pool_log_path)
    # return
    gemm_addr_list = [runtime_addr(addr_dict['MlasGemm_addr'])]
    gemm_log_path = './gemm_info.log'
    gemm_log_path = os.path.abspath(gemm_log_path)
    nnfusion_gemm(prog_path, data_path, gemm_addr_list, gemm_log_path)
    gemm_list = get_gemm_list(gemm_log_path)
    # return
    conv_addr_list = [runtime_addr(addr_dict['MlasConvPrepare_addr'])]
    conv_log_path = './conv_info.log'
    conv_log_path = os.path.abspath(conv_log_path)
    nnfusion_conv(prog_path, data_path, conv_addr_list, conv_log_path)
    conv_list = get_conv_list(conv_log_path)


def get_func_trace(op_list: list):
    fun_addr_list = []
    for fun_name in op_list:
        fun_name = os.path.basename(fun_name)
        fun_addr = fun_name.split('_')[1]
        fun_addr = fun_addr.split('.')[0]
        fun_addr = '0x' + fun_addr
        if fun_addr not in fun_addr_list:
            fun_addr_list.append(runtime_addr(fun_addr))
    trace_log_path = './func_trace.log'
    trace_log_path = os.path.abspath(trace_log_path)
    nnfusion_trace(prog_path, data_path, fun_addr_list, trace_log_path)


def extract_param(overall_list: list):
    # human-in-the-loop PoC
    func_meta_data = [['0046.sub_404F50.txt', ('64', '2', '3', '3'), '0x404f50', 'conv2d'], 
                      ['0075.sub_408A70.txt', (1, '64'), '0x408a70', 'add'], 
                      [], [], 
                      ['0047.sub_405110.txt', ('128', '2', '3', '3'), '0x405110', 'conv2d'], 
                      ['0071.sub_408230.txt', (1, '128'), '0x408230', 'add'], 
                      [], [], 
                      ['0039.sub_404860.txt', ('256', '2', '3', '3'), '0x404860', 'conv2d'], 
                      ['0055.sub_405C10.txt', (1, '256'), '0x405c10', 'add'], 
                      [], 
                      ['0049.sub_4053B0.txt', ('256', '2', '3', '3'), '0x4053b0', 'conv2d'], 
                      ['0055.sub_405C10.txt', (1, '256'), '0x405c10', 'add'], 
                      [], [], 
                      ['0044.sub_404CB0.txt', ('512', '2', '3', '3'), '0x404cb0', 'conv2d'], 
                      ['0084.sub_409DB0.txt', (1, '512'), '0x409db0', 'add'], 
                      [], 
                      ['0043.sub_404AE0.txt', ('512', '2', '3', '3'), '0x404ae0', 'conv2d'], 
                      ['0084.sub_409DB0.txt', (1, '512'), '0x409db0', 'add'], 
                      [], [], 
                      ['0069.sub_407DB0.txt', ('512', '2', '3', '3'), '0x407db0', 'conv2d'], 
                      ['0053.sub_405930.txt', (1, '512'), '0x405930', 'add'], 
                      [], 
                      ['0069.sub_407DB0.txt', ('512', '2', '3', '3'), '0x407db0', 'conv2d'], 
                      ['0053.sub_405930.txt', (1, '512'), '0x405930', 'add'], 
                      [], [], 
                      ['0042.sub_404A90.txt', ['25088', '4096'], '0x404a90', 'dense'], 
                      ['0058.sub_4062E0.txt', (1, '4096'), '0x4062e0', 'add'], 
                      ['0040.sub_404A30.txt', ['4096', '4096'], '0x404a30', 'dense'], 
                      ['0058.sub_4062E0.txt', (1, '4096'), '0x4062e0', 'add'], 
                      ['0038.sub_404810.txt', ['4096', '1001'], '0x404810', 'dense'], 
                      ['0052.sub_4057C0.txt', (1, '1001'), '0x4057c0', 'add']
                      ]

    func_meta_data = [ [] for i in range(len(overall_list))]
    pool_idx = conv_idx = 0
    gemm_idx = len(gemm_list) - 1
    for i in range(len(overall_list)):
        f_name, op_type = overall_list[i]
        start_addr, end_addr = utils.get_func_range(os.path.join(funcs_dir, f_name))
        if op_type == 'Conv':
            func_meta_data[i] = [f_name, conv_list[conv_idx][0], start_addr, 'conv2d']
            conv_idx += 1
        elif 'Add' in op_type and 'Conv' in overall_list[i-1][1]:
            func_meta_data[i] = [f_name, (1, conv_list[conv_idx-1][0][0]), start_addr, 'add']
        
    for i in range(len(overall_list)-1, -1, -1):
        f_name, op_type = overall_list[i]
        start_addr, end_addr = utils.get_func_range(os.path.join(funcs_dir, f_name))
        if op_type == 'Dense':
            func_meta_data[i] = [f_name, gemm_list[gemm_idx], start_addr, 'dense']
            gemm_idx -= 1
        elif 'Add' in op_type and 'Dense' in overall_list[i-1][1]:
            func_meta_data[i] = [f_name, (1, gemm_list[gemm_idx][1]), start_addr, 'add']

    print(func_meta_data)

    in_data = data_path
    mem_dump_log_path = './mem_dump.log'
    mem_dump_log_path = os.path.abspath(mem_dump_log_path)
    utils.funcs_dir = funcs_dir
    for fun_data in func_meta_data:
        if len(fun_data) == 0:
            continue
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        dump_point = runtime_addr(dump_point)
        func_type = fun_data[3]
        logger.info('Extract Params for {}'.format(func_name))
        if func_type == 'conv2d':
            # w_shape = (w_shape[0], w_shape[2], w_shape[3], w_shape[1])
            utils.extract_params_nnfusion(prog_path, in_data, w_shape, dump_point,
                                          mem_dump_log_path, func_name, 1)
        elif func_type == 'add':
            utils.extract_params_nnfusion(prog_path, in_data, w_shape, dump_point,
                                          mem_dump_log_path, func_name, 1)
        elif func_type == 'dense':
            utils.extract_params_nnfusion(prog_path, in_data, w_shape, dump_point,
                                          mem_dump_log_path, func_name, reg_num=1)  #rdi, 0->rsi, 1->rdx, 2->rcx


# def read_param():
#     mem_dump_log_path = './mem_dump.log'
#     constatn_folder = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion/Constant/"
#     func_meta_data = [
#         ('Constant_17_0.bin', (64, 3, 3, 3), (3, 3, 3, 64), 'conv2d'),
#         ('Constant_16_0.bin', (1, 64), 'add'),
#         ('Constant_19_0.bin', (128, 64, 3, 3), (3, 3, 64, 128), 'conv2d'),
#         ('Constant_18_0.bin', (1, 128), 'add'),
#         ('Constant_21_0.bin', (256, 128, 3, 3), (3, 3, 128, 256), 'conv2d'),
#         ('Constant_20_0.bin', (1, 256), 'add'),
#         ('Constant_23_0.bin', (256, 256, 3, 3), (3, 3, 256, 256), 'conv2d'),
#         ('Constant_22_0.bin', (1, 256), 'add'),
#         ('Constant_25_0.bin', (512, 256, 3, 3), (3, 3, 256, 512), 'conv2d'),
#         ('Constant_24_0.bin', (1, 512), 'add'),
#         ('Constant_27_0.bin', (512, 512, 3, 3), (3, 3, 512, 512), 'conv2d'),
#         ('Constant_26_0.bin', (1, 512), 'add'),
#         ('Constant_29_0.bin', (512, 512, 3, 3), (3, 3, 512, 512), 'conv2d'),
#         ('Constant_28_0.bin', (1, 512),  'add'),
#         ('Constant_31_0.bin', (512, 512, 3, 3), (3, 3, 512, 512), 'conv2d'),
#         ('Constant_30_0.bin', (1, 512), 'add'),

#         ('Constant_10_0.bin', (25088, 4096), '0x5420', 'dense'),
#         ('Constant_13_0.bin', (1, 4096), '0x52f0', 'add'),
#         ('Constant_11_0.bin', (4096, 4096), '0x4f30', 'dense'),
#         ('Constant_14_0.bin', (1, 4096), '0x52f0', 'add'),
#         ('Constant_12_0.bin', (4096, 1001), '0x4d10', 'dense'),
#         ('Constant_15_0.bin', (1, 1001), '0x85f0', 'add'),

#     ]
#     for func in func_meta_data:
#         print(func[0])
#         constant_path = os.path.join(constatn_folder, func[0])
#         w_shape = func[1]
#         func_type = func[-1]
#         if func_type == 'conv2d':
#             org_shape = func[-2]
#         float_len = 1
#         for w_l in w_shape:
#             float_len *= w_l
#         json_path = constant_path[:-4] + '.json'
#         convert_constant_to_txt(constant_path, mem_dump_log_path, float_len)
#         float_array = convert_txt_to_float(mem_dump_log_path, float_len)
#         w = np.asarray(float_array)
#         if func_type == 'conv2d':
#             w = w.reshape(org_shape)
#             w = w.transpose(3, 2, 0, 1)
#         w = w.reshape(w_shape)
#         lists = w.tolist()
#         json_str = json.dumps(lists)
#         json_str = json_str.replace('],', '],\n')
#         with open(json_path, 'w') as wf:
#             wf.write(json_str)
#             wf.close()



def convert_constant_to_txt(constant_path: str, txt_path: str, float_len: int):
    constant_path = os.path.abspath(constant_path)
    txt_path = os.path.abspath(txt_path)
    with open(constant_path, 'rb') as f:
        with open(txt_path, 'w') as w:

            while float_len > 0:
                float_bytes = f.read(4)
                # print('0x'+float_bytes[::-1].hex()+'\n')
                w.write('0x'+float_bytes[::-1].hex()+'\n')
                float_len -= 1
            w.close()
        f.close()


def convert_txt_to_float(txt_path: str, float_len: int):
    txt_path = os.path.abspath(txt_path)
    with open(txt_path, 'r') as f:
        dwords_txt = f.read()
        float_array = convert_dwords2float(dwords_txt, float_len)
        return float_array


if __name__ == '__main__':
    #test
    # read_param()
    # exit(0)
    # ------------------
    
    # ------------------
    
    # Step 1
    get_shape_info()  # get the shape information of conv, pool and gemm
    # ------------------

    # Step 2
    operator_list = get_all_operator()  # the list of operaotrs
    # print('operator_list: ', operator_list)
    get_func_trace(operator_list)  # log the tracec of operator
    # ------------------

    # Step 3
    overall_list = []
    for op_func_path in operator_list:  # identify operators
       print('\n{}'.format(op_func_path))
       if op_func_path:
            callee_type = identify_operator(op_func_path)
            if callee_type:
                overall_list.append(callee_type)
    # ------------------

    # Step 4 Extract Parameters
    extract_param(overall_list)
    
