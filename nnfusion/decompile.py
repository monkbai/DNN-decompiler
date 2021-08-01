#!/usr/bin/python3
import os
import sys
sys.path.append("../..")
from scripts.pin_tools import nnfusion_conv, nnfusion_gemm, nnfusion_pool, nnfusion_trace, convert_dwords2float
from scripts import utils
import numpy as np
import json
import logging
print('get logger: {}'.format('decompiler.'+__name__))
logger = logging.getLogger('decompiler.'+__name__)

#
# Find some special function addresses
# Automatically or manually
addr_dict = {
    #
    'base_addr': '0x0',  # recompiled with -no-pie
    'kernel_entry_addr': '0x40BE00',
    'MlasConvPrepare_addr': '0x415750',
    'MlasConv_addr': '0x414F40',
    'MlasGemm_addr': '0x413130',
    'MlasPool_addr': '0x415C80',
    'concurrency_addr': '0x4494d0',
    #
    'Broadcast': '0x410Fe0',
    'Reshape': '0x40F530',
}

prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion/vgg_nnfusion"
data_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion/cat.bin"


funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion/vgg_nnfusion_strip_funcs/"


def runtime_addr(addr: str):
    return hex(int(addr_dict['base_addr'], 16) + int(addr, 16))


def get_dict_name(func_offset: str):
    for name, offset in addr_dict.items():
        if int(offset, 16) == int(func_offset, 16):
            return name


def get_func_name(func_addr: str):
    if func_addr.startswith('0x'):
        func_addr = func_addr[2:]
    files = os.listdir(funcs_dir)
    func_addr = func_addr.upper()
    func_name = '_{}.txt'.format(func_addr)
    for file in files:
        if file.endswith(func_name):
            return os.path.join(funcs_dir, file)


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
            callee_type = 'unknown'
        if 'concurrency' in callee_type:
            func_ptr_list, asm_path_list = get_concurrency_func_ptr(func_name)
            print(func_ptr_list)
            callee_type += ' --> ' + simple_label(asm_path_list[0])
        # final label
        print('predict the operator type: {}'.format(callee_type))
        print(callee_addr_list)


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


def get_shape_info():
    pool_addr_list = [runtime_addr(addr_dict['MlasPool_addr'])]
    pool_log_path = './pool_info.log'
    pool_log_path = os.path.abspath(pool_log_path)
    nnfusion_pool(prog_path, data_path, pool_addr_list, pool_log_path)
    # return
    gemm_addr_list = [runtime_addr(addr_dict['MlasGemm_addr'])]
    gemm_log_path = './gemm_info.log'
    gemm_log_path = os.path.abspath(gemm_log_path)
    nnfusion_gemm(prog_path, data_path, gemm_addr_list, gemm_log_path)
    # return
    conv_addr_list = [runtime_addr(addr_dict['MlasConvPrepare_addr'])]
    conv_log_path = './conv_info.log'
    conv_log_path = os.path.abspath(conv_log_path)
    nnfusion_conv(prog_path, data_path, conv_addr_list, conv_log_path)


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


def extract_param():
    func_meta_data = [
                      ('0068.sub_406A50.txt', (64, 3, 3, 3), '0x406A50', 'conv2d'),
                      ('0055.sub_404C00.txt', (1, 64), '0x404C00', 'add'),
                      ('0070.sub_406Ec0.txt', (128, 64, 3, 3), '0x406Ec0', 'conv2d'),
                      ('0054.sub_404A90.txt', (1, 128), '0x404A90', 'add'),
                      ('0046.sub_4040f0.txt', (256, 128, 3, 3), '0x4040f0', 'conv2d'),
                      ('0052.sub_4047b0.txt', (1, 256), '0x4047b0', 'add'),
                      ('0057.sub_404Ee0.txt', (256, 256, 3, 3), '0x404Ee0', 'conv2d'),
                      # ('0052.sub_4047b0.txt', (1, 256), '0x4047b0', 'add'),
                      ('0081.sub_408650.txt', (512, 256, 3, 3), '0x408650', 'conv2d'),
                      ('0066.sub_406630.txt', (1, 512), '0x406630', 'add'),
                      ('0051.sub_4045e0.txt', (512, 512, 3, 3), '0x4045e0', 'conv2d'),
                      # ('0066.sub_406630.txt', (1, 512), '0x406630', 'add'),
                      ('0044.sub_403Ed0.txt', (512, 512, 3, 3), '0x403Ed0', 'conv2d'),
                      ('0056.sub_404D70.txt', (1, 512), '0x404D70', 'add'),
                      # ('0044.sub_403Ed0.txt', (512, 512, 3, 3), '0x403Ed0', 'conv2d'),
                      # ('0056.sub_404D70.txt', (1, 512), '0x404D70', 'add'),

                      ('0050.sub_404590.txt', (25088, 4096), '0x404590', 'dense'),
                      ('0049.sub_404460.txt', (1, 4096), '0x404460', 'add'),
                      ('0045.sub_4040A0.txt', (4096, 4096), '0x4040a0', 'dense'),
                      # ('0049.sub_404460.txt', (1, 4096), '0x404460', 'add'),
                      ('0043.sub_403E80.txt', (4096, 1001), '0x403E80', 'dense'),
                      ('0074.sub_407760.txt', (1, 1001), '0x407760', 'add'),

                      ]
    in_data = data_path
    mem_dump_log_path = './mem_dump.log'
    mem_dump_log_path = os.path.abspath(mem_dump_log_path)
    utils.funcs_dir = funcs_dir
    for fun_data in func_meta_data:
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


def read_param():
    mem_dump_log_path = './mem_dump.log'
    constatn_folder = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion/Constant/"
    func_meta_data = [
        ('Constant_17_0.bin', (64, 3, 3, 3), (3, 3, 3, 64), 'conv2d'),
        ('Constant_16_0.bin', (1, 64), 'add'),
        ('Constant_19_0.bin', (128, 64, 3, 3), (3, 3, 64, 128), 'conv2d'),
        ('Constant_18_0.bin', (1, 128), 'add'),
        ('Constant_21_0.bin', (256, 128, 3, 3), (3, 3, 128, 256), 'conv2d'),
        ('Constant_20_0.bin', (1, 256), 'add'),
        ('Constant_23_0.bin', (256, 256, 3, 3), (3, 3, 256, 256), 'conv2d'),
        ('Constant_22_0.bin', (1, 256), 'add'),
        ('Constant_25_0.bin', (512, 256, 3, 3), (3, 3, 256, 512), 'conv2d'),
        ('Constant_24_0.bin', (1, 512), 'add'),
        ('Constant_27_0.bin', (512, 512, 3, 3), (3, 3, 512, 512), 'conv2d'),
        ('Constant_26_0.bin', (1, 512), 'add'),
        ('Constant_29_0.bin', (512, 512, 3, 3), (3, 3, 512, 512), 'conv2d'),
        ('Constant_28_0.bin', (1, 512),  'add'),
        ('Constant_31_0.bin', (512, 512, 3, 3), (3, 3, 512, 512), 'conv2d'),
        ('Constant_30_0.bin', (1, 512), 'add'),

        ('Constant_10_0.bin', (25088, 4096), '0x5420', 'dense'),
        ('Constant_13_0.bin', (1, 4096), '0x52f0', 'add'),
        ('Constant_11_0.bin', (4096, 4096), '0x4f30', 'dense'),
        ('Constant_14_0.bin', (1, 4096), '0x52f0', 'add'),
        ('Constant_12_0.bin', (4096, 1001), '0x4d10', 'dense'),
        ('Constant_15_0.bin', (1, 1001), '0x85f0', 'add'),

    ]
    for func in func_meta_data:
        print(func[0])
        constant_path = os.path.join(constatn_folder, func[0])
        w_shape = func[1]
        func_type = func[-1]
        if func_type == 'conv2d':
            org_shape = func[-2]
        float_len = 1
        for w_l in w_shape:
            float_len *= w_l
        json_path = constant_path[:-4] + '.json'
        convert_constant_to_txt(constant_path, mem_dump_log_path, float_len)
        float_array = convert_txt_to_float(mem_dump_log_path, float_len)
        w = np.asarray(float_array)
        if func_type == 'conv2d':
            w = w.reshape(org_shape)
            w = w.transpose(3, 2, 0, 1)
        w = w.reshape(w_shape)
        lists = w.tolist()
        json_str = json.dumps(lists)
        json_str = json_str.replace('],', '],\n')
        with open(json_path, 'w') as wf:
            wf.write(json_str)
            wf.close()



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
    read_param()
    exit(0)
    # ------------------
    
    # ------------------
    
    # Step 1
    #get_shape_info()  # get the shape information of conv, pool and gemm
    # ------------------

    # Step 2
    #operator_list = get_all_operator()  # the list of operaotrs
    # print(operator_list)
    #get_func_trace(operator_list)  # log the tracec of operator
    # ------------------

    # Step 3
    #for op_func_path in operator_list:  # identify operators
    #    print('\n{}'.format(op_func_path))
    #    if op_func_path:
    #        identify_operator(op_func_path)
    # ------------------

    # Step 4 Extract Parameters
    #extract_param()
    
