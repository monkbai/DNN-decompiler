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

# ====================================================
def dict_to_json(dict_obj: dict, output_path: str):
    j = json.dumps(dict_obj)
    with open(output_path, 'w') as f:
        f.write(j)


def json_to_dict(json_path: str):
    if not os.path.exists(json_path):
        return dict()
    with open(json_path, 'r') as f:
        j_txt = f.read()
        dict_obj = json.loads(s=j_txt)
        return dict_obj

# ====================================================

#
# Find some special function addresses
# Automatically or manually
addr_dict = {
    #
    'base_addr': '0x0',  # recompiled with -no-pie
    'kernel_entry_addr': '0x417FE0',
    'MlasConvPrepare_addr': '0x424290',
    'MlasConv_addr': '0x423A80',
    'MlasGemm_addr': '0x421C70',
    'MlasPool_addr': '0x4247C0',
    'concurrency_addr': '0x458010',
    #
    'Broadcast': '0x41EA80',
    'Reshape': '0x41D0B0',
    'BatchNorm': '0x41CCD0',
    'Sum': '0x41E490',
    # 
    'Eigen_parallel': '0x41B330',	
    #
    'GetRawThreadPool': '0x45C0B0'
}

prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion_resnet/resnet50_nnfusion"
data_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion_resnet/cat1.bin"


funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion_resnet/resnet50_nnfusion_strip_funcs/"


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
                    elif 'BatchNorm' in callee_name:
                        callee_type = 'BatchNorm'
                    elif 'Sum' in callee_name:
                        callee_type = 'Sum'
                    elif 'Eigen' in callee_name:
                        callee_type = 'Pad'
                    else:
                        pass
                        # callee_type = 'unknown'
        if len(callee_type) == 0:
            callee_type = 'ignored'
        if 'concurrency' in callee_type:
            func_ptr_list, asm_path_list = get_concurrency_func_ptr(func_name)
            print(func_ptr_list)
            callee_type += ' --> ' + simple_label(asm_path_list[0])
        # final label
        print('predict the operator type: {}'.format(callee_type))
        print(callee_addr_list)
        return callee_type


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
        if asm_txt.find('vdivps') != -1:
            label += 'Divide '    
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
    # func_meta_data come from previous output
    func_meta_data = [
                      ('0045.sub_403DB0.txt', (64, 3, 7, 7), 'conv2d'),
                      ('0343.sub_415DA0.txt', (1, 64), 'BatchNorm'),
                      ('0130.sub_408F70.txt', (64, 64, 1, 1), 'conv2d'),
                      ('0346.sub_416010.txt', (1, 64), 'BatchNorm'),
                      ('0055.sub_404F50.txt', (64, 64, 3, 3), 'conv2d'),
                      #('0346.sub_416010.txt', (1, 64), 'BatchNorm'),  # 1
                      ('0057.sub_4052D0.txt', (256, 64, 1, 1), 'conv2d'),
                      ('0344.sub_415E70.txt', (1, 256), 'BatchNorm'),
                      #('0057.sub_4052D0.txt', (256, 64, 1, 1), 'conv2d'),  # 1
                      #('0344.sub_415E70.txt', (1, 256), 'BatchNorm'),  # 1
                      ('0052.sub_404A00.txt', (64, 256, 1, 1), 'conv2d'),
                      #('0346.sub_416010.txt', (1, 64), 'BatchNorm'),  # 2
                      #('0055.sub_404F50.txt', (64, 64, 3, 3), 'conv2d'),  # 1
                      #('0346.sub_416010.txt', (1, 64), 'BatchNorm'),  # 3
                      #('0057.sub_4052D0.txt', (256, 64, 1, 1), 'conv2d'),  # 2
                      #('0344.sub_415E70.txt', (1, 256), 'BatchNorm'),  # 2
                      #('0052.sub_404A00.txt', (64, 256, 1, 1), 'conv2d'),  # 1
                      #('0346.sub_416010.txt', (1, 64), 'BatchNorm'),  # 4
                      #('0055.sub_404F50.txt', (64, 64, 3, 3), 'conv2d'),  # 2
                      #('0346.sub_416010.txt', (1, 64), 'BatchNorm'),  # 5
                      #('0057.sub_4052D0.txt', (256, 64, 1, 1), 'conv2d'),  # 3
                      #('0344.sub_415E70.txt', (1, 256), 'BatchNorm'),  # 3
                      
                      ('0053.sub_404BC0.txt', (128, 256, 1, 1), 'conv2d'),
                      ('0347.sub_4160E0.txt', (1, 128), 'BatchNorm'),
                      ('0060.sub_405810.txt', (128, 128, 3, 3), 'conv2d'),
                      #('0347.sub_4160E0.txt', (1, 128), 'BatchNorm'),  # 1
                      ('0056.sub_405110.txt', (512, 128, 1, 1), 'conv2d'),
                      ('0348.sub_4161B0.txt', (1, 512), 'BatchNorm'),
                      ('0058.sub_405490.txt', (512, 256, 1, 1), 'conv2d'),
                      #('0348.sub_4161B0.txt', (1, 512), 'BatchNorm'),  # 1
                      ('0042.sub_403870.txt', (128, 512, 1, 1), 'conv2d'),
                      #('0347.sub_4160E0.txt', (1, 128), 'BatchNorm'),  # 2
                      #('0060.sub_405810.txt', (128, 128, 3, 3), 'conv2d'),  # 1
                      #('0347.sub_4160E0.txt', (1, 128), 'BatchNorm'),  # 3
                      #('0056.sub_405110.txt', (512, 128, 1, 1), 'conv2d'),  # 1
                      #('0348.sub_4161B0.txt', (1, 512), 'BatchNorm'),  # 2
                      #('0042.sub_403870.txt', (128, 512, 1, 1), 'conv2d'),  # 1
                      #('0347.sub_4160E0.txt', (1, 128), 'BatchNorm'),  # 4
                      #('0060.sub_405810.txt', (128, 128, 3, 3), 'conv2d'),  # 2
                      #('0347.sub_4160E0.txt', (1, 128), 'BatchNorm'),  # 5
                      #('0056.sub_405110.txt', (512, 128, 1, 1), 'conv2d'),  # 2
                      #('0348.sub_4161B0.txt', (1, 512), 'BatchNorm'),  # 3
                      #('0042.sub_403870.txt', (128, 512, 1, 1), 'conv2d'),  # 2
                      #('0347.sub_4160E0.txt', (1, 128), 'BatchNorm'),  # 6
                      #('0060.sub_405810.txt', (128, 128, 3, 3), 'conv2d'),  # 3
                      #('0347.sub_4160E0.txt', (1, 128), 'BatchNorm'),  # 7
                      #('0056.sub_405110.txt', (512, 128, 1, 1), 'conv2d'),  # 3
                      #('0348.sub_4161B0.txt', (1, 512), 'BatchNorm'),  # 4
                      
                      ('0059.sub_405650.txt', (256, 512, 1, 1), 'conv2d'),
                      ('0349.sub_416280.txt', (1, 256), 'BatchNorm'),
                      ('0050.sub_404670.txt', (256, 256, 3, 3), 'conv2d'),
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 1
                      ('0043.sub_403A30.txt', (1024, 256, 1, 1), 'conv2d'),
                      ('0342.sub_415CD0.txt', (1, 1024), 'BatchNorm'),
                      ('0044.sub_403BF0.txt', (1024, 512, 1, 1), 'conv2d'),
                      #('0342.sub_415CD0.txt', (1, 1024), 'BatchNorm'),  # 1
                      ('0048.sub_4042F0.txt', (256, 1024, 1, 1), 'conv2d'),
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 2
                      #('0050.sub_404670.txt', (256, 256, 3, 3), 'conv2d'),  # 1
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 3
                      #('0043.sub_403A30.txt', (1024, 256, 1, 1), 'conv2d'),  # 1
                      #('0342.sub_415CD0.txt', (1, 1024), 'BatchNorm'),  # 2
                      #('0048.sub_4042F0.txt', (256, 1024, 1, 1), 'conv2d'),  # 1
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 4
                      #('0050.sub_404670.txt', (256, 256, 3, 3), 'conv2d'),  # 2
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 5
                      #('0043.sub_403A30.txt', (1024, 256, 1, 1), 'conv2d'),  # 2
                      #('0342.sub_415CD0.txt', (1, 1024), 'BatchNorm'),  # 3
                      #('0048.sub_4042F0.txt', (256, 1024, 1, 1), 'conv2d'),  # 2
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 6
                      #('0050.sub_404670.txt', (256, 256, 3, 3), 'conv2d'),  # 3
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 7
                      #('0043.sub_403A30.txt', (1024, 256, 1, 1), 'conv2d'),  # 3
                      #('0342.sub_415CD0.txt', (1, 1024), 'BatchNorm'),  # 4
                      #('0048.sub_4042F0.txt', (256, 1024, 1, 1), 'conv2d'),  # 3
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 8
                      #('0050.sub_404670.txt', (256, 256, 3, 3), 'conv2d'),  # 4
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 9
                      #('0043.sub_403A30.txt', (1024, 256, 1, 1), 'conv2d'),  # 4
                      #('0342.sub_415CD0.txt', (1, 1024), 'BatchNorm'),  # 5
                      #('0048.sub_4042F0.txt', (256, 1024, 1, 1), 'conv2d'),  # 4
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 10
                      #('0050.sub_404670.txt', (256, 256, 3, 3), 'conv2d'),  # 5
                      #('0349.sub_416280.txt', (1, 256), 'BatchNorm'),  # 11
                      #('0043.sub_403A30.txt', (1024, 256, 1, 1), 'conv2d'),  # 5
                      #('0342.sub_415CD0.txt', (1, 1024), 'BatchNorm'),  # 6
                      
                      ('0051.sub_404840.txt', (512, 1024, 1, 1), 'conv2d'),
                      ('0341.sub_415C00.txt', (1, 512), 'BatchNorm'),
                      ('0054.sub_404D80.txt', (512, 512, 3, 3), 'conv2d'),
                      #('0341.sub_415C00.txt', (1, 512), 'BatchNorm'),  # 1
                      ('0049.sub_4044B0.txt', (2048, 512, 1, 1), 'conv2d'),
                      ('0345.sub_415F40.txt', (1, 2048), 'BatchNorm'),
                      ('0047.sub_404130.txt', (2048, 1024, 1, 1), 'conv2d'),
                      #('0345.sub_415F40.txt', (1, 2048), 'BatchNorm'),  # 1
                      ('0046.sub_403F70.txt', (512, 2048, 1, 1), 'conv2d'),
                      #('0341.sub_415C00.txt', (1, 512), 'BatchNorm'),  # 2
                      #('0054.sub_404D80.txt', (512, 512, 3, 3), 'conv2d'),  # 1
                      #('0341.sub_415C00.txt', (1, 512), 'BatchNorm'),  # 3
                      #('0049.sub_4044B0.txt', (2048, 512, 1, 1), 'conv2d'),  # 1
                      #('0345.sub_415F40.txt', (1, 2048), 'BatchNorm'),  # 2
                      #('0046.sub_403F70.txt', (512, 2048, 1, 1), 'conv2d'),  # 1
                      #('0341.sub_415C00.txt', (1, 512), 'BatchNorm'),  # 4
                      #('0054.sub_404D80.txt', (512, 512, 3, 3), 'conv2d'),  # 2
                      #('0341.sub_415C00.txt', (1, 512), 'BatchNorm'),  # 5
                      #('0049.sub_4044B0.txt', (2048, 512, 1, 1), 'conv2d'),  # 2
                      #('0345.sub_415F40.txt', (1, 2048), 'BatchNorm'),  # 3
                      

                      ('0257.sub_40EF30.txt', (2048, 1001), 'dense'),
                      ('0114.sub_4082C0.txt', (1, 1001), 'add'),
                      ]
    in_data = data_path
    mem_dump_log_path = './mem_dump.log'
    mem_dump_log_path = os.path.abspath(mem_dump_log_path)
    utils.funcs_dir = funcs_dir
    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        func_type = fun_data[2]
        dump_point = func_name.split('.')[1]
        dump_point = dump_point.replace('sub_', '0x')
        print('dump_point {}'.format(dump_point))
        logger.info('Extract Params for {}'.format(func_name))
        if func_type == 'conv2d':
            continue
            # w_shape = (w_shape[0], w_shape[2], w_shape[3], w_shape[1])
            utils.extract_params_nnfusion(prog_path, in_data, w_shape, dump_point,
                                          mem_dump_log_path, func_name, 1)
        elif func_type == 'add':
            continue
            utils.extract_params_nnfusion(prog_path, in_data, w_shape, dump_point,
                                          mem_dump_log_path, func_name, 1)
        elif func_type == 'dense':
            w_shape = (w_shape[1], w_shape[0])
            utils.extract_params_nnfusion(prog_path, in_data, w_shape, dump_point,
                                          mem_dump_log_path, func_name, reg_num=1) 
        elif func_type == 'BatchNorm':
            continue
            utils.extract_params_nnfusion(prog_path, in_data, w_shape, dump_point,
                                          mem_dump_log_path, func_name, reg_num=-1)  # gamma
            utils.extract_params_nnfusion(prog_path, in_data, w_shape, dump_point,
                                          mem_dump_log_path, func_name, reg_num=0)  # beta
            utils.extract_params_nnfusion(prog_path, in_data, w_shape, dump_point,  
                                          mem_dump_log_path, func_name, reg_num=2)  # mean
            utils.extract_params_nnfusion(prog_path, in_data, w_shape, dump_point,
                                          mem_dump_log_path, func_name, reg_num=3)  # var

'''
def read_param():
    mem_dump_log_path = './mem_dump.log'
    constatn_folder = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion/Constant/"
    func_meta_data = [
        ('Constant_17_0.bin', (64, 3, 3, 3), (3, 3, 3, 64), 'conv2d'),
        ('Constant_16_0.bin', (), 'add'),
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

        ('Constant_10_0.bin', (25088, 4096), 'dense'),
        ('Constant_13_0.bin', (1, 4096), 'add'),
        ('Constant_11_0.bin', (4096, 4096), 'dense'),
        ('Constant_14_0.bin', (1, 4096), 'add'),
        ('Constant_12_0.bin', (4096, 1001), 'dense'),
        ('Constant_15_0.bin', (1, 1001), 'add'),

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
'''


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
    #read_param()  # test
    #exit(0)
    # ------------------
    extract_param()
    exit(0)
    # ------------------
    
    # Step 1
    get_shape_info()  # get the shape information of conv, pool and gemm
    # ------------------

    # Step 2
    operator_list = get_all_operator()  # the list of operaotrs
    # print(operator_list)
    get_func_trace(operator_list)  # log the tracec of operator
    # ------------------

    addr2func = dict()
    # Step 3
    for op_func_path in operator_list:  # identify operators
        callee_addr = op_func_path[op_func_path.find('sub_') + 4:]
        callee_addr = callee_addr[:-4].strip()
        callee_name = get_dict_name(callee_addr)
        if callee_name and 'GetRawThread' in callee_name:
            continue  # can be ignored
        print('\n{}'.format(op_func_path))
        if op_func_path:
            callee_type = identify_operator(op_func_path)
            addr2func[int(callee_addr, 16)] = callee_type
    dict_to_json(addr2func, 'addr2func.json')
    # ------------------

    # Step 4 Extract Parameters
    #extract_param()
    
