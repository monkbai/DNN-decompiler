import os
import sys
sys.path.append("../..")
from scripts.pin_tools import nnfusion_conv, nnfusion_gemm, nnfusion_pool, nnfusion_trace
from scripts import utils

#
# Find some special function addresses
# Automatically or manually
addr_dict = {
    #
    'base_addr': '0x555555554000',
    'kernel_entry_addr': '0xcc90',
    'MlasConvPrepare_addr': '0x165e0',
    'MlasConv_addr': '0x15dd0',
    'MlasGemm_addr': '0x13fc0',
    'MlasPool_addr': '0x16b10',
    'concurrency_addr': '0x4a360',
    #
    'Broadcast': '0x11e70',
    'Reshape': '0x103c0',
}

prog_path = '/home/lifter/Documents/tvm_output/vgg11_nnfusion'
data_path = '/home/lifter/Documents/tvm_output/cat.bin'

#funcs_dir = '/home/lifter/Documents/tvm_output/vgg11_strip_nn_funcs'
funcs_dir = '/home/lifter/Documents/tvm_output/vgg11_nnfusion_funcs'


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
                      ('0068.sub_78E0.txt', (64, 3, 3, 3), '0x78e0', 'conv2d'),
                      ('0055.sub_5A90.txt', (1, 64), '0x5a90', 'add'),
                      ('0070.sub_7D50.txt', (128, 64, 3, 3), '0x7d50', 'conv2d'),
                      ('0054.sub_5920.txt', (1, 128), '0x5920', 'add'),
                      ('0046.sub_4F80.txt', (256, 128, 3, 3), '0x4f80', 'conv2d'),
                      ('0052.sub_5640.txt', (1, 256), '0x5640', 'add'),
                      ('0057.sub_5D70.txt', (256, 256, 3, 3), '0x5d70', 'conv2d'),
                      # ('0052.sub_5640.txt', (1, 256), '0x5640', 'add'),
                      ('0081.sub_94E0.txt', (512, 256, 3, 3), '0x94e0', 'conv2d'),
                      ('0066.sub_74C0.txt', (1, 512), '0x74c0', 'add'),
                      ('0051.sub_5470.txt', (512, 512, 3, 3), '0x5470', 'conv2d'),
                      # ('0066.sub_74C0.txt', (1, 512), '0x74c0', 'add'),
                      ('0044.sub_4D60.txt', (512, 512, 3, 3), '0x4d60', 'conv2d'),
                      ('0056.sub_5C00.txt', (1, 512), '0x5c00', 'add'),
                      # ('0044.sub_4D60.txt', (512, 512, 3, 3), '0x4d60', 'conv2d'),
                      # ('0056.sub_5C00.txt', (1, 512), '0x5c00', 'add'),

                      ('0050.sub_5420.txt', (25088, 4096), '0x5420', 'dense'),
                      ('0049.sub_52F0.txt', (1, 4096), '0x52f0', 'add'),
                      ('0045.sub_4F30.txt', (4096, 4096), '0x4f30', 'dense'),
                      # ('0049.sub_52F0.txt', (1, 4096), '0x52f0', 'add'),
                      ('0043.sub_4D10.txt', (4096, 1001), '0x4d10', 'dense'),
                      ('0074.sub_85F0.txt', (1, 1001), '0x85f0', 'add'),

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

        if func_name.startswith('0050.'):  # huge file, run with server
            continue
        elif 'conv2d' not in func_type:
            continue

        if func_type == 'conv2d':
            w_shape = (w_shape[0], w_shape[2], w_shape[3], w_shape[1])
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1)
        elif func_type == 'add':
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, 1)
        elif func_type == 'dense':
            utils.extract_params_glow(prog_path, in_data, w_shape, dump_point,
                                      mem_dump_log_path, func_name, reg_num=1)  # 0->rsi, 1->rdx, 2->rcx


if __name__ == '__main__':
    extract_param()
    exit(0)
    # ------------------
    #get_shape_info()

    operator_list = get_all_operator()
    # print(operator_list)
    #get_func_trace(operator_list)

    for op_func_path in operator_list:
        print('\n{}'.format(op_func_path))
        if op_func_path:
            identify_operator(op_func_path)


