#! /usr/bin/python3
import os
import sys
import json
sys.path.append("../..")
from scripts import trace_filter
from scripts import utils
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

funcs_dir = './resnet_O0_funcs/'
# ==============================================================
# Generate the function call trace
# ==============================================================


def get_addr_list(label_path: str, fused=False):
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

                addr2label[addr] = label.strip()
                addr2funcs[addr] = name.strip()
                if fused and 'fused' not in label:
                    continue
                elif not fused and 'fused' in label:
                    continue

                addr_list.append(addr)
    return addr_list


def get_funcs_trace(prog_path: str, in_data: str, log_path: str, label_file: str, only_fused=False):

    # prog_path = './vgg16_strip'
    prog_path = os.path.abspath(prog_path)
    # in_data = './cat.bin'
    in_data = os.path.abspath(in_data)
    # log_path = './vgg16_strip_func_call.log'
    log_path = os.path.abspath(log_path)
    # label_file = './step1.txt'
    label_file = os.path.abspath(label_file)

    addr_list = get_addr_list(label_file, fused=only_fused)
    # addr_list = '0x40d89a,0x42a324,0x42dcc0,0x417b3e,0x4221f5,0x420ee0,0x421b40,0x4213f5,0x42065a,0x40e400,0x41c11e,0x42953e,0x406bca,0x428d9a,0x418edf,0x42d740,0x42286a,0x41979a,0x41826a,0x41688e,0x420296,0x401e4e,0x4250f0,0x401720,0x42e224,0x41cd7a,0x420ac0,0x416330,0x402b4e,0x424d80,0x413f9a,0x40eafe,0x40383a,0x42de41,0x40f75a,0x429a40,0x429160,0x40746a,0x42dad2,0x429ad9,0x4127ea,0x42867a,0x425dba,0x417100,0x41f9ce,0x429c62,0x4011f4,0x40dd9e,0x42da02,0x42dbb0,0x4257f4,0x429cc0,0x417533,0x41346e,0x40a8ba'

    # tmp_log_path = os.path.abspath('./fused_data.log')
    fused_rdi(prog_path, in_data, addr_list, log_path)
    # func_call_trace(prog_path, in_data, addr_list, log_path)

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


def print_fused_trace(trace_log_path: str, new_log_path: str):
    global addr2label, addr2funcs
    addr2label = json_to_dict('./addr2label.json')
    addr2funcs = json_to_dict('./addr2funcs.json')

    addr2label_list = list(addr2label.items())
    addr2label_list = sorted(addr2label_list, key=lambda x: x[0])

    trace_log_path = os.path.abspath(trace_log_path)
    new_log_path = os.path.abspath(new_log_path)
    new_log = open(new_log_path, 'w')
    with open(trace_log_path, 'r') as f:
        trace_txt = f.read()
        lines = trace_txt.split('\n')
        for line in lines:
            if not line.startswith('0x'):
                continue
            addr = line.split(':')[0]
            addr = hex(int(addr.strip(), 16))
            if 'reshape' in addr2label[addr]:
                continue
            elif 'fused' in addr2label[addr]:
                idx = addr2label_list.index((addr, addr2label[addr]))
                new_log.write('{}, {: <18}'.format(addr2label_list[idx+1][0], addr2label_list[idx+1][1]) + ' - ')
                new_log.write(line + '\n')


def get_call_graph_list(trace_log_path: str):
    global addr2label, addr2funcs

    trace_log_path = os.path.abspath(trace_log_path)
    call_graph_list = []
    with open(trace_log_path, 'r') as f:
        trace_txt = f.read()
        lines = trace_txt.split('\n')
        for line in lines:
            if not line.startswith('0x'):
                continue
            addr_label = line.split('-')[0].strip()
            addr = addr_label.split(',')[0]
            label = addr_label.split(',')[1].strip()
            args = line.split(':')[1].strip()
            args = args.strip(',')
            args = args.split(',')
            call_graph_list.append(((addr, label), args))
        return call_graph_list


def show_graph(call_graph: list):
    dg = nx.DiGraph()
    for i in range(len(call_graph)):
        node, args = call_graph[i]
        dg.add_node(str((i, node[0], node[1])))

    for i in range(len(call_graph)):
        node, args = call_graph[i]
        addr = node[0]
        label = node[1]
        j = i + 1
        while j < len(call_graph):
            c_node, c_args = call_graph[j]
            in_c_args = c_args[:-1]
            if args[-1] in in_c_args:  # or j == i+1:
                dg.add_edge(str((i, addr, label)), str((j, c_node[0], c_node[1])))
            if args[-1] == c_args[-1]:
                break
            j += 1
    net = Network(notebook=True)
    net.height = 1000
    net.width = 1000
    net.from_nx(dg)
    net.show('tmp.html')
    # nx.draw_spectral(dg, with_labels=True)
    # plt.savefig("tmp.png")


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


def generate_symbolic_expression(func_name: str, inst_log_path: str, exp_log_path: str, max_inst=5000000):
    func_asm_path = os.path.join(funcs_dir, func_name)
    func_asm_path = os.path.abspath(func_asm_path)

    inst_log_path = os.path.abspath(inst_log_path)
    exp_log_path = os.path.abspath(exp_log_path)

    lightweight_SymEx(func_asm_path, inst_log_path, exp_log_path, max_inst_num=max_inst)  # TODO: max_inst_num


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
    elif func_type == 'add':
        mem_read_log(mem_read_log_path, start_addr, end_addr, prog_path, data_path)
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        read_mem_regions = memory_slices(mem_read_log_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        if len(read_mem_regions) > 1 and \
                read_mem_regions[0][1] - read_mem_regions[0][0] == read_mem_regions[1][1] - read_mem_regions[1][0]:
            # the case of add layer after dense/fully-connected layer
            return (read_mem_regions[0][1] - read_mem_regions[0][0]) / 4
        bias_length = explain_tvm_add_result(mem_exp_log, read_mem_regions, write_mem_regions)
        return bias_length
    elif func_type.startswith('max'):
        mem_write_log(mem_write_log_path, start_addr, end_addr, prog_path, data_path)
        write_mem_regions = memory_slices(mem_write_log_path)
        kernel_size, stride = explain_tvm_maxpool_result(mem_exp_log, write_mem_regions)
        return kernel_size, stride


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
            if len(label.strip()) > 0 and 'conv2d' in label:
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


def extract_params(prog_path: str, in_data: str, w_shape: tuple, dump_point: str, log_path: str, func_name: str, func_type='conv2d', data_index=1):
    prog_path = os.path.abspath(prog_path)
    in_data = os.path.abspath(in_data)
    log_path = os.path.abspath(log_path)
    dwords_len = 0
    if func_type == 'conv2d':
        dwords_len = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
    elif func_type == 'dense' or func_type == 'add':
        dwords_len = w_shape[0] * w_shape[1]
    elif func_type == 'var' or func_type == 'gamma' or func_type == 'mean' or func_type == 'beta':
        dwords_len = w_shape[0] * w_shape[1]
    else:
        assert 'the func_type {} is not defined'.format(func_type) and False

    # dump the memory
    rm_log(log_path)
    dump_dwords_2(prog_path, in_data, dump_point, dwords_len, log_path, data_index)

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
            elif func_type == 'add':
                json_name = func_name[:func_name.rfind('.')] + '.biases_{}.json'.format(i)
            elif func_type == 'var':
                json_name = func_name[:func_name.rfind('.')] + '.var_{}.json'.format(i)
            elif func_type == 'gamma':
                json_name = func_name[:func_name.rfind('.')] + '.gamma_{}.json'.format(i)
            elif func_type == 'mean':
                json_name = func_name[:func_name.rfind('.')] + '.mean_{}.json'.format(i)
            elif func_type == 'beta':
                json_name = func_name[:func_name.rfind('.')] + '.beta_{}.json'.format(i)
            else:
                assert 'func_type {} is not supported'.format(func_type) and False
            with open(json_name, 'w') as wf:
                wf.write(json_str)
                wf.close()
    rm_log(log_path)
"""


if __name__ == '__main__':
    utils.funcs_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/resnet18_funcs/"
    prog_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/resnet18_tvm_O0_strip"
    in_data = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/cat.bin"
    log_path = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/func_call.log"
    label_file = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/step1.txt"

    tmp_log_path = './inst_trace.log'
    exp_log_path = './mem_exp.log'
    mem_read_log_path = './mem_read.log'
    mem_write_log_path = './mem_write.log'
    mem_dump_log_path = 'mem_dump.log'

    # ==============================================================
    # Step 1 --- Get the Sequence of Layers ---
    # ==============================================================

    # get_funcs_trace(prog_path, in_data, log_path, label_file, only_fused=False)
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm')
    utils.print_layer_label_tvm(log_path)
    utils.get_funcs_trace(prog_path, in_data, log_path, label_file, compiler='tvm', only_fused=True)
    utils.print_layer_label_tvm(log_path, config_path='config.json', only_fused=True)
    #utils.print_input_id(log_path)  # to reconstruct the conputational graph
    #exit(0)
    
    """ # to be removed
    log_path = './resnet18_strip_func_call_fused.log'
    get_funcs_trace(prog_path, in_data, log_path, label_file, only_fused=True)
    new_log_path = './resnet18_strip_func_call_fused_2.log'
    print_fused_trace(log_path, new_log_path)
    call_graph_list = get_call_graph_list('./resnet18_strip_func_call_fused_3.log')
    show_graph(call_graph_list)
    # print_layer_label(log_path)
    """

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
    # Step 2.2.1 Conv and Matmul layers
    func_trace_map = {'0170.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0170_slice.log', 
                      '0160.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0160_slice.log', 
                      '0153.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0153_slice.log', 
                      '0142.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0142_slice.log', 
                      '0122.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0122_slice.log', 
                      '0115.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0115_slice.log', 
                      '0090.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0090_slice.log', 
                      '0063.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0063_slice.log', 
                      '0059.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0059_slice.log', 
                      '0051.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0051_slice.log', 
                      '0028.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0028_slice.log',  # conv

                      '0092.txt': '/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/0092_slice.log',  # dense
                      }

    func_rndaddr_map = {'0170.txt': ('0x46917c8', 64, '0x432d50', '0x434875'), 
                        '0160.txt': ('0x377d7c8', 64, '0x42ec00', '0x430F8F'), 
                        '0153.txt': ('0x377d82c', 64, '0x42aec0', '0x42D6D0'), 
                        '0142.txt': ('0x46acbd4', 64, '0x4262e0', '0x428869'), 
                        '0122.txt': ('0x377d7d4', 64, '0x420600', '0x422F30'), 
                        '0115.txt': ('0x377d7d4', 64, '0x41c230', '0x41EB80'), 
                        '0090.txt': ('0x46acbd8', 64, '0x415810', '0x417E09'), 
                        '0063.txt': ('0x3469024', 64, '0x40ff40', '0x411274'), 
                        '0059.txt': ('0x3468fc8', 64, '0x40d190', '0x40EB4B'), 
                        '0051.txt': ('0x377d850', 64, '0x409280', '0x40BBC0'), 
                        '0028.txt': ('0x346913c', 64, '0x402ea0', '0x404FB7'),  # conv
                        
                        '0092.txt': ('0x377bbc0', 64, '0x4181a0', '0x4185f1'),  # dense
                        }
    #shape = utils.recover_shape_tvm('0153.txt', exp_log_path, mem_read_log_path, mem_write_log_path, prog_path, in_data, func_type='conv2d')
    #print(shape)
    #exit(0)
    # We have to pass the external function address to SE engine
    # This can be done automatically, but we do it manually for simplicity
    '''
    se_engine.extern_functions = {'0x400c10': 'memset'}  # address in .plt, name
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
    '''
    # ==============================================================
    
    # Step 2.2.2 Other layers
    # the BatchNorm2d is implemented with a special sequence (add, sqrt, divide, multiply, expand_dims, multiply, negative, multiply, add, expand_dims, add)
    '''
    asm_files = os.listdir(utils.funcs_dir)
    se_engine.extern_functions = {'0x400c10': 'memset'}  # address in .plt, name
    results_dict = dict()
    for asm_file in asm_files:
        if 'labels' not in asm_file and asm_file.endswith('.txt'):
            asm_path = os.path.join(utils.funcs_dir, asm_file)
            start_addr, _ = utils.get_func_range(asm_path)
            if start_addr in utils.addr2label.keys():
                func_type = utils.addr2label[start_addr]
                if 'pool' in func_type or 'bias_add' in func_type:
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
    """
    # conv2d layers
    func_type = 'conv2d'
    #func_name = '0057.function_402ea0.txt'  # 64, 3, 7, 7
    func_name = '0080.function_409280.txt'
    #func_name = '0088.function_40d190.txt'  # 256, 128, 1, 1
    #func_name = '0092.function_40ff40.txt'  # 128, 64, 1, 1
    #func_name = '0119.function_415810.txt'  # 512, 512, 3, 3
    #func_name = '0144.function_41c230.txt'  # 128, 128, 3, 3
    #func_name = '0151.function_420600.txt'  # 256, 128, 3, 3
    #func_name = '0171.function_4262e0.txt'  # 512, 256, 3, 3
    #func_name = '0182.function_42aec0.txt'  # 128, 64, 3, 3 --> 128, 16, 6, 6, not the same
    #func_name = '0189.function_42ec00.txt'  # 256, 256, 3, 3
    #func_name = '0199.function_432d50.txt'  # 512, 256, 1, 1
    # dense/fully-connected layers
    #func_type = 'dense'
    #func_name = '0121.function_4181a0.txt'  # 1000, 512
    # bias add layer
    #func_type = 'add'
    #func_name = '0184.function_42da10.txt'  # dense add 1000
    #func_name = '0112.function_414370.txt'  # batch norm 512
    #func_name = '0129.function_419630.txt'  # batch norm 257 --> should be 256
    #func_name = '0201.function_434b50.txt'  # batch norm 128
    #func_name = '0146.function_41ee50.txt'  # batch norm 64
    # max-poll layers
    #func_type = 'max'
    #func_name = '0127.function_4190a0.txt'  # max 3, 2, kernel, stride

    tmp_log_path = './inst_trace.log'
    generate_inst_trace(func_name, tmp_log_path, prog_path, in_data)
    exp_log_path = './mem_exp.log'
    generate_symbolic_expression(func_name, tmp_log_path, exp_log_path, max_inst=5000000)

    # --- try to interpret the filter shape from symbolic expression log
    mem_read_log_path = 'mem_read.log'
    mem_write_log_path = 'mem_write.log'
    shape = recover_shape(func_name, exp_log_path,
                          mem_read_log_path, mem_write_log_path,
                          prog_path, in_data, func_type=func_type)
    print(shape)
    """
    # ==============================================================
    # Step 3 --- Extract Weights/Biases from Binary (dynamically)
    # ==============================================================

    
    # (name, shape, fused_func, type, padding, stride, param_index)
    func_meta_data = [('0028.txt', (64, 3, 7, 7), '0x4022b0', 'conv2d', 3, 2, 1),
                      ('0051.txt', (64, 64, 3, 3), '0x408330', 'conv2d', 1, 1, 1),
                      ('0059.txt', (256, 128, 1, 1), '0x40c450', 'conv2d', 0, 2, 1),
                      ('0063.txt', (128, 64, 1, 1), '0x40eb70', 'conv2d', 0, 2, 1),
                      ('0090.txt', (512, 512, 3, 3), '0x414820', 'conv2d', 1, 1, 1),
                      ('0115.txt', (128, 128, 3, 3), '0x41b190', 'conv2d', 1, 1, 1),
                      ('0122.txt', (256, 128, 3, 3), '0x41ef20', 'conv2d', 1, 2, 1),
                      ('0142.txt', (512, 256, 3, 3), '0x425450', 'conv2d', 1, 2, 1),
                      # ('0153.txt', (*, 16, 6, 6), '0x429b40', 'conv2d', 1, 2, 1),  # <-- wrong shape
                      ('0153.txt', (128, 64, 3, 3), '0x429b40', 'conv2d', 1, 2, 1),
                      ('0160.txt', (256, 256, 3, 3), '0x42db40', 'conv2d', 1, 1, 1),
                      ('0170.txt', (512, 256, 1, 1), '0x431950', 'conv2d', 0, 2, 1),

                      ('0092.txt', (1000, 512), '0x417e30', 'dense', 0, 0, 1),
                      ('0155.txt', (1, 1000), '0x42d700', 'add', 0, 0, 1),  # bias add

                      ('0083.txt', (1, 512), '0x4140c0', 'var', 0, 0, 0),  # norm var - used in add
                      ('0085.txt', (1, 512), '0x414440', 'gamma', 0, 0, 1),  # norm gamma - used in multiply
                      ('0129.txt', (1, 512), '0x423e00', 'mean', 0, 0, 0),  # norm mean - used in negative
                      ('0096.txt', (1, 512), '0x418a00', 'beta', 0, 0, 1),  # norm beta - used in add

                      ('0100.txt', (1, 256), '0x419380', 'var', 0, 0, 0),  # norm var - used in add
                      ('0081.txt', (1, 256), '0x413ce0', 'gamma', 0, 0, 1),  # norm gamma - used in multiply
                      ('0067.txt', (1, 256), '0x411680', 'mean', 0, 0, 0),  # norm mean - used in negative
                      ('0065.txt', (1, 256), '0x4112a0', 'beta', 0, 0, 1),  # norm beta - used in add

                      ('0172.txt', (1, 128), '0x4348a0', 'var', 0, 0, 0),  # norm var - used in add
                      ('0131.txt', (1, 128), '0x4240d0', 'gamma', 0, 0, 1),  # norm gamma - used in multiply
                      ('0023.txt', (1, 128), '0x401fe0', 'mean', 0, 0, 0),  # norm mean - used in negative
                      ('0166.txt', (1, 128), '0x431570', 'beta', 0, 0, 1),  # norm beta - used in add

                      ('0117.txt', (1, 64), '0x41eba0', 'var', 0, 0, 0),  # norm var - used in add
                      ('0176.txt', (1, 64), '0x435180', 'gamma', 0, 0, 1),  # norm gamma - used in multiply
                      ('0162.txt', (1, 64), '0x430fb0', 'mean', 0, 0, 0),  # norm mean - used in negative
                      ('0077.txt', (1, 64), '0x413290', 'beta', 0, 0, 1),  # norm beta - used in add

                      ]

    for fun_data in func_meta_data:
        func_name = fun_data[0]
        w_shape = fun_data[1]
        dump_point = fun_data[2]
        func_type = fun_data[3]
        data_index = fun_data[6]
        #if not func_name.startswith('0153'):
        #    continue # failed to recover shape for function 0153
        utils.extract_params_tvm(prog_path, in_data, w_shape, dump_point, mem_dump_log_path, func_name, func_type, data_index)
