import os
import re


merged_stacked_kernels = False


def refine_all(rootdir: str):
    global merged_stacked_kernels
    refine_count = 0

    the_dir = os.path.abspath(rootdir)
    for root, dirs, files in os.walk(the_dir):
        root = os.path.abspath(root)
        for f in files:
            if f.endswith('_ir.txt'):
                merged_stacked_kernels = False
                funcs_dir = f.replace('_ir.txt', '.out')
                funcs_dir += '_funcs'
                funcs_dir = os.path.join(root, funcs_dir)
                assert os.path.exists(funcs_dir)
                labels_path = os.path.join(funcs_dir, 'labels.txt')

                func_call_list = extract_entry_func(funcs_dir)
                op_list = extract_ir(os.path.join(root, f))
                if not len(func_call_list) == len(op_list):
                    merged_stacked_kernels = True
                    op_list = merge_stacked_kernels(op_list)
                    assert len(func_call_list) == len(op_list)
                print('Refining', funcs_dir, '...')
                refine_labels(labels_path, func_call_list, op_list)
                refine_count += 1
    print(refine_count)


def merge_stacked_kernels(op_list):
    new_op_list = []
    i = 0
    while i < len(op_list):
        if op_list[i].startswith('element'):
            current_label = op_list[i]
            while op_list[i+1].startswith('element') or op_list[i+1].startswith('relu'):
                current_label += ', ' + op_list[i+1]
                i += 1
            new_op_list.append(current_label)
        else:
            new_op_list.append(op_list[i])

        i += 1
    return new_op_list


def refine_labels(labels_path: str, func_call_list, op_list):
    new_labels_txt = ''
    with open(labels_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            func_name, label = l.split(': ')
            label = label.strip()
            tmp_func_name = func_name
            if len(label) == 0:
                new_labels_txt += tmp_func_name + ': ' + '\n'
                continue
            func_name = func_name[5:-4]

            for i in range(len(func_call_list)):
                if func_call_list[i] == func_name:
                    if 'stacked_kernel' not in func_name:
                        assert check(label, op_list[i])
                    else:  # refine the 'stacked_kernel' label
                        label = translate(op_list[i])
                    break
            new_labels_txt += tmp_func_name + ': ' + label + '\n'

    with open(labels_path, 'w') as f:
        f.write(new_labels_txt)


def translate(op_type):
    if op_type == 'relu':
        return 'relu'
    elif op_type.startswith('element'):
        return op_type
    return 'TODO'


def check(original_label, ir_label):
    if original_label == ir_label:
        return True
    elif ir_label.startswith('conv') and original_label.startswith('conv'):
        return True
    elif ir_label.startswith('cpuconv') and original_label.startswith('conv'):
        return True
    elif ir_label.startswith('fullyconnected') and original_label.startswith('fc'):
        return True
    elif ir_label.startswith('maxpool') and original_label.startswith('max_pool'):
        return True
    elif ir_label.startswith('avgpool') and original_label.startswith('avg_pool'):
        return True
    elif ir_label.startswith('inserttensor') and original_label.startswith('insert_tensor'):
        return True
    elif ir_label.startswith('extracttensor') and original_label.startswith('extract_tensor'):
        return True
    elif ir_label.startswith('localresponsenormalization') and original_label.startswith('local_response_normalization'):
        return True
    return False


def extract_entry_func(funcs_dir: str):
    func_call_list = []

    files = os.listdir(funcs_dir)
    for f_name in files:
        if f_name.startswith('0008.'):
            f_path = os.path.join(funcs_dir, f_name)
            with open(f_path, 'r') as f:
                asm_txt = f.read()
                asm_lines = asm_txt.split('\n')
                for l in asm_lines:
                    l = l[50:]
                    l = l.strip()
                    if l.startswith('call'):
                        func_name = l.replace('call', '').strip()
                        func_call_list.append(func_name)

    return func_call_list


def extract_ir(ir_path: str):
    op_list = []

    with open(ir_path, 'r') as f:
        ir_txt = f.read()
        segments = ir_txt.split('\n\n')
        ir_txt = segments[-1]
        assert ir_txt.startswith('code {')
        ir_lines = ir_txt.split('\n')
        for l in ir_lines:
            if ' = allocactivation' in l or ' = deallocactivation' in l or ' = touch' in l or ' = tensorview' in l or not l.startswith('  '):
                continue
            mat = re.search(' = ([a-z\d]+) ', l)
            op_type = mat.group(1)
            op_list.append(op_type)

    return op_list


# =======

def refine_all_tvm(rootdir: str):
    refine_count = 0

    the_dir = os.path.abspath(rootdir)
    for root, dirs, files in os.walk(the_dir):
        root = os.path.abspath(root)
        for f in files:
            if f.endswith('labels.txt'):
                labels_path = os.path.join(root, f)
                print('Refining', root, '...')

                with open(labels_path, 'r') as f:
                    lines = f.readlines()
                    for i in range(len(lines)):
                        l = lines[i]
                        func_name, label = l.split(': ')
                        label = label.strip()
                        if 'avg' in label:
                            prev_func_name, prev_label = lines[i-1].split(': ')
                            prev_label = prev_label.strip()
                            if 'sub_' in prev_func_name and prev_label == '':
                                lines[i-1] = prev_func_name + ": " + label + '\n'
                                lines[i] = func_name + ": " + prev_label + '\n'
                with open(labels_path, 'w') as f:
                    for l in lines:
                        f.write(l)

                refine_count += 1
    print(refine_count)


if __name__ == '__main__':
    refine_all_tvm('/home/lifter/Documents/DL_compiler/BTD_DATA/labeled_dataset_2022/binaries_rm_section/TVM_v0.7_O0')
    refine_all_tvm('/home/lifter/Documents/DL_compiler/BTD_DATA/labeled_dataset_2022/binaries_rm_section/TVM_v0.7_O3')
    refine_all_tvm('/home/lifter/Documents/DL_compiler/BTD_DATA/labeled_dataset_2022/binaries_rm_section/TVM_v0.8_O0')
    refine_all_tvm('/home/lifter/Documents/DL_compiler/BTD_DATA/labeled_dataset_2022/binaries_rm_section/TVM_v0.8_O3')
    refine_all_tvm('/home/lifter/Documents/DL_compiler/BTD_DATA/labeled_dataset_2022/binaries_rm_section/TVM_v0.9.dev_O0')
    refine_all_tvm('/home/lifter/Documents/DL_compiler/BTD_DATA/labeled_dataset_2022/binaries_rm_section/TVM_v0.9.dev_O3')
    # refine_all('/home/lifter/Documents/DL_compiler/BTD_DATA/labeled_dataset_2022/binaries_rm_section/Glow_2020')
    # refine_all('/home/lifter/Documents/DL_compiler/BTD_DATA/labeled_dataset_2022/binaries_rm_section/Glow_2021')
    # refine_all('/home/lifter/Documents/DL_compiler/BTD_DATA/labeled_dataset_2022/binaries_rm_section/Glow_2022')
