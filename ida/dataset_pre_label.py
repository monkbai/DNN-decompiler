#!/usr/bin/python3
import subprocess
import os
import sys



class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def cmd(commandline):
    with cd(project_dir):
        print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


project_dir = rootdir = r"./"
labels_set = set()


def get_TVM_O0_labels(file_name: str):
    global labels_set
    file_name = file_name.split('.')[1]
    if not file_name.startswith('fused'):
        return []
    else:
        file_name = file_name.replace('fused_', '')
    if file_name[-1].isdigit():
        file_name = file_name[:file_name.rfind('_')]

    file_name = file_name.replace('nn_', '')
    labels_set.add(file_name)
    return [file_name]


def get_TVM_O3_labels(file_name: str):
    global labels_set
    contrib_flag = False
    labels = []
    file_name = file_name.split('.')[1]
    if not file_name.startswith('fused'):
        return []
    else:
        file_name = file_name.replace('fused_', '')

    if 'contrib' in file_name:
        contrib_flag = True
    file_name = file_name.replace('nn_', '')
    file_name = file_name.replace('contrib_', '')
    file_name = file_name.replace('NCHWc_', '')
    file_name = file_name.replace('NCHWc', '')
    file_name = file_name.strip('_')

    if file_name[-1].isdigit():
        file_name = file_name[:file_name.rfind('_')]

    if contrib_flag:
        labels = file_name.split('_')
    """    
    if 'conv2d_multiply_add' == file_name:
        labels = ['multiply', 'add', 'conv2d']
    if 'conv2d_multiply_add' == file_name:
        labels = ['multiply', 'add', 'conv2d']
    if 'conv2d_multiply_add' == file_name:
        labels = ['multiply', 'add', 'conv2d']
    elif 'conv2d_add_relu' == file_name:
        labels = ['conv2d', 'add', 'relu']
    elif 'multiply_add_add_relu_layout_transform' == file_name:
        labels = ['multiply', 'add', 'add', 'relu', 'layout_transform']
    elif 'layout_transform_reshape' == file_name:
        labels = ['layout_transform', 'reshape']
    elif 'reshape_batch_flatten' == file_name:
        labels = ['reshape', 'batch_flatten']
    elif 'layout_transform_reshape_batch_flatten' == file_name:
        labels = ['layout_transform', 'reshape', 'batch_flatten']
    elif 'layout_transform_batch_flatten_batch_flatten' == file_name:
        labels = ['layout_transform', 'batch_flatten', 'batch_flatten']
    """
    if len(labels) > 0:
        for label in labels:
            labels_set.add(label)
        return labels

    for single_label in labels_set:
        while file_name.count(single_label) > 0:
            file_name = file_name.replace(single_label, '', 1)
            labels.append(single_label)
    if len(file_name) > 0 and len(file_name) != file_name.count('_'):
        labels.append(file_name.strip('_'))

    for label in labels:
        labels_set.add(label)
    return labels


def main_TVM(the_dir: str, opt_level=0):
    if opt_level == 3:
        labels_set.add('batch_flatten')
        labels_set.add('layout_transform')
        labels_set.add('dense')
        labels_set.add('conv2d')
        labels_set.add('reshape')

    the_dir = os.path.abspath(the_dir)
    for root, dirs, files in os.walk(the_dir):
        root = os.path.abspath(root)
        for d in dirs:
            curr_d = os.path.join(root, d)
            label_path = os.path.join(curr_d, 'labels.txt')
            # if os.path.exists(label_path):
            #     continue
            print(curr_d)
            asm_files = os.listdir(curr_d)
            asm_files.sort()
            current_labels = []
            labels_list = []
            with open(label_path, 'w') as f:
                index = 0
                while index < len(asm_files):
                    asm_f = asm_files[index]
                    f.write('{}: '.format(asm_f))
                    if 'fused' in asm_f:
                        current_labels.append('entry')
                        if opt_level == 0:
                            labels_list = get_TVM_O0_labels(asm_f)
                        else:
                            labels_list = get_TVM_O3_labels(asm_f)

                    if index < len(asm_files) - 1 and ('fused' in asm_files[index + 1] or 'TVMSystem' in asm_files[index+1]):
                        current_labels += labels_list
                        labels_list = []

                    if len(current_labels) > 0:
                        idx = 0
                        while idx < len(current_labels) - 1:
                            f.write('{}, '.format(current_labels[idx]))
                            idx += 1
                        f.write(current_labels[idx])
                        current_labels = []

                    f.write('\n')
                    index += 1
                f.close()
                
            print('continue?')


def main_GLOW(the_dir: str):
    global labels_set
    the_dir = os.path.abspath(the_dir)
    for root, dirs, files in os.walk(the_dir):
        root = os.path.abspath(root)
        for d in dirs:
            curr_d = os.path.join(root, d)
            label_path = os.path.join(curr_d, 'labels.txt')
            # if os.path.exists(label_path):
            #     continue
            print(curr_d)
            asm_files = os.listdir(curr_d)
            asm_files.sort()
            current_labels = []
            labels_list = []
            with open(label_path, 'w') as f:
                index = 0
                while index < len(asm_files):
                    asm_f = asm_files[index]
                    f.write('{}: '.format(asm_f))
                    if 'libjit' in asm_f:
                        label = asm_f.split('.')[1]
                        label = label.replace('libjit_', '')
                        if label[-1].isdigit():
                            label = label[:label.rfind('_')]
                        if label.endswith('_f'):
                            label = label[:-2]
                        f.write(label)
                        labels_set.add(label)
                    f.write('\n')
                    index += 1
                f.close()

            print('continue?')


if __name__ == '__main__':
    # main_TVM('./TVM_binaries/O0/')
    # main_TVM('./TVM_binaries/O3/', opt_level=3)
    main_GLOW('./GLOW_binaries/')
    for label in labels_set:
        print(label)
