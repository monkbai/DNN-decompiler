import os
import json

import utils

def extract_op(input_dir, output_dir):
    '''
    Extract op from assmbly code.
    You may need to adjust the structure of `for` loop and
    `if` condition based the hierarchy of your folder.
    '''
    sub_list = [
            'Glow_2020/', 'Glow_2021/', 'Glow_2022/',
            'TVM_v0.7_O0/', 'TVM_v0.7_O3/',
            'TVM_v0.8_O0/', 'TVM_v0.8_O3/',
            'TVM_v0.9.dev_O0/', 'TVM_v0.9.dev_O3/'
            ]
    
    utils.make_path(output_dir)

    for sub in sub_list:
        s_sp = sub.split('/')
        for i in range(len(s_sp)):
            utils.make_path(output_dir + '/'.join(s_sp[:i]))
        model_list = os.listdir(input_dir + sub)
        for model in model_list:
            if not 'funcs' in model:
                continue
            utils.make_path(output_dir + sub + model)
            label_file = input_dir + sub + model + '/labels.txt'
            with open(label_file, 'r') as f:
                lines = f.readlines()
            label_dict = {}
            for l in lines:
                if len(l):
                    [func, label] = l.split(':')
                    if len(label.strip()) and label.strip() != 'entry':
                        label_dict[func] = label.strip()
            for func in label_dict.keys():
                with open(input_dir + sub + model + '/' + func, 'r') as f:
                    lines = f.readlines()
                op_list = []
                for l in lines[2:]:
                    cont = l[50:]
                    op = cont.split(' ')[0].strip()
                    op_list.append(op)
                with open(output_dir + sub + model + '/' + func, 'w') as f:
                    for op in op_list:
                        f.write('%s ' % op)

def op_all(input_dir, output_path, compiler='TVM'):
    '''
    Put all op extracted from `extract_op` into a single for
    BPE processing and building vocabulary.
    Here, `input_dir` is the `output_dir` of `extract_op`.
    '''
    if compiler == 'TVM':    
        sub_list = [
                'TVM_v0.7_O0/', 'TVM_v0.7_O3/',
                'TVM_v0.8_O0/', 'TVM_v0.8_O3/',
                'TVM_v0.9.dev_O0/', 'TVM_v0.9.dev_O3/'
            ]
    elif compiler == 'GLOW':
        sub_list = [
                'Glow_2020/', 'Glow_2021/', 'Glow_2022/'
            ]
    else:
        raise NotImplementedError

    all_op_list = []
    for sub in sub_list:
        model_list = os.listdir(input_dir + sub)
        for model in model_list:
            if not 'funcs' in model:
                continue
            func_list = os.listdir(input_dir + sub + model)
            for func in func_list:
                with open(input_dir + sub + model + '/' + func, 'r') as f:
                    lines = f.readlines()
                op_list = lines[0].strip().split(' ')
                for op in op_list:
                    if len(op):
                        all_op_list.append(op)

    with open(output_path, 'w') as f:
        for op in all_op_list:
            f.write('%s ' % op)


def data_split(input_dir, output_dir, split='train', compiler='TVM'):
    '''
    Split all op extracted from `extract_op` as non-overlapping
    training and test split.
    Here, `input_dir` is the `output_dir` of `extract_op`.
    '''
    assert split in ['train', 'test']
    if compiler == 'TVM':
        version_list = [
            'TVM_v0.7_O0', 'TVM_v0.7_O3',
            'TVM_v0.8_O0', 'TVM_v0.8_O3',
            'TVM_v0.9.dev_O0', 'TVM_v0.9.dev_O3'
        ]
        test_model_list = ['vgg16', 'resnet18', 'mobilenet', 'efficientnet', 'inception_v1', 'shufflenet_v2']
    elif compiler == 'GLOW':
        version_list = ['Glow_2020', 'Glow_2021', 'Glow_2022']
        test_model_list = ['vgg16', 'resnet18_v1_7', 'mobilenetv2_7', 'efficientnet_lite4', 'inception_v1', 'shufflenet_v2']
    else:
        raise NotImplementedError

    utils.make_path(output_dir)

    for version in version_list:
        model_list = os.listdir(input_dir + version)
        for model in model_list:
            if compiler == 'GLOW':
                model_name = model.split('.')[0]
            elif compiler == 'TVM':
                model_name = '_'.join(model.split('_')[:-1])
            else:
                raise NotImplementedError
            if not ((split == 'test') ^ (model_name in test_model_list)):
                '''
                Equivalent to the above code
                if split == 'train':
                    if model_name not in test_model_list:
                        # execute
                else:
                    if model_name in test_model_list:
                        # execute
                '''    
                func_list = os.listdir(input_dir + version + '/' + model)
                for func in func_list:
                    in_path = input_dir + version + '/' + model + '/' + func
                    out_path = output_dir + ('%s-%s-%s' % (version, model_name, func.split('.')[1]))
                    os.system('cp %s %s' % (in_path, out_path))
            else:
                print(model_name)



def build_label(input_dir, output_dir, compiler='TVM'):
    '''
    Build the label and label_index dict.
    Here, the `input_dir` is same as the `input_dir` of `extract_op`
    '''
    if compiler == 'TVM':        
        version_list = [
            'TVM_v0.7_O0', 'TVM_v0.7_O3',
            'TVM_v0.8_O0', 'TVM_v0.8_O3',
            'TVM_v0.9.dev_O0', 'TVM_v0.9.dev_O3'
        ]
    elif compiler == 'GLOW':
        version_list = ['Glow_2020', 'Glow_2021', 'Glow_2022']
    else:
        raise NotImplementedError

    output_dict = {}

    for version in version_list:
        model_list = os.listdir(input_dir + version)
        for model in model_list:
            if not 'funcs' in model:
                continue
            if compiler == 'GLOW':
                model_name = model.split('.')[0]
            elif compiler == 'TVM':
                model_name = '_'.join(model.split('_')[:-1])
            else:
                raise NotImplementedError
            label_file = input_dir + version + '/' + model + '/labels.txt'
            with open(label_file, 'r') as f:
                lines = f.readlines()
            label_dict = {}
            for l in lines:
                if len(l):
                    [func, label] = l.split(':')
                    if len(label.strip()) and label.strip() != 'entry':
                        label_dict[func] = label.strip()
            for func in label_dict.keys():
                k = ('%s-%s-%s' % (version, model_name, func.split('.')[1]))
                v = label_dict[func].split(', ')
                output_dict[k] = v

    with open('%s/%s_label.json' % (output_dir, compiler), 'w') as f:
        json.dump(output_dict, f)


    with open('%s/%s_label.json' % (output_dir, compiler), 'r') as f:
        label_dict = json.load(f)

    index_dict = {}
    cnt = 0

    for v_list in label_dict.values():
        for v in v_list:
            if v not in index_dict.keys():
                index_dict[v] = cnt
                cnt += 1

    with open('%s/%s_index.json' % (output_dir, compiler), 'w') as f:
        json.dump(index_dict, f, indent=2)
