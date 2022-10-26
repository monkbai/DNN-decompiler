import os
import re
import sys
import json
import torch
import numpy as np


def read_json(json_path: str):
    with open(json_path, 'r') as f:
        j_txt = f.read()
        list_obj = json.loads(s=j_txt)
        arr_obj = np.array(list_obj, dtype=np.float32)
        tensor_obj = torch.from_numpy(arr_obj)
        return tensor_obj


def param_count(param_file: str):
    tensor_obj = read_json(param_file)
    s = 1
    dim = len(tensor_obj.shape)
    for c in tensor_obj.shape:
        s = s * c
    if tensor_obj.shape[0] == 1:
        dim -= 1
    return s, dim

def traverse(param_dir: str):
    visited_file = []
    param_num = 0
    dim_num = 0
    for filename in os.listdir(param_dir):
        if re.match(r"\d{4,4}\.", filename) and filename.endswith('.json'):
            f_path = os.path.join(param_dir, filename)
            p_count, s_count = param_count(f_path)
            param_num += p_count
            
            mat = re.match(r"(\d{4,4})\.", filename)
            if mat.group(1) in visited_file or 'gamma' in filename or 'beta' in filename or 'var' in filename:
                continue
            else:
                visited_file.append(mat.group(1))
                dim_num += s_count
    return float(param_num), float(dim_num)


if __name__ == '__main__':
    print("TVM v0.7 O0 Resnet18")
    param_num, dim_num = traverse("./evaluation/resnet18_tvm_v07_O0/")
    error_param, error_dim = param_count("./evaluation/resnet18_tvm_v07_O0/0153.weights_0.json")
    print("Dimension accuracy: {}%".format((1-4/(dim_num))*100))
    print("Parameter accuracy: {}%".format((1-error_param/param_num)*100))

    # print("TVM v0.8 O0 Resnet18")
    # param_num, dim_num = traverse("./evaluation/resnet18_tvm_v08_O0/")
    # error_param, error_dim = param_count("./evaluation/resnet18_tvm_v08_O0/0166.weights_0.json")
    # print("Dimension accuracy: {}%".format((1-4/(dim_num))*100))
    # print("Parameter accuracy: {}%".format((1-error_param/param_num)*100))

    print("TVM v0.9.dev O0 Resnet18")
    param_num, dim_num = traverse("./evaluation/resnet18_tvm_v09_O0/")
    error_param, error_dim = param_count("./evaluation/resnet18_tvm_v09_O0/0164.weights_0.json")
    print("Dimension accuracy: {}%".format((1-4/(dim_num))*100))
    print("Parameter accuracy: {}%".format((1-error_param/param_num)*100))

    print("TVM v0.7 O3 Resnet18")
    param_num, dim_num = traverse("./evaluation/resnet18_tvm_v07_O3/")
    error_param, error_dim = param_count("./evaluation/resnet18_tvm_v07_O3/0038.weights_0.json")
    print("Parameter accuracy: {}%".format((1-error_param/param_num)*100))

    # print("TVM v0.8 O3 Resnet18")
    # param_num, dim_num = traverse("./evaluation/resnet18_tvm_v08_O3/")
    # error_param, error_dim = param_count("./evaluation/resnet18_tvm_v08_O3/0101.weights_0.json")
    # print("Parameter accuracy: {}%".format((1-error_param/param_num)*100))

    print("TVM v0.9.dev O3 Resnet18")
    param_num, dim_num = traverse("./evaluation/resnet18_tvm_v09_O3/")
    error_param, error_dim = param_count("./evaluation/resnet18_tvm_v09_O3/0101.weights_0.json")
    print("Parameter accuracy: {}%".format((1-error_param/param_num)*100))
