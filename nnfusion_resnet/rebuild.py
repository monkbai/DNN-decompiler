import os
import subprocess


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
        # print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


project_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion_resnet/"  # linrary and Constant in this file

# =========================================================================

import torch.nn as nn
import torch
import json
import numpy as np
import time
import torch.nn.functional as F


def read_json(json_path: str):
    with open(json_path, 'r') as f:
        j_txt = f.read()
        list_obj = json.loads(s=j_txt)
        arr_obj = np.array(list_obj, dtype=np.float32)
        tensor_obj = torch.from_numpy(arr_obj)
        return tensor_obj


def set_weights(module: nn.modules, json_path: str):
    w = read_json(json_path)
    module.weight = torch.nn.Parameter(w)
    if module.bias is not None:
        torch.nn.init.zeros_(module.bias)


def set_linear_weights(module: nn.modules, json_path: str):
    with open(json_path, 'r') as f:
        j_txt = f.read()
        list_obj = json.loads(s=j_txt)
        arr_obj = np.array(list_obj, dtype=np.float32)
        arr_obj = arr_obj.transpose(1, 0)
        w = torch.from_numpy(arr_obj)
        
    # w = read_json(json_path)
    module.weight = torch.nn.Parameter(w)
    if module.bias is not None:
        torch.nn.init.zeros_(module.bias)


def set_biases(module: nn.modules, json_path: str):
    w = read_json(json_path)
    w = w.reshape(w.shape[1])
    module.bias = torch.nn.Parameter(w)


def set_bn_weights(module: nn.modules, json_path: str):
    w = read_json(json_path)
    w = w.reshape(w.shape[1])
    module.weight = torch.nn.Parameter(w)
    module.training = False


def set_mean(module: nn.modules, json_path: str):
    w = read_json(json_path)
    w = w.reshape(w.shape[1])
    module.running_mean = w  # torch.nn.Parameter(w)
    #w = torch.nn.Parameter(w)
    #module.running_mean = w


def set_var(module: nn.modules, json_path: str):
    w = read_json(json_path)
    w = w.reshape(w.shape[1])
    module.running_var = w  # torch.nn.Parameter(w)
    #w = torch.nn.Parameter(w)
    #module.running_var = w


class MyResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyResNet, self).__init__()
        net = [nn.Identity()]

        net.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))  # 0
        set_weights(net[-1], './0045.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=64, eps=1.001e-05, track_running_stats=False))  # 1
        set_bn_weights(net[-1], './0343.gamma_0.json')
        set_biases(net[-1], './0343.beta_0.json')
        set_var(net[-1], './0343.var_0.json')
        set_mean(net[-1], './0343.mean_0.json')
        net.append(nn.ReLU())  # 2
        net.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=0))  # 3
        
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0))  # 4
        set_weights(net[-1], './0130.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=64, eps=1.001e-05, track_running_stats=False))  # 5
        set_bn_weights(net[-1], './0346.gamma_0.json')
        set_biases(net[-1], './0346.beta_0.json')
        set_var(net[-1], './0346.var_0.json')
        set_mean(net[-1], './0346.mean_0.json')
        net.append(nn.ReLU())  # 6
        
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # 7
        set_weights(net[-1], './0055.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=64, eps=1.001e-05, track_running_stats=False))  # 8
        set_bn_weights(net[-1], './0346.gamma_1.json')
        set_biases(net[-1], './0346.beta_1.json')
        set_var(net[-1], './0346.var_1.json')
        set_mean(net[-1], './0346.mean_1.json')
        net.append(nn.ReLU())  # 9
        
        net.append(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0))  # 10
        set_weights(net[-1], './0057.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 11
        set_bn_weights(net[-1], './0344.gamma_0.json')
        set_biases(net[-1], './0344.beta_0.json')
        set_var(net[-1], './0344.var_0.json')
        set_mean(net[-1], './0344.mean_0.json')
        
        net.append(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0))  # 12
        set_weights(net[-1], './0057.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 13
        set_bn_weights(net[-1], './0344.gamma_1.json')
        set_biases(net[-1], './0344.beta_1.json')
        set_var(net[-1], './0344.var_1.json')
        set_mean(net[-1], './0344.mean_1.json')
        
        net.append(nn.ReLU())  # 14 add relu
        
        net.append(nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0))  # 15
        set_weights(net[-1], './0052.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=64, eps=1.001e-05, track_running_stats=False))  # 16
        set_bn_weights(net[-1], './0346.gamma_2.json')
        set_biases(net[-1], './0346.beta_2.json')
        set_var(net[-1], './0346.var_2.json')
        set_mean(net[-1], './0346.mean_2.json')
        net.append(nn.ReLU())  # 17

        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # 18
        set_weights(net[-1], './0055.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=64, eps=1.001e-05, track_running_stats=False))  # 19
        set_bn_weights(net[-1], './0346.gamma_3.json')
        set_biases(net[-1], './0346.beta_3.json')
        set_var(net[-1], './0346.var_3.json')
        set_mean(net[-1], './0346.mean_3.json')
        net.append(nn.ReLU())  # 20
        
        net.append(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0))  # 21
        set_weights(net[-1], './0057.weights_2.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 22
        set_bn_weights(net[-1], './0344.gamma_2.json')
        set_biases(net[-1], './0344.beta_2.json')
        set_var(net[-1], './0344.var_2.json')
        set_mean(net[-1], './0344.mean_2.json')
        
        net.append(nn.ReLU())  # 23 add relu
        
        net.append(nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0))  # 24
        set_weights(net[-1], './0052.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=64, eps=1.001e-05, track_running_stats=False))  # 25
        set_bn_weights(net[-1], './0346.gamma_4.json')
        set_biases(net[-1], './0346.beta_4.json')
        set_var(net[-1], './0346.var_4.json')
        set_mean(net[-1], './0346.mean_4.json')
        net.append(nn.ReLU())  # 26
        
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # 27
        set_weights(net[-1], './0055.weights_2.json')
        net.append(nn.BatchNorm2d(num_features=64, eps=1.001e-05, track_running_stats=False))  # 28
        set_bn_weights(net[-1], './0346.gamma_5.json')
        set_biases(net[-1], './0346.beta_5.json')
        set_var(net[-1], './0346.var_5.json')
        set_mean(net[-1], './0346.mean_5.json')
        net.append(nn.ReLU())  # 29
        
        net.append(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0))  # 30
        set_weights(net[-1], './0057.weights_3.json')        
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 31
        set_bn_weights(net[-1], './0344.gamma_3.json')
        set_biases(net[-1], './0344.beta_3.json')
        set_var(net[-1], './0344.var_3.json')
        set_mean(net[-1], './0344.mean_3.json')
        
        net.append(nn.ReLU())  # 32 add relu

        # ==================================

        net.append(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=2, padding=0))  # 33
        set_weights(net[-1], './0053.weights_0.json')        
        net.append(nn.BatchNorm2d(num_features=128, eps=1.001e-05, track_running_stats=False))  # 34
        set_bn_weights(net[-1], './0347.gamma_0.json')
        set_biases(net[-1], './0347.beta_0.json')
        set_var(net[-1], './0347.var_0.json')
        set_mean(net[-1], './0347.mean_0.json')
        net.append(nn.ReLU())  # 35
        
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # 36
        set_weights(net[-1], './0060.weights_0.json')        
        net.append(nn.BatchNorm2d(num_features=128, eps=1.001e-05, track_running_stats=False))  # 37
        set_bn_weights(net[-1], './0347.gamma_1.json')
        set_biases(net[-1], './0347.beta_1.json')
        set_var(net[-1], './0347.var_1.json')
        set_mean(net[-1], './0347.mean_1.json')
        net.append(nn.ReLU())  # 38
        
        net.append(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0))  # 39
        set_weights(net[-1], './0056.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 40
        set_bn_weights(net[-1], './0348.gamma_0.json')
        set_biases(net[-1], './0348.beta_0.json')
        set_var(net[-1], './0348.var_0.json')
        set_mean(net[-1], './0348.mean_0.json')
        
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0))  # 41
        set_weights(net[-1], './0058.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 42
        set_bn_weights(net[-1], './0348.gamma_1.json')
        set_biases(net[-1], './0348.beta_1.json')
        set_var(net[-1], './0348.var_1.json')
        set_mean(net[-1], './0348.mean_1.json')

        net.append(nn.ReLU())  # 43 add relu
        
        net.append(nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0))  # 44
        set_weights(net[-1], './0042.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=128, eps=1.001e-05, track_running_stats=False))  # 45
        set_bn_weights(net[-1], './0347.gamma_2.json')
        set_biases(net[-1], './0347.beta_2.json')
        set_var(net[-1], './0347.var_2.json')
        set_mean(net[-1], './0347.mean_2.json')
        net.append(nn.ReLU())  # 46
        
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # 47
        set_weights(net[-1], './0060.weights_1.json')        
        net.append(nn.BatchNorm2d(num_features=128, eps=1.001e-05, track_running_stats=False))  # 48
        set_bn_weights(net[-1], './0347.gamma_3.json')
        set_biases(net[-1], './0347.beta_3.json')
        set_var(net[-1], './0347.var_3.json')
        set_mean(net[-1], './0347.mean_3.json')
        net.append(nn.ReLU())  # 49
        
        net.append(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0))  # 50
        set_weights(net[-1], './0056.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 51
        set_bn_weights(net[-1], './0348.gamma_2.json')
        set_biases(net[-1], './0348.beta_2.json')
        set_var(net[-1], './0348.var_2.json')
        set_mean(net[-1], './0348.mean_2.json')
        
        net.append(nn.ReLU())  # 52 add relu
        
        net.append(nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0))  # 53
        set_weights(net[-1], './0042.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=128, eps=1.001e-05, track_running_stats=False))  # 54
        set_bn_weights(net[-1], './0347.gamma_4.json')
        set_biases(net[-1], './0347.beta_4.json')
        set_var(net[-1], './0347.var_4.json')
        set_mean(net[-1], './0347.mean_4.json')
        net.append(nn.ReLU())  # 55
        
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # 56
        set_weights(net[-1], './0060.weights_2.json')
        net.append(nn.BatchNorm2d(num_features=128, eps=1.001e-05, track_running_stats=False))  # 57
        set_bn_weights(net[-1], './0347.gamma_5.json')
        set_biases(net[-1], './0347.beta_5.json')
        set_var(net[-1], './0347.var_5.json')
        set_mean(net[-1], './0347.mean_5.json')
        net.append(nn.ReLU())  # 58
        
        net.append(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0))  # 59
        set_weights(net[-1], './0056.weights_2.json')        
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 60
        set_bn_weights(net[-1], './0348.gamma_3.json')
        set_biases(net[-1], './0348.beta_3.json')
        set_var(net[-1], './0348.var_3.json')
        set_mean(net[-1], './0348.mean_3.json')
        
        net.append(nn.ReLU())  # 61 add relu

        net.append(nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0))  # 62
        set_weights(net[-1], './0042.weights_2.json')
        net.append(nn.BatchNorm2d(num_features=128, eps=1.001e-05, track_running_stats=False))  # 63
        set_bn_weights(net[-1], './0347.gamma_6.json')
        set_biases(net[-1], './0347.beta_6.json')
        set_var(net[-1], './0347.var_6.json')
        set_mean(net[-1], './0347.mean_6.json')
        net.append(nn.ReLU())  # 64

        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # 65
        set_weights(net[-1], './0060.weights_3.json')
        net.append(nn.BatchNorm2d(num_features=128, eps=1.001e-05, track_running_stats=False))  # 66
        set_bn_weights(net[-1], './0347.gamma_7.json')
        set_biases(net[-1], './0347.beta_7.json')
        set_var(net[-1], './0347.var_7.json')
        set_mean(net[-1], './0347.mean_7.json')
        net.append(nn.ReLU())  # 67

        net.append(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0))  # 68
        set_weights(net[-1], './0056.weights_3.json')
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 69
        set_bn_weights(net[-1], './0348.gamma_4.json')
        set_biases(net[-1], './0348.beta_4.json')
        set_var(net[-1], './0348.var_4.json')
        set_mean(net[-1], './0348.mean_4.json')
        
        net.append(nn.ReLU())  # 70 add relu

        # ==================================

        net.append(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=2, padding=0))  # 71
        set_weights(net[-1], './0059.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 72
        set_bn_weights(net[-1], './0349.gamma_0.json')
        set_biases(net[-1], './0349.beta_0.json')
        set_var(net[-1], './0349.var_0.json')
        set_mean(net[-1], './0349.mean_0.json')
        net.append(nn.ReLU())  # 73

        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))  # 74
        set_weights(net[-1], './0050.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 75
        set_bn_weights(net[-1], './0349.gamma_1.json')
        set_biases(net[-1], './0349.beta_1.json')
        set_var(net[-1], './0349.var_1.json')
        set_mean(net[-1], './0349.mean_1.json')
        net.append(nn.ReLU())  # 76

        net.append(nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1, padding=0))  # 77
        set_weights(net[-1], './0043.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=1024, eps=1.001e-05, track_running_stats=False))  # 78
        set_bn_weights(net[-1], './0342.gamma_0.json')
        set_biases(net[-1], './0342.beta_0.json')
        set_var(net[-1], './0342.var_0.json')
        set_mean(net[-1], './0342.mean_0.json')

        net.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2, padding=0))  # 79
        set_weights(net[-1], './0044.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=1024, eps=1.001e-05, track_running_stats=False))  # 80
        set_bn_weights(net[-1], './0342.gamma_1.json')
        set_biases(net[-1], './0342.beta_1.json')
        set_var(net[-1], './0342.var_1.json')
        set_mean(net[-1], './0342.mean_1.json')
        
        net.append(nn.ReLU())  # 81  add relu

        net.append(nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0))  # 82
        set_weights(net[-1], './0048.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 83
        set_bn_weights(net[-1], './0349.gamma_2.json')
        set_biases(net[-1], './0349.beta_2.json')
        set_var(net[-1], './0349.var_2.json')
        set_mean(net[-1], './0349.mean_2.json')
        net.append(nn.ReLU())  # 84

        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))  # 85
        set_weights(net[-1], './0050.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 86
        set_bn_weights(net[-1], './0349.gamma_3.json')
        set_biases(net[-1], './0349.beta_3.json')
        set_var(net[-1], './0349.var_3.json')
        set_mean(net[-1], './0349.mean_3.json')
        net.append(nn.ReLU())  # 87

        net.append(nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1, padding=0))  # 88
        set_weights(net[-1], './0043.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=1024, eps=1.001e-05, track_running_stats=False))  # 89
        set_bn_weights(net[-1], './0342.gamma_2.json')
        set_biases(net[-1], './0342.beta_2.json')
        set_var(net[-1], './0342.var_2.json')
        set_mean(net[-1], './0342.mean_2.json')
        
        net.append(nn.ReLU())  # 90 add relu

        net.append(nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0))  # 91
        set_weights(net[-1], './0048.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 92
        set_bn_weights(net[-1], './0349.gamma_4.json')
        set_biases(net[-1], './0349.beta_4.json')
        set_var(net[-1], './0349.var_4.json')
        set_mean(net[-1], './0349.mean_4.json')
        net.append(nn.ReLU())  # 93

        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))  # 94
        set_weights(net[-1], './0050.weights_2.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 95
        set_bn_weights(net[-1], './0349.gamma_5.json')
        set_biases(net[-1], './0349.beta_5.json')
        set_var(net[-1], './0349.var_5.json')
        set_mean(net[-1], './0349.mean_5.json')
        net.append(nn.ReLU())  # 96

        net.append(nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1, padding=0))  # 97
        set_weights(net[-1], './0043.weights_2.json')
        net.append(nn.BatchNorm2d(num_features=1024, eps=1.001e-05, track_running_stats=False))  # 98
        set_bn_weights(net[-1], './0342.gamma_3.json')
        set_biases(net[-1], './0342.beta_3.json')
        set_var(net[-1], './0342.var_3.json')
        set_mean(net[-1], './0342.mean_3.json')
        
        net.append(nn.ReLU())  # 99 add relu

        net.append(nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0))  # 100
        set_weights(net[-1], './0048.weights_2.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 101
        set_bn_weights(net[-1], './0349.gamma_6.json')
        set_biases(net[-1], './0349.beta_6.json')
        set_var(net[-1], './0349.var_6.json')
        set_mean(net[-1], './0349.mean_6.json')
        net.append(nn.ReLU())  # 102

        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))  # 103
        set_weights(net[-1], './0050.weights_3.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 104
        set_bn_weights(net[-1], './0349.gamma_7.json')
        set_biases(net[-1], './0349.beta_7.json')
        set_var(net[-1], './0349.var_7.json')
        set_mean(net[-1], './0349.mean_7.json')
        net.append(nn.ReLU())  # 105

        net.append(nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1, padding=0))  # 106
        set_weights(net[-1], './0043.weights_3.json')
        net.append(nn.BatchNorm2d(num_features=1024, eps=1.001e-05, track_running_stats=False))  # 107
        set_bn_weights(net[-1], './0342.gamma_4.json')
        set_biases(net[-1], './0342.beta_4.json')
        set_var(net[-1], './0342.var_4.json')
        set_mean(net[-1], './0342.mean_4.json')
        
        net.append(nn.ReLU())  # 108 add relu

        net.append(nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0))  # 109
        set_weights(net[-1], './0048.weights_3.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 110
        set_bn_weights(net[-1], './0349.gamma_8.json')
        set_biases(net[-1], './0349.beta_8.json')
        set_var(net[-1], './0349.var_8.json')
        set_mean(net[-1], './0349.mean_8.json')
        net.append(nn.ReLU())  # 111

        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))  # 112
        set_weights(net[-1], './0050.weights_4.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 113
        set_bn_weights(net[-1], './0349.gamma_9.json')
        set_biases(net[-1], './0349.beta_9.json')
        set_var(net[-1], './0349.var_9.json')
        set_mean(net[-1], './0349.mean_9.json')
        net.append(nn.ReLU())  # 114

        net.append(nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1, padding=0))  # 115
        set_weights(net[-1], './0043.weights_4.json')
        net.append(nn.BatchNorm2d(num_features=1024, eps=1.001e-05, track_running_stats=False))  # 116
        set_bn_weights(net[-1], './0342.gamma_5.json')
        set_biases(net[-1], './0342.beta_5.json')
        set_var(net[-1], './0342.var_5.json')
        set_mean(net[-1], './0342.mean_5.json')
        
        net.append(nn.ReLU())  # 117 add relu

        net.append(nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0))  # 118
        set_weights(net[-1], './0048.weights_4.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 119
        set_bn_weights(net[-1], './0349.gamma_10.json')
        set_biases(net[-1], './0349.beta_10.json')
        set_var(net[-1], './0349.var_10.json')
        set_mean(net[-1], './0349.mean_10.json')
        net.append(nn.ReLU())  # 120

        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))  # 121
        set_weights(net[-1], './0050.weights_5.json')
        net.append(nn.BatchNorm2d(num_features=256, eps=1.001e-05, track_running_stats=False))  # 122
        set_bn_weights(net[-1], './0349.gamma_11.json')
        set_biases(net[-1], './0349.beta_11.json')
        set_var(net[-1], './0349.var_11.json')
        set_mean(net[-1], './0349.mean_11.json')
        net.append(nn.ReLU())  # 123

        net.append(nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1, padding=0))  # 124
        set_weights(net[-1], './0043.weights_5.json')
        net.append(nn.BatchNorm2d(num_features=1024, eps=1.001e-05, track_running_stats=False))  # 125
        set_bn_weights(net[-1], './0342.gamma_6.json')
        set_biases(net[-1], './0342.beta_6.json')
        set_var(net[-1], './0342.var_6.json')
        set_mean(net[-1], './0342.mean_6.json')
        
        net.append(nn.ReLU())  # 126 add relu

        # ===================================

        net.append(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=2, padding=0))  # 127
        set_weights(net[-1], './0051.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 128
        set_bn_weights(net[-1], './0341.gamma_0.json')
        set_biases(net[-1], './0341.beta_0.json')
        set_var(net[-1], './0341.var_0.json')
        set_mean(net[-1], './0341.mean_0.json')
        net.append(nn.ReLU())  # 129

        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))  # 130
        set_weights(net[-1], './0054.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 131
        set_bn_weights(net[-1], './0341.gamma_1.json')
        set_biases(net[-1], './0341.beta_1.json')
        set_var(net[-1], './0341.var_1.json')
        set_mean(net[-1], './0341.mean_1.json')
        net.append(nn.ReLU())  # 132

        net.append(nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1, padding=0))  # 133
        set_weights(net[-1], './0049.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=2048, eps=1.001e-05, track_running_stats=False))  # 134
        set_bn_weights(net[-1], './0345.gamma_0.json')
        set_biases(net[-1], './0345.beta_0.json')
        set_var(net[-1], './0345.var_0.json')
        set_mean(net[-1], './0345.mean_0.json')
        
        net.append(nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=2, padding=0))  # 135
        set_weights(net[-1], './0047.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=2048, eps=1.001e-05, track_running_stats=False))  # 136
        set_bn_weights(net[-1], './0345.gamma_1.json')
        set_biases(net[-1], './0345.beta_1.json')
        set_var(net[-1], './0345.var_1.json')
        set_mean(net[-1], './0345.mean_1.json')
        
        net.append(nn.ReLU())  # 137 add relu

        net.append(nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0))  # 138
        set_weights(net[-1], './0046.weights_0.json')
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 139
        set_bn_weights(net[-1], './0341.gamma_2.json')
        set_biases(net[-1], './0341.beta_2.json')
        set_var(net[-1], './0341.var_2.json')
        set_mean(net[-1], './0341.mean_2.json')
        net.append(nn.ReLU())  # 140

        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))  # 141
        set_weights(net[-1], './0054.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 142
        set_bn_weights(net[-1], './0341.gamma_3.json')
        set_biases(net[-1], './0341.beta_3.json')
        set_var(net[-1], './0341.var_3.json')
        set_mean(net[-1], './0341.mean_3.json')
        net.append(nn.ReLU())  # 143

        net.append(nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1, padding=0))  # 144
        set_weights(net[-1], './0049.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=2048, eps=1.001e-05, track_running_stats=False))  # 145
        set_bn_weights(net[-1], './0345.gamma_2.json')
        set_biases(net[-1], './0345.beta_2.json')
        set_var(net[-1], './0345.var_2.json')
        set_mean(net[-1], './0345.mean_2.json')
        
        net.append(nn.ReLU())  # 146 add relu

        net.append(nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0))  # 147
        set_weights(net[-1], './0046.weights_1.json')
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 148
        set_bn_weights(net[-1], './0341.gamma_4.json')
        set_biases(net[-1], './0341.beta_4.json')
        set_var(net[-1], './0341.var_4.json')
        set_mean(net[-1], './0341.mean_4.json')
        net.append(nn.ReLU())  # 149

        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))  # 150
        set_weights(net[-1], './0054.weights_2.json')
        net.append(nn.BatchNorm2d(num_features=512, eps=1.001e-05, track_running_stats=False))  # 151
        set_bn_weights(net[-1], './0341.gamma_5.json')
        set_biases(net[-1], './0341.beta_5.json')
        set_var(net[-1], './0341.var_5.json')
        set_mean(net[-1], './0341.mean_5.json')
        net.append(nn.ReLU())  # 152

        net.append(nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1, padding=0))  # 153
        set_weights(net[-1], './0049.weights_2.json')
        net.append(nn.BatchNorm2d(num_features=2048, eps=1.001e-05, track_running_stats=False))  # 154
        set_bn_weights(net[-1], './0345.gamma_3.json')
        set_biases(net[-1], './0345.beta_3.json')
        set_var(net[-1], './0345.var_3.json')
        set_mean(net[-1], './0345.mean_3.json')
        
        net.append(nn.ReLU())  # 155 add relu

        net.append(nn.AvgPool2d(kernel_size=7))  # 156 
        # Sum + Divide = AvgPool
        # kernel size of AvgPool can get from Divide 
        net.append(nn.Flatten())  # 157
        
        net.append(nn.Linear(in_features=512, out_features=1000))  # 158
        set_linear_weights(net[-1], './0257.params_0.json')
        set_biases(net[-1], './0114.params_0.json')
        
        net.append(nn.Identity()) # 159
        self.net = nn.ModuleList(net)

    def forward(self, x):
        length = len(self.net)
        input_list = [[], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [4], [13], [14, 12], [15], [16], [17], [18], [19], [20], [21], [22], [15, 23], [24], [25], [26], [27], [28], [29], [30], [31], [24, 32], [33], [34], [35], [36], [37], [38], [39], [40], [33], [42], [43, 41], [44], [45], [46], [47], [48], [49], [50], [51], [44, 52], [53], [54], [55], [56], [57], [58], [59], [60], [53, 61], [62], [63], [64], [65], [66], [67], [68], [69], [62, 70], [71], [72], [73], [74], [75], [76], [77], [78], [71], [80], [81, 79], [82], [83], [84], [85], [86], [87], [88], [89], [82, 90], [91], [92], [93], [94], [95], [96], [97], [98], [91, 99], [100], [101], [102], [103], [104], [105], [106], [107], [100, 108], [109], [110], [111], [112], [113], [114], [115], [116], [109, 117], [118], [119], [120], [121], [122], [123], [124], [125], [118, 126], [127], [128], [129], [130], [131], [132], [133], [134], [127], [136], [137, 135], [138], [139], [140], [141], [142], [143], [144], [145], [138, 146], [147], [148], [149], [150], [151], [152], [153], [154], [147, 155], [156], [157], [158], [159]]
        out = [x for i in range(length)]
        for i in range(length):
            # print('{}: {}'.format(i, type(self.net[i])))
            if i == 0:
                out[i] = self.net[i](x)
            elif len(input_list[i]) == 1:
                out[i] = self.net[i](out[input_list[i][0]])
            elif len(input_list[i]) == 2:
                idx_0 = input_list[i][0]
                idx_1 = input_list[i][1]
                input_0 = out[idx_0]
                input_1 = out[idx_1]
                out[i] = self.net[i](input_0 + input_1)
            else:
                assert 'wrong input list' and False
            #if i == 3:
            #    print(out[3])
        return out[length-1]


def validate():
    vgg = MyResNet(num_classes=1001)

    input_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/imagenet_part/"
    files = os.listdir(input_dir)
    same_count = 0
    not_same_count = 0
    for f in files:
        if f.endswith(".BIN"):
            print(f)
            f = os.path.join(input_dir, f)
            status, output = cmd("./resnet50_nnfusion_strip {}".format(f))
            index_1 = output[output.find("Result: ")+8:]
            index_1 = index_1[:index_1.find("\n")].strip()
            max_index_1 = int(index_1)
            val_1 = output[output.find("Confidence: ")+12:]
            val_1 = val_1[:val_1.find("\n")].strip()
            max_val_1 = float(val_1)
            print(max_index_1, max_val_1)

            with open(f, 'br') as f:
                bin_data = f.read()
                np_arr = np.frombuffer(bin_data, dtype=np.float32)
                # print(np_arr.shape)
                np_arr = np_arr.reshape(224, 224, 3)
                np_arr = np.transpose(np_arr, (2, 0, 1))
                np_arr = np_arr.reshape((1, 3, 224, 224))
                x = torch.Tensor(np_arr)
                out = vgg(x)
                max_index_2 = np.argmax(out.detach().numpy())
                max_val_2 = out.detach().numpy()[0, max_index_2]
                print(max_index_2, max_val_2)
                if abs(max_val_1-max_val_2) < 0.1:
                    print('same')
                    same_count += 1
                else:
                    print('not same')
                    not_same_count += 1 
    print('same count', same_count)
    print('not same count', not_same_count)
    exit(0)


if __name__ == "__main__":
    validate()

    with open("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion_resnet/cat1.bin", 'br') as f:
        bin_data = f.read()
        np_arr = np.frombuffer(bin_data, dtype=np.float32)
        print(np_arr.shape)
        np_arr = np_arr.reshape(224, 224, 3)
        np_arr = np.transpose(np_arr, (2, 0, 1))
        np_arr = np_arr.reshape((1, 3, 224, 224))
        x = torch.Tensor(np_arr)
        print(x.shape)

    time1 = time.time()
    print('building the model:', end=' ')
    vgg = MyResNet(num_classes=1001)
    time2 = time.time()
    print('{}s'.format(time2 - time1))

    print('predicting the label:', end=' ')
    out = vgg(x)
    time3 = time.time()
    print('{}s'.format(time3 - time2))

    print(out.size())
    print(type(out))
    max_index = np.argmax(out.detach().numpy())
    print(max_index)
    print(out)
    print(out.detach().numpy()[0, max_index])

    
