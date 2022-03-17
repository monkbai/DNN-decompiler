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
        net = []
        net.append(nn.Identity())  # 0
        # 444e30
        net.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))  # 1
        set_weights(net[-1], './0093.weights_0.json')
        set_biases(net[-1], './0093.biases_0.json')
        net.append(nn.ReLU())  # 2
        net.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 3
        # 43cb60, 0
        net.append(nn.Identity())  # 4
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # 5
        set_weights(net[-1], './0097.weights_0.json')
        set_biases(net[-1], './0097.biases_0.json')
        net.append(nn.ReLU())  # 6
        # 430940, 0
        net.append(nn.Identity())  # 7
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # 8
        set_weights(net[-1], './0068.weights_0.json')
        set_biases(net[-1], './0068.biases_0.json')
        net.append(nn.Identity())  # 9 add
        net.append(nn.ReLU())  # 10
        # 43cb60, 1
        net.append(nn.Identity())  # 11
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # 12
        set_weights(net[-1], './0097.weights_1.json')
        set_biases(net[-1], './0097.biases_1.json')
        net.append(nn.ReLU())  # 13
        # 430940, 1
        net.append(nn.Identity())  # 14
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))  # 15
        set_weights(net[-1], './0068.weights_1.json')
        set_biases(net[-1], './0068.biases_1.json')
        net.append(nn.Identity())  # 16 add
        net.append(nn.ReLU())  # 17
        # 413860
        net.append(nn.Identity())  # 18
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))  # 19
        set_weights(net[-1], './0101.weights_0.json')
        set_biases(net[-1], './0101.biases_0.json')
        net.append(nn.ReLU())  # 20
        # 438930
        net.append(nn.Identity())  # 21
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # 22
        set_weights(net[-1], './0056.weights_0.json')
        set_biases(net[-1], './0056.biases_0.json')
        # 42a470
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0))  # 23
        set_weights(net[-1], './0071.weights_0.json')
        set_biases(net[-1], './0071.biases_0.json')
        net.append(nn.Identity())  # 24 add
        net.append(nn.ReLU())  # 25
        # 42c660
        net.append(nn.Identity())  # 26
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # 27
        set_weights(net[-1], './0105.weights_0.json')
        set_biases(net[-1], './0105.biases_0.json')
        net.append(nn.ReLU())  # 28
        # 40f1e0
        net.append(nn.Identity())  # 29
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # 30
        set_weights(net[-1], './0075.weights_0.json')
        set_biases(net[-1], './0075.biases_0.json')
        net.append(nn.Identity())  # 31 add
        net.append(nn.ReLU())  # 32
        # 40b360
        net.append(nn.Identity())  # 33
        net.append(nn.Identity())  # 34
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1))  # 35
        set_weights(net[-1], './0109.weights_0.json')
        set_biases(net[-1], './0109.biases_0.json')
        net.append(nn.ReLU())  # 36
        # 41c830
        net.append(nn.Identity())  # 37
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))  # 38
        set_weights(net[-1], './0060.weights_0.json')
        set_biases(net[-1], './0060.biases_0.json')
        # 423330
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0))  # 39
        set_weights(net[-1], './0078.weights_0.json')
        set_biases(net[-1], './0078.biases_0.json')
        net.append(nn.Identity())  # 40 add
        net.append(nn.ReLU())  # 41
        # 434e70
        net.append(nn.Identity())  # 42
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))  # 43
        set_weights(net[-1], './0113.weights_0.json')
        set_biases(net[-1], './0113.biases_0.json')
        net.append(nn.ReLU())  # 44
        # 440cf0
        net.append(nn.Identity())  # 45
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))  # 46
        set_weights(net[-1], './0082.weights_0.json')
        set_biases(net[-1], './0082.biases_0.json')
        net.append(nn.Identity())  # 47 add
        net.append(nn.ReLU())  # 48
        # 418a00
        net.append(nn.Identity())  # 49
        net.append(nn.Identity())  # 50
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1))  # 51
        set_weights(net[-1], './0117.weights_0.json')
        set_biases(net[-1], './0117.biases_0.json')
        net.append(nn.ReLU())  # 52
        # 404810
        net.append(nn.Identity())  # 53
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))  # 54
        set_weights(net[-1], './0064.weights_0.json')
        set_biases(net[-1], './0064.biases_0.json')
        # 401550
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0))  # 55
        set_weights(net[-1], './0085.weights_0.json')
        set_biases(net[-1], './0085.biases_0.json')
        net.append(nn.Identity())  # 56 add
        net.append(nn.ReLU())  # 57
        # 407d60
        net.append(nn.Identity())  # 58
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))  # 59
        set_weights(net[-1], './0121.weights_0.json')
        set_biases(net[-1], './0121.biases_0.json')
        net.append(nn.ReLU())  # 60
        # 41f8e0
        net.append(nn.Identity())  # 61
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))  # 62
        set_weights(net[-1], './0089.weights_0.json')
        set_biases(net[-1], './0089.biases_0.json')
        net.append(nn.Identity())  # 63 add
        net.append(nn.ReLU())  # 64
        #
        net.append(nn.AvgPool2d(kernel_size=7, stride=1))  # 65
        net.append(nn.Flatten())  # 66
        # 41bea0
        net.append(nn.Linear(in_features=512, out_features=1000))  # 67
        set_weights(net[-1], './0124.dense_weights_0.json')
        set_biases(net[-1], './0124.biases_0.json')
        self.net = nn.ModuleList(net)

    def forward(self, x):
        length = len(self.net)
        # input_list = [[], [0], [1], [2], [3], [4], [5], [6], [7], [8], [3, 9], [10], [11], [12], [13], [14], [15], [10, 16], [17], [17], [19], [20], [21], [22], [23], [18, 24], [25], [26], [27], [28], [29], [30], [25, 31], [32], [32], [34], [35], [36], [37], [38], [39], [33, 40], [41], [42], [43], [44], [45], [46], [41, 47], [48], [48], [50], [51], [52], [53], [54], [55], [49, 56], [57], [58], [59], [60], [61], [62], [57, 63], [64], [65], [66]]
        #input_list = [[], [0], [1], [2], [3], [4], [5], [6], [7], [8], [3, 9], [10], [11], [12], [13], [14], [15], [10, 16], [17], [17], [19], [20], [21], [18], [23], [22, 24], [25], [26], [27], [28], [29], [30], [25, 31], [32], [32], [34], [35], [36], [37], [33], [39], [38, 40], [41], [42], [43], [44], [45], [46], [41, 47], [48], [48], [50], [51], [52], [53], [49], [55], [54, 56], [57], [58], [59], [60], [61], [62], [57, 63], [64], [65], [66]]
        input_list = [[], [0], [1], [2], [3], [4], [5], [6], [7], [3, 8], [9], [10], [11], [12], [13], [14], [10, 15], [16], [17], [17], [19], [20], [21], [18], [22, 23], [24], [25], [26], [27], [28], [29], [25, 30], [31], [32], [32], [34], [35], [36], [37], [33], [38, 39], [40], [41], [42], [43], [44], [45], [41, 46], [47], [48], [48], [50], [51], [52], [53], [49], [54, 55], [56], [57], [58], [59], [60], [61], [57, 62], [63], [64], [65], [66]]
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
            #if i == 24:  # debug
            #    print(out[24])
        return out[length-1]


if __name__ == "__main__":
    # x = torch.rand(size=(1, 3, 224, 224))
    with open("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.9.dev/resnet18_tvm_O3/cat.bin", 'br') as f:
        bin_data = f.read()
        np_arr = np.frombuffer(bin_data, dtype=np.float32)
        print(np_arr.shape)
        np_arr = np_arr.reshape(3, 224, 224)
        np_arr = np_arr.reshape((1, 3, 224, 224))
        x = torch.Tensor(np_arr)
        print(x.shape)

    time1 = time.time()
    print('building the model:', end=' ')
    vgg = MyResNet(num_classes=1000)
    time2 = time.time()
    print('{}s'.format(time2 - time1))

    print('predicting the label:', end=' ')
    out = vgg(x)
    time3 = time.time()
    print('{}s'.format(time3 - time2))

    #print(out.size())
    #print(type(out))
    max_index = np.argmax(out.detach().numpy())
    print(max_index)
    # print(out)
    print(out.detach().numpy()[0, max_index])
    exit(0)

    # Input to the model
    vgg.eval()
    batch_size = 1
    torch_out = vgg(x)

    # Export the model
    torch.onnx.export(vgg,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "resnet18_tvmO3_rebuild.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      )
    exit(0)
