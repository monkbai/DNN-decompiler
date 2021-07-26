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
    weight = torch.nn.Parameter(w)
    module.weight = weight
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
    def __init__(self):
        super(MyResNet, self).__init__()
        net = []
        net.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))
        set_weights(net[-1], './0010.weights_0.json')
        set_biases(net[-1], '0010.biases_0.json')
        net.append(nn.ReLU())

        net.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0012.weights_0.json')
        set_biases(net[-1], '0012.biases_0.json')
        net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0013.weights_0.json')
        set_biases(net[-1], '0013.biases_0.json')
        net.append(nn.Identity())  # net.append(nn.ReLU())

        net.append(nn.ReLU())  # add relu

        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        # set_weights(net[-1], '../resnet18_tvm_O0/0080.function_409280.weights_2.json')
        set_weights(net[-1], './0012.weights_1.json')
        set_biases(net[-1], './0012.biases_1.json')
        net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0013.weights_1.json')
        set_biases(net[-1], '0013.biases_1.json')
        net.append(nn.Identity())  # net.append(nn.ReLU())

        net.append(nn.ReLU())  # add relu

        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0))
        set_weights(net[-1], './0016.weights_0.json')
        set_biases(net[-1], '0016.biases_0.json')
        net.append(nn.Identity())  # net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))  # correct the shape
        set_weights(net[-1], './0017.weights_0.json')
        set_biases(net[-1], '0017.biases_0.json')
        net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0018.weights_0.json')
        set_biases(net[-1], '0018.biases_0.json')
        net.append(nn.Identity())  # net.append(nn.ReLU())

        net.append(nn.ReLU())  # add relu

        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0020.weights_0.json')
        set_biases(net[-1], '0020.biases_0.json')
        net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0018.weights_1.json')
        set_biases(net[-1], '0018.biases_1.json')
        net.append(nn.Identity())  # net.append(nn.ReLU())

        net.append(nn.ReLU())  # add relu

        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0))
        set_weights(net[-1], './0022.weights_0.json')
        set_biases(net[-1], '0022.biases_0.json')
        net.append(nn.Identity())  # net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1))
        set_weights(net[-1], './0023.weights_0.json')
        set_biases(net[-1], '0023.biases_0.json')
        net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0024.weights_0.json')
        set_biases(net[-1], '0024.biases_0.json')
        net.append(nn.Identity())  # net.append(nn.ReLU())

        net.append(nn.ReLU())  # add relu

        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0026.weights_0.json')
        set_biases(net[-1], '0026.biases_0.json')
        net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0024.weights_1.json')
        set_biases(net[-1], '0024.biases_1.json')
        net.append(nn.Identity())  # net.append(nn.ReLU())

        net.append(nn.ReLU())  # add relu

        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0))
        set_weights(net[-1], './0028.weights_0.json')
        set_biases(net[-1], '0028.biases_0.json')
        net.append(nn.Identity())  #  net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1))
        set_weights(net[-1], './0029.weights_0.json')
        set_biases(net[-1], '0029.biases_0.json')
        net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0030.weights_0.json')
        set_biases(net[-1], '0030.biases_0.json')
        net.append(nn.Identity())  # net.append(nn.ReLU())

        net.append(nn.ReLU())  # add relu

        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0032.weights_0.json')
        set_biases(net[-1], '0032.biases_0.json')
        net.append(nn.ReLU())

        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0030.weights_1.json')
        set_biases(net[-1], '0030.biases_1.json')
        net.append(nn.Identity())  # net.append(nn.ReLU())

        net.append(nn.ReLU())  # add relu

        net.append(nn.AvgPool2d(kernel_size=7, stride=1))
        net.append(nn.Flatten())

        net.append(nn.Linear(in_features=512, out_features=1000))
        set_weights(net[-1], './0035.params_0.json')
        set_biases(net[-1], './0036.params_0.json')
        self.net = net

    def forward(self, x):
        length = len(self.net)
        input_list = [[], [0], [1], [2], [3], [4], [5], [6, 2], [7], [8], [9], [10], [11, 7], [12], [13], [12], [15], [16], [17], [18, 14], [19], [20], [21], [22], [23, 19], [24], [25], [24], [27], [28], [29], [30, 26], [31], [32], [33], [34], [35, 31], [36], [37], [36], [39], [40], [41], [42, 38], [43], [44], [45], [46], [47, 43], [48], [49], [50]]
        out = [x for i in range(length)]
        for i in range(length):
            # print('{}: {}'.format(i, type(self.net[i])))
            if i == 0:
                out[i] = self.net[i](x)
            elif i == 51 and len(input_list[i]) == 1:
                out[i] = self.net[i](out[input_list[i][0]])
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


if __name__ == "__main__":
    # x = torch.rand(size=(1, 3, 224, 224))
    with open("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/cat.bin", 'br') as f:
        bin_data = f.read()
        np_arr = np.frombuffer(bin_data, dtype=np.float32)
        print(np_arr.shape)
        np_arr = np_arr.reshape(3, 224, 224)
        np_arr = np_arr.reshape((1, 3, 224, 224))
        x = torch.Tensor(np_arr)
        print(x.shape)

    time1 = time.time()
    print('building the model:', end=' ')
    vgg = MyResNet()
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
    # print(out)
    print(out.detach().numpy()[0, max_index])
