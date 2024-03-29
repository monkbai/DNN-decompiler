#!/usr/bin/python3
import torch.nn as nn
import torch
import json
import numpy as np
import time


def read_json(json_path: str):
    with open(json_path, 'r') as f:
        j_txt = f.read()
        list_obj = json.loads(s=j_txt)
        arr_obj = np.array(list_obj, dtype=np.float32)
        tensor_obj = torch.from_numpy(arr_obj)
        return tensor_obj


def set_weights(module: nn.modules, json_path: str):
    # https://stackoverflow.com/a/59468760
    w = read_json(json_path)
    if len(w.shape) == 2:
        w = np.transpose(w, (1, 0))
    module.weight = torch.nn.Parameter(w)


def set_biases(module: nn.modules, json_path: str):
    # https://stackoverflow.com/a/59468760
    w = read_json(json_path)
    w = w.reshape(w.shape[1])
    module.bias = torch.nn.Parameter(w)


class SE_VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        net = []

        # block 1
        net.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1))  # 0
        set_weights(net[-1], 'Constant_17_0.json')
        set_biases(net[-1], 'Constant_16_0.json')
        net.append(nn.ReLU())  # 1
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 2

        # block 2
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))  # 3
        set_weights(net[-1], 'Constant_19_0.json')
        set_biases(net[-1], 'Constant_18_0.json')
        net.append(nn.ReLU())  # 4
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 5

        # block 3
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))  # 10
        set_weights(net[-1], 'Constant_21_0.json')
        set_biases(net[-1], 'Constant_20_0.json')
        net.append(nn.ReLU())  # 11
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))  # 12
        set_weights(net[-1], 'Constant_23_0.json')
        set_biases(net[-1], 'Constant_22_0.json')
        net.append(nn.ReLU())  # 13
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 16

        # block 4
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))  # 17
        set_weights(net[-1], 'Constant_25_0.json')
        set_biases(net[-1], 'Constant_24_0.json')
        net.append(nn.ReLU())  # 18
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 19
        set_weights(net[-1], 'Constant_27_0.json')
        set_biases(net[-1], 'Constant_26_0.json')
        net.append(nn.ReLU())  # 20
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 23

        # block 5
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 24
        set_weights(net[-1], 'Constant_29_0.json')
        set_biases(net[-1], 'Constant_28_0.json')
        net.append(nn.ReLU())  # 25
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 26
        set_weights(net[-1], 'Constant_31_0.json')
        set_biases(net[-1], 'Constant_30_0.json')
        net.append(nn.ReLU())  # 27
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 30

        # add net into class property
        self.extract_feature = nn.Sequential(*net)
        self.net = net

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=512 * 7 * 7, out_features=4096))
        set_weights(classifier[-1], 'Constant_10_0.json')
        print(classifier[-1].weight.shape)
        set_biases(classifier[-1], 'Constant_13_0.json')
        classifier.append(nn.ReLU())

        classifier.append(nn.Linear(in_features=4096, out_features=4096))
        set_weights(classifier[-1], 'Constant_11_0.json')
        set_biases(classifier[-1], 'Constant_14_0.json')
        classifier.append(nn.ReLU())

        classifier.append(nn.Linear(in_features=4096, out_features=self.num_classes))
        print(classifier[-1].weight.shape)
        set_weights(classifier[-1], 'Constant_12_0.json')
        print(classifier[-1].weight.shape)
        set_biases(classifier[-1], 'Constant_15_0.json')

        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        # for debug
        out0 = self.net[0](x) # conv 1st
        print(out0[0][0][0][0])
        out0 = self.net[1](out0) # relu
        out0 = self.net[2](out0) # max
        out0 = self.net[3](out0) # conv 2nd
        print(out0[0][0][0][0])
        out0 = self.net[4](out0) # relu
        out0 = self.net[5](out0) # max
        out0 = self.net[6](out0) # conv 3rd
        print(out0[0][0][0][0])
        out0 = self.net[7](out0) # relu
        out0 = self.net[8](out0) # conv 4th
        print(out0[0][0][0][0])
        out0 = self.net[9](out0) 
        out0 = self.net[10](out0) 
        out0 = self.net[11](out0) # conv 5th
        print(out0[0][0][0][0])
        out0 = self.net[12](out0) # relu
        out0 = self.net[13](out0) # conv 6th
        print(out0[0][0][0])
        out0 = self.net[14](out0) 
        out0 = self.net[15](out0)
        out0 = self.net[16](out0) # conv 7th
        print(out0)
        out0 = self.net[17](out0) # relu
        out0 = self.net[18](out0) # conv 8th
        print(out0[0][0][0][0])

        feature = self.extract_feature(x)
        print(feature)
        feature = feature.view(x.size(0), -1)
        classify_result = self.classifier(feature)
        return classify_result


if __name__ == "__main__":
    # x = torch.rand(size=(1, 3, 224, 224))
    with open("cat2.bin", 'br') as f:
        bin_data = f.read()
        np_arr = np.frombuffer(bin_data, dtype=np.float32)
        print(np_arr.shape)
        np_arr = np_arr.reshape(224, 224, 3)
        np_arr = np.transpose(np_arr, (2, 0, 1))
        np_arr = np_arr.reshape((1, 3, 224, 224))
        x = torch.Tensor(np_arr)
        print(x.shape)
    #print(x)
    time1 = time.time()
    print('building the model:', end=' ')
    vgg = SE_VGG(num_classes=1001)
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
