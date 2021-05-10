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


"""
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)
"""


class MyResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyResNet, self).__init__()
        net = []
        # 402ea0
        net.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))
        set_weights(net[-1], './0057.function_402ea0.weights_0.json')
        # 41ee50 var, 435470 gamma, 4311a0 mean, 413580 beta, 0
        net.append(nn.BatchNorm2d(num_features=64, affine=False, track_running_stats=True))
        set_bn_weights(net[-1], './0205.function_435470.gamma_0.json')
        set_biases(net[-1], './0106.function_413580.beta_0.json')
        set_var(net[-1], './0146.function_41ee50.var_0.json')
        set_mean(net[-1], './0191.function_4311a0.mean_0.json')
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # 409280, 0
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0080.function_409280.weights_0.json')
        # 41ee50 var, 435470 gamma, 4311a0 mean, 413580 beta, 1
        net.append(nn.BatchNorm2d(num_features=64, track_running_stats=True))
        set_bn_weights(net[-1], './0205.function_435470.gamma_2.json')
        set_biases(net[-1], './0106.function_413580.beta_1.json')
        set_var(net[-1], './0146.function_41ee50.var_1.json')
        set_mean(net[-1], './0191.function_4311a0.mean_1.json')
        net.append(nn.ReLU())
        # 409280, 1
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0080.function_409280.weights_1.json')
        # 41ee50 var, 435470 gamma, 4311a0 mean, 413580 beta, 2
        net.append(nn.BatchNorm2d(num_features=64, track_running_stats=True))
        set_bn_weights(net[-1], './0205.function_435470.gamma_4.json')
        set_biases(net[-1], './0106.function_413580.beta_2.json')
        set_var(net[-1], './0146.function_41ee50.var_2.json')
        set_mean(net[-1], './0191.function_4311a0.mean_2.json')
        # net.append(nn.add
        net.append(nn.ReLU())
        # 409280, 2
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0080.function_409280.weights_2.json')
        # 41ee50 var, 435470 gamma, 4311a0 mean, 413580 beta, 3
        net.append(nn.BatchNorm2d(num_features=64, track_running_stats=True))
        set_bn_weights(net[-1], './0205.function_435470.gamma_6.json')
        set_biases(net[-1], './0106.function_413580.beta_3.json')
        set_var(net[-1], './0146.function_41ee50.var_3.json')
        set_mean(net[-1], './0191.function_4311a0.mean_3.json')
        net.append(nn.ReLU())
        # 409280, 3
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0080.function_409280.weights_3.json')
        # 41ee50 var, 435470 gamma, 4311a0 mean, 413580 beta, 4
        net.append(nn.BatchNorm2d(num_features=64, track_running_stats=True))
        set_bn_weights(net[-1], './0205.function_435470.gamma_8.json')
        set_biases(net[-1], './0106.function_413580.beta_4.json')
        set_var(net[-1], './0146.function_41ee50.var_4.json')
        set_mean(net[-1], './0191.function_4311a0.mean_4.json')
        # net.append(nn.add
        net.append(nn.ReLU())
        # 0x40ff40, 0
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0))
        set_weights(net[-1], './0092.function_40ff40.weights_0.json')
        # 434b50 var, 4243d0 gamma, 4021e0 mean, 431870 beta, 0
        net.append(nn.BatchNorm2d(num_features=128, track_running_stats=True))
        set_bn_weights(net[-1], './0160.function_4243d0.gamma_0.json')
        set_biases(net[-1], './0195.function_431870.beta_0.json')
        set_var(net[-1], './0201.function_434b50.var_0.json')
        set_mean(net[-1], './0052.function_4021e0.mean_0.json')
        # net.append(nn.Conv2d(in_channels=16, out_channels=128, kernel_size=6, stride=2, padding=1))  # wrong shape
        # 0x42aec0, 0
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))  # correct the shape
        set_weights(net[-1], './0182.function_42aec0.weights_0.json')
        # 434b50 var, 4243d0 gamma, 4021e0 mean, 431870 beta, 1
        net.append(nn.BatchNorm2d(num_features=128, track_running_stats=True))
        set_bn_weights(net[-1], './0160.function_4243d0.gamma_2.json')
        set_biases(net[-1], './0195.function_431870.beta_1.json')
        set_var(net[-1], './0201.function_434b50.var_1.json')
        set_mean(net[-1], './0052.function_4021e0.mean_1.json')
        net.append(nn.ReLU())
        # 0x41c230, 0
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0144.function_41c230.weights_0.json')
        # 434b50 var, 4243d0 gamma, 4021e0 mean, 431870 beta, 2
        net.append(nn.BatchNorm2d(num_features=128, track_running_stats=True))
        set_bn_weights(net[-1], './0160.function_4243d0.gamma_4.json')
        set_biases(net[-1], './0195.function_431870.beta_2.json')
        set_var(net[-1], './0201.function_434b50.var_2.json')
        set_mean(net[-1], './0052.function_4021e0.mean_2.json')
        # net.append(nn.add
        net.append(nn.ReLU())
        # 0x41c230, 1
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0144.function_41c230.weights_1.json')
        # 434b50 var, 4243d0 gamma, 4021e0 mean, 431870 beta, 3
        net.append(nn.BatchNorm2d(num_features=128, track_running_stats=True))
        set_bn_weights(net[-1], './0160.function_4243d0.gamma_6.json')
        set_biases(net[-1], './0195.function_431870.beta_3.json')
        set_var(net[-1], './0201.function_434b50.var_3.json')
        set_mean(net[-1], './0052.function_4021e0.mean_3.json')
        net.append(nn.ReLU())
        # 0x41c230, 2
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0144.function_41c230.weights_2.json')
        # 434b50 var, 4243d0 gamma, 4021e0 mean, 431870 beta, 4
        net.append(nn.BatchNorm2d(num_features=128, track_running_stats=True))
        set_bn_weights(net[-1], './0160.function_4243d0.gamma_8.json')
        set_biases(net[-1], './0195.function_431870.beta_4.json')
        set_var(net[-1], './0201.function_434b50.var_4.json')
        set_mean(net[-1], './0052.function_4021e0.mean_4.json')
        # net.append(nn.add
        net.append(nn.ReLU())
        # 0x40d190, 0
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0))
        set_weights(net[-1], './0088.function_40d190.weights_0.json')
        # 419630 var, 413fe0 gamma, 411880 mean, 4115a0 beta, 0
        net.append(nn.BatchNorm2d(num_features=256, track_running_stats=True))
        set_bn_weights(net[-1], './0110.function_413fe0.gamma_0.json')
        set_biases(net[-1], './0094.function_4115a0.beta_0.json')
        set_var(net[-1], './0129.function_419630.var_0.json')
        set_mean(net[-1], './0096.function_411880.mean_0.json')
        # 0x420600, 0
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1))
        set_weights(net[-1], './0151.function_420600.weights_0.json')
        # 419630 var, 413fe0 gamma, 411880 mean, 4115a0 beta, 1
        net.append(nn.BatchNorm2d(num_features=256, track_running_stats=True))
        set_bn_weights(net[-1], './0110.function_413fe0.gamma_2.json')
        set_biases(net[-1], './0094.function_4115a0.beta_1.json')
        set_var(net[-1], './0129.function_419630.var_1.json')
        set_mean(net[-1], './0096.function_411880.mean_1.json')
        net.append(nn.ReLU())
        # 0x42ec00, 0
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0189.function_42ec00.weights_0.json')
        # 419630 var, 413fe0 gamma, 411880 mean, 4115a0 beta, 2
        net.append(nn.BatchNorm2d(num_features=256, track_running_stats=True))
        set_bn_weights(net[-1], './0110.function_413fe0.gamma_4.json')
        set_biases(net[-1], './0094.function_4115a0.beta_2.json')
        set_var(net[-1], './0129.function_419630.var_2.json')
        set_mean(net[-1], './0096.function_411880.mean_2.json')
        # net.append(nn.add
        net.append(nn.ReLU())
        # 0x42ec00, 1
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0189.function_42ec00.weights_1.json')
        # 419630 var, 413fe0 gamma, 411880 mean, 4115a0 beta, 3
        net.append(nn.BatchNorm2d(num_features=256, track_running_stats=True))
        set_bn_weights(net[-1], './0110.function_413fe0.gamma_6.json')
        set_biases(net[-1], './0094.function_4115a0.beta_3.json')
        set_var(net[-1], './0129.function_419630.var_3.json')
        set_mean(net[-1], './0096.function_411880.mean_3.json')
        net.append(nn.ReLU())
        # 0x42ec00, 2
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0189.function_42ec00.weights_2.json')
        # 419630 var, 413fe0 gamma, 411880 mean, 4115a0 beta, 4
        net.append(nn.BatchNorm2d(num_features=256, track_running_stats=True))
        set_bn_weights(net[-1], './0110.function_413fe0.gamma_8.json')
        set_biases(net[-1], './0094.function_4115a0.beta_4.json')
        set_var(net[-1], './0129.function_419630.var_4.json')
        set_mean(net[-1], './0096.function_411880.mean_4.json')
        # net.append(nn.add
        net.append(nn.ReLU())
        # 0x432d50, 0
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0))
        set_weights(net[-1], './0199.function_432d50.weights_0.json')
        # 414370 var, 414740 gamma, 424000 mean, 418d00 beta, 0
        net.append(nn.BatchNorm2d(num_features=512, track_running_stats=True))
        set_bn_weights(net[-1], './0114.function_414740.gamma_0.json')
        set_biases(net[-1], './0125.function_418d00.beta_0.json')
        set_var(net[-1], './0112.function_414370.var_0.json')
        set_mean(net[-1], './0158.function_424000.mean_0.json')
        # 0x4262e0, 0
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1))
        set_weights(net[-1], './0171.function_4262e0.weights_0.json')
        # 414370 var, 414740 gamma, 424000 mean, 418d00 beta, 1
        net.append(nn.BatchNorm2d(num_features=512, track_running_stats=True))
        set_bn_weights(net[-1], './0114.function_414740.gamma_2.json')
        set_biases(net[-1], './0125.function_418d00.beta_1.json')
        set_var(net[-1], './0112.function_414370.var_1.json')
        set_mean(net[-1], './0158.function_424000.mean_1.json')
        net.append(nn.ReLU())
        # 0x415810, 0
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0119.function_415810.weights_0.json')
        # 414370 var, 414740 gamma, 424000 mean, 418d00 beta, 2
        net.append(nn.BatchNorm2d(num_features=512, track_running_stats=True))
        set_bn_weights(net[-1], './0114.function_414740.gamma_4.json')
        set_biases(net[-1], './0125.function_418d00.beta_2.json')
        set_var(net[-1], './0112.function_414370.var_2.json')
        set_mean(net[-1], './0158.function_424000.mean_2.json')
        # net.append(nn.add
        net.append(nn.ReLU())
        # 0x415810, 1
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0119.function_415810.weights_1.json')
        # 414370 var, 414740 gamma, 424000 mean, 418d00 beta, 3
        net.append(nn.BatchNorm2d(num_features=512, track_running_stats=True))
        set_bn_weights(net[-1], './0114.function_414740.gamma_6.json')
        set_biases(net[-1], './0125.function_418d00.beta_3.json')
        set_var(net[-1], './0112.function_414370.var_3.json')
        set_mean(net[-1], './0158.function_424000.mean_3.json')
        net.append(nn.ReLU())
        # 0x415810, 2
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0119.function_415810.weights_2.json')
        # 414370 var, 414740 gamma, 424000 mean, 418d00 beta, 4
        net.append(nn.BatchNorm2d(num_features=512, track_running_stats=True))
        set_bn_weights(net[-1], './0114.function_414740.gamma_8.json')
        set_biases(net[-1], './0125.function_418d00.beta_4.json')
        set_var(net[-1], './0112.function_414370.var_4.json')
        set_mean(net[-1], './0158.function_424000.mean_4.json')
        # net.append(nn.add
        net.append(nn.ReLU())
        net.append(nn.AvgPool2d(kernel_size=7, stride=1))
        net.append(nn.Flatten())
        # 4181a0 weights, 42da10 bias add
        net.append(nn.Linear(in_features=512, out_features=1000))
        set_weights(net[-1], './0121.function_4181a0.dense_weights_0.json')
        set_biases(net[-1], '0184.function_42da10.biases_0.json')
        self.net = net

    def forward(self, x):
        length = len(self.net)
        input_list = [[], [0], [1], [2], [3], [4], [5], [6], [7], [3, 8], [9], [10], [11], [12], [13], [9, 14], [15], [16], [15], [18], [19], [20], [21], [17, 22], [23], [24], [25], [26], [27], [23, 28], [29], [30], [29], [32], [33], [34], [35], [31, 36], [37], [38], [39], [40], [41], [37, 42], [43], [44], [43], [46], [47], [48], [49], [45, 50], [51], [52], [53], [54], [55], [51, 56], [57], [58], [59], [60], [61], ]
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


if __name__ == "__main__":
    # x = torch.rand(size=(1, 3, 224, 224))
    with open('cat.bin', 'br') as f:
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

    print(out.size())
    print(type(out))
    max_index = np.argmax(out.detach().numpy())
    print(max_index)
    # print(out)
    print(out.detach().numpy()[0, max_index])
