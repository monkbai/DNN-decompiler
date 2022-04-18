import torch
import torch.nn as nn
import numpy as np
import json
import math
import torch.functional as F


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


# ==============================================


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        #nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        #nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, weights_path_list: list, bias_path_list: list):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            assert len(weights_path_list) == len(bias_path_list) == 2
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(oup),
            )
            set_weights(self.conv[0], weights_path_list[0])
            set_biases(self.conv[0], bias_path_list[0])
            set_weights(self.conv[2], weights_path_list[1])
            set_biases(self.conv[2], bias_path_list[1])
        else:
            assert len(weights_path_list) == len(bias_path_list) == 3
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(oup),
            )
            set_weights(self.conv[0], weights_path_list[0])
            set_biases(self.conv[0], bias_path_list[0])
            set_weights(self.conv[2], weights_path_list[1])
            set_biases(self.conv[2], bias_path_list[1])
            set_weights(self.conv[4], weights_path_list[2])
            set_biases(self.conv[4], bias_path_list[2])

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()

        # building first layer
        layers = [conv_3x3_bn(3, 32, 2)]
        set_weights(layers[0][0], '0063.weights_0.json')
        set_biases(layers[0][0], '0063.biases_0.json')

        layers.append(InvertedResidual(32, 16, 1, 1, ['0091.weights_0.json', '0026.weights_0.json'],
                                                     ['0091.biases_0.json', '0026.biases_0.json']))

        layers.append(InvertedResidual(16, 24, 2, 6, ['0066.weights_0.json', '0095.weights_0.json', '0029.weights_0.json'],
                                                     ['0066.biases_0.json', '0095.biases_0.json', '0029.biases_0.json']))
        layers.append(InvertedResidual(24, 24, 1, 6, ['0069.weights_0.json', '0099.weights_0.json', '0047.weights_0.json'],
                                                     ['0069.biases_0.json', '0099.biases_0.json', '0047.biases_0.json']))

        layers.append(InvertedResidual(24, 32, 2, 6, ['0069.weights_1.json', '0103.weights_0.json', '0032.weights_0.json'],
                                                     ['0069.biases_1.json', '0103.biases_0.json', '0032.biases_0.json']))
        layers.append(InvertedResidual(32, 32, 1, 6, ['0072.weights_0.json', '0107.weights_0.json', '0050.weights_0.json'],
                                                     ['0072.biases_0.json', '0107.biases_0.json', '0050.biases_0.json']))
        layers.append(InvertedResidual(32, 32, 1, 6, ['0072.weights_1.json', '0107.weights_1.json', '0050.weights_1.json'],
                                                     ['0072.biases_1.json', '0107.biases_1.json', '0050.biases_1.json']))

        layers.append(InvertedResidual(32, 64, 2, 6, ['0072.weights_2.json', '0111.weights_0.json', '0035.weights_0.json'],
                                                     ['0072.biases_2.json', '0111.biases_0.json', '0035.biases_0.json']))
        layers.append(InvertedResidual(64, 64, 1, 6, ['0075.weights_0.json', '0115.weights_0.json', '0053.weights_0.json'],
                                                     ['0075.biases_0.json', '0115.biases_0.json', '0053.biases_0.json']))
        layers.append(InvertedResidual(64, 64, 1, 6, ['0075.weights_1.json', '0115.weights_1.json', '0053.weights_1.json'],
                                                     ['0075.biases_1.json', '0115.biases_1.json', '0053.biases_1.json']))
        layers.append(InvertedResidual(64, 64, 1, 6, ['0075.weights_2.json', '0115.weights_2.json', '0053.weights_2.json'],
                                                     ['0075.biases_2.json', '0115.biases_2.json', '0053.biases_2.json']))

        layers.append(InvertedResidual(64, 96, 1, 6, ['0075.weights_3.json', '0115.weights_3.json', '0038.weights_0.json'],
                                                     ['0075.biases_3.json', '0115.biases_3.json', '0038.biases_0.json']))
        layers.append(InvertedResidual(96, 96, 1, 6, ['0078.weights_0.json', '0119.weights_0.json', '0056.weights_0.json'],
                                                     ['0078.biases_0.json', '0119.biases_0.json', '0056.biases_0.json']))
        layers.append(InvertedResidual(96, 96, 1, 6, ['0078.weights_1.json', '0119.weights_1.json', '0056.weights_1.json'],
                                                     ['0078.biases_1.json', '0119.biases_1.json', '0056.biases_1.json']))

        layers.append(InvertedResidual(96, 160, 2, 6, ['0078.weights_2.json', '0123.weights_0.json', '0041.weights_0.json'],
                                                      ['0078.biases_2.json', '0123.biases_0.json', '0041.biases_0.json']))
        layers.append(InvertedResidual(160, 160, 1, 6, ['0081.weights_0.json', '0127.weights_0.json', '0059.weights_0.json'],
                                                       ['0081.biases_0.json', '0127.biases_0.json', '0059.biases_0.json']))
        layers.append(InvertedResidual(160, 160, 1, 6, ['0081.weights_1.json', '0127.weights_1.json', '0059.weights_1.json'],
                                                       ['0081.biases_1.json', '0127.biases_1.json', '0059.biases_1.json']))

        layers.append(InvertedResidual(160, 320, 1, 6, ['0081.weights_2.json', '0127.weights_2.json', '0044.weights_0.json'],
                                                       ['0081.biases_2.json', '0127.biases_2.json', '0044.biases_0.json']))

        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(320, 1280)
        set_weights(self.conv[0], '0084.weights_0.json')
        set_biases(self.conv[0], '0084.biases_0.json')

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)
        set_weights(self.classifier, '0087.dense_weights_0.json')
        set_biases(self.classifier, '0087.biases_0.json')

        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()


if __name__ == '__main__':
    model = MobileNetV2()
    model.eval()
    # print(model)
    # exit(0)

    # input = torch.randn(1, 3, 224, 224)
    with open("/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.9.dev/mobilenetv2_tvm_v09_O3/cat.bin", 'br') as f:
            bin_data = f.read()
            np_arr = np.frombuffer(bin_data, dtype=np.float32)
            print(np_arr.shape)
            np_arr = np_arr.reshape(3, 224, 224)
            np_arr = np_arr.reshape((1, 3, 224, 224))

            x = torch.Tensor(np_arr)
            print(x.shape)
    input = x
    out = model(input)

    max_index = np.argmax(out.detach().numpy())
    print(max_index)
    # print(out)
    print(out.detach().numpy()[0, max_index])
    exit(0)
