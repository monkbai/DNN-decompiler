import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import numpy as np
import json

total_size = 0


def get_size(w):
    res = 1 
    for j in w.shape:
        res *= j
    return res


def read_json(json_path: str):
    global total_size
    with open(json_path, 'r') as f:
        j_txt = f.read()
        list_obj = json.loads(s=j_txt)
        arr_obj = np.array(list_obj, dtype=np.float32)
        tensor_obj = torch.from_numpy(arr_obj)
        total_size += get_size(tensor_obj)
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


# Building Blocks
def conv_bn(inp, oup, stride, weights_path: str, bias_path: str):
    blk = nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        # nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
    set_weights(blk[0], weights_path)
    set_biases(blk[0], bias_path)
    return blk


def conv_1x1_bn(inp, oup, weights_path: str, bias_path: str):
    blk = nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        # nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
    set_weights(blk[0], weights_path)
    set_biases(blk[0], bias_path)
    return blk


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel, weight_path_list: list, bias_path_list: list):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=True),
                # nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            set_weights(self.banch2[0], weight_path_list[0])
            set_biases(self.banch2[0], bias_path_list[0])
            set_weights(self.banch2[2], weight_path_list[1])
            set_biases(self.banch2[2], bias_path_list[1])
            set_weights(self.banch2[3], weight_path_list[2])
            set_biases(self.banch2[3], bias_path_list[2])
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True),
                # nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            set_weights(self.banch1[0], weight_path_list[0])
            set_biases(self.banch1[0], bias_path_list[0])
            set_weights(self.banch1[1], weight_path_list[1])
            set_biases(self.banch1[1], bias_path_list[1])

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=True),
                # nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=True),
                # nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            set_weights(self.banch2[0], weight_path_list[2])
            set_biases(self.banch2[0], bias_path_list[2])
            set_weights(self.banch2[2], weight_path_list[3])
            set_biases(self.banch2[2], bias_path_list[3])
            set_weights(self.banch2[3], weight_path_list[4])
            set_biases(self.banch2[3], bias_path_list[4])

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2, './0010.weights_0.json', './0010.biases_0.json')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # building inverted residual blocks
        self.features = [
            InvertedResidual(24, 116, 2, 2,
                             ['./0012.weights_0.json',
                              './0013.weights_0.json',
                              './0014.weights_0.json',
                              './0015.weights_0.json',
                              './0016.weights_0.json'],
                             ['./0012.biases_0.json',
                              './0013.biases_0.json',
                              './0014.biases_0.json',
                              './0015.biases_0.json',
                              './0016.biases_0.json']),
            InvertedResidual(116, 116, 1, 1,
                             ['./0022.weights_0.json',
                              './0023.weights_0.json',
                              './0022.weights_1.json'],
                             ['./0022.biases_0.json',
                              './0023.biases_0.json',
                              './0022.biases_1.json']),
            InvertedResidual(116, 116, 1, 1,
                             ['./0022.weights_2.json',
                              './0023.weights_1.json',
                              './0022.weights_3.json'],
                             ['./0022.biases_2.json',
                              './0023.biases_1.json',
                              './0022.biases_3.json']),
            InvertedResidual(116, 116, 1, 1,
                             ['./0022.weights_4.json',
                              './0023.weights_2.json',
                              './0022.weights_5.json'],
                             ['./0022.biases_4.json',
                              './0023.biases_2.json',
                              './0022.biases_5.json']),

            InvertedResidual(116, 232, 2, 2,
                             ['./0028.weights_0.json',
                              './0029.weights_0.json',
                              './0030.weights_0.json',
                              './0028.weights_1.json',
                              './0029.weights_1.json'],
                             ['./0028.biases_0.json',
                              './0029.biases_0.json',
                              './0030.biases_0.json',
                              './0028.biases_1.json',
                              './0029.biases_1.json']),
            InvertedResidual(232, 232, 1, 1,
                             ['./0036.weights_0.json',
                              './0037.weights_0.json',
                              './0036.weights_1.json'],
                             ['./0036.biases_0.json',
                              './0037.biases_0.json',
                              './0036.biases_1.json']),
            InvertedResidual(232, 232, 1, 1,
                             ['./0036.weights_2.json',
                              './0037.weights_1.json',
                              './0036.weights_3.json'],
                             ['./0036.biases_2.json',
                              './0037.biases_1.json',
                              './0036.biases_3.json']),
            InvertedResidual(232, 232, 1, 1,
                             ['./0036.weights_4.json',
                              './0037.weights_2.json',
                              './0036.weights_5.json'],
                             ['./0036.biases_4.json',
                              './0037.biases_2.json',
                              './0036.biases_5.json']),
            InvertedResidual(232, 232, 1, 1,
                             ['./0036.weights_6.json',
                              './0037.weights_3.json',
                              './0036.weights_7.json'],
                             ['./0036.biases_6.json',
                              './0037.biases_3.json',
                              './0036.biases_7.json']),
            InvertedResidual(232, 232, 1, 1,
                             ['./0036.weights_8.json',
                              './0037.weights_4.json',
                              './0036.weights_9.json'],
                             ['./0036.biases_8.json',
                              './0037.biases_4.json',
                              './0036.biases_9.json']),
            InvertedResidual(232, 232, 1, 1,
                             ['./0036.weights_10.json',
                              './0037.weights_5.json',
                              './0036.weights_11.json'],
                             ['./0036.biases_10.json',
                              './0037.biases_5.json',
                              './0036.biases_11.json']),
            InvertedResidual(232, 232, 1, 1,
                             ['./0036.weights_12.json',
                              './0037.weights_6.json',
                              './0036.weights_13.json'],
                             ['./0036.biases_12.json',
                              './0037.biases_6.json',
                              './0036.biases_13.json']),

            InvertedResidual(232, 464, 2, 2,
                             ['./0042.weights_0.json',
                              './0043.weights_0.json',
                              './0044.weights_0.json',
                              './0042.weights_1.json',
                              './0043.weights_1.json'],
                             ['./0042.biases_0.json',
                              './0043.biases_0.json',
                              './0044.biases_0.json',
                              './0042.biases_1.json',
                              './0043.biases_1.json']),
            InvertedResidual(464, 464, 1, 1,
                             ['./0050.weights_0.json',
                              './0051.weights_0.json',
                              './0050.weights_1.json'],
                             ['./0050.biases_0.json',
                              './0051.biases_0.json',
                              './0050.biases_1.json']),
            InvertedResidual(464, 464, 1, 1,
                             ['./0050.weights_2.json',
                              './0051.weights_1.json',
                              './0050.weights_3.json'],
                             ['./0050.biases_2.json',
                              './0051.biases_1.json',
                              './0050.biases_3.json']),
            InvertedResidual(464, 464, 1, 1,
                             ['./0050.weights_4.json',
                              './0051.weights_2.json',
                              './0050.weights_5.json'],
                             ['./0050.biases_4.json',
                              './0051.biases_2.json',
                              './0050.biases_5.json']),
        ]
        # for idxstage in range(len(self.stage_repeats)):
        #     numrepeat = self.stage_repeats[idxstage]
        #     output_channel = self.stage_out_channels[idxstage + 2]
        #     for i in range(numrepeat):
        #         if i == 0:
        #             # inp, oup, stride, benchmodel):
        #             # self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
        #             print('input_channel: {}, output_channel: {}, 2, 2'.format(input_channel, output_channel))
        #         else:
        #             # self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
        #             print('input_channel: {}, output_channel: {}, 1, 1'.format(input_channel, output_channel))
        #         input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1], './0056.weights_0.json', './0056.biases_0.json')
        self.globalpool = nn.Sequential(nn.AvgPool2d(7))

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))
        set_weights(self.classifier[0], './0058.params_0.json')
        set_biases(self.classifier[0], './0058.biases_0.json')

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x


def shufflenetv2(width_mult=1.):
    model = ShuffleNetV2(width_mult=width_mult)
    return model


if __name__ == "__main__":
    """Testing
    """
    input_cat = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2021/shufflenet_v2/cat.bin"
    if len(sys.argv) == 2:
        input_cat = sys.argv[1]
        
    model = ShuffleNetV2()
    # print(model)
    # input = torch.randn(1, 3, 224, 224)
    with open(input_cat, 'br') as f:
        bin_data = f.read()
        np_arr = np.frombuffer(bin_data, dtype=np.float32)
        # print(np_arr.shape)
        np_arr = np_arr.reshape(3, 224, 224)
        np_arr = np_arr.reshape((1, 3, 224, 224))
        x = torch.Tensor(np_arr)
        # print(x.shape)
    input = x
    out = model(input)

    max_index = np.argmax(out.detach().numpy())
    print("Result:", max_index)
    # print(out)
    print("Confidence:", out.detach().numpy()[0, max_index])
    # print(total_size)
    exit(0)