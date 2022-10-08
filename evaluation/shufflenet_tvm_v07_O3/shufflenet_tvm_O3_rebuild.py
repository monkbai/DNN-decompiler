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


# Building Blocks
def conv_bn(inp, oup, stride, weights_path: str, bias_path: str,
            var_path='', gamma_path='', mean_path='', beta_path=''):
    blk = nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        # nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
    set_weights(blk[0], weights_path)
    if len(bias_path) > 0:
        set_biases(blk[0], bias_path)

    # set_var(blk[1], var_path)
    # set_bn_weights(blk[1], gamma_path)
    # set_mean(blk[1], mean_path)
    # set_biases(blk[1], beta_path)
    return blk


def conv_1x1_bn(inp, oup, weights_path: str, bias_path: str,
                var_path='', gamma_path='', mean_path='', beta_path=''):
    blk = nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        # nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
    set_weights(blk[0], weights_path)
    if len(bias_path) > 0:
        set_biases(blk[0], bias_path)

    # set_var(blk[1], var_path)
    # set_bn_weights(blk[1], gamma_path)
    # set_mean(blk[1], mean_path)
    # set_biases(blk[1], beta_path)
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
    def __init__(self, inp, oup, stride, benchmodel, weight_path_list: list, bias_path_list: list,
                 var_path_list=[], gamma_path_list=[], mean_path_list=[], beta_path_list=[]):
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
               #  nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            set_weights(self.banch2[0], weight_path_list[0])
            if len(bias_path_list) > 0:
                set_biases(self.banch2[0], bias_path_list[0])

            set_weights(self.banch2[2], weight_path_list[1])
            if len(bias_path_list) > 0:
                set_biases(self.banch2[2], bias_path_list[1])

            set_weights(self.banch2[3], weight_path_list[2])
            if len(bias_path_list) > 0:
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
            if len(bias_path_list) > 0:
                set_biases(self.banch1[0], bias_path_list[0])

            set_weights(self.banch1[1], weight_path_list[1])
            if len(bias_path_list) > 0:
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
            if len(bias_path_list) > 0:
                set_biases(self.banch2[0], bias_path_list[2])

            set_weights(self.banch2[2], weight_path_list[3])
            if len(bias_path_list) > 0:
                set_biases(self.banch2[2], bias_path_list[3])

            set_weights(self.banch2[3], weight_path_list[4])
            if len(bias_path_list) > 0:
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
        self.conv1 = conv_bn(3, input_channel, 2, './0028.weights_0.json', '0028.biases_0.json',)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # building inverted residual blocks
        # for idxstage in range(len(self.stage_repeats)):
        #     numrepeat = self.stage_repeats[idxstage]
        #     output_channel = self.stage_out_channels[idxstage + 2]
        self.features = [
            InvertedResidual(24, 116, 2, 2,
                             ['./0033.weights_0.json',
                              './0073.weights_0.json',
                              './0058.weights_0.json',
                              './0061.weights_0.json',
                              './0044.weights_0.json'],
                             ['./0033.biases_0.json',
                              './0073.biases_0.json',
                              './0058.biases_0.json',
                              './0061.biases_0.json',
                              './0044.biases_0.json'],
                             ),
            InvertedResidual(116, 116, 1, 1,
                             ['./0044.weights_1.json',
                              './0095.weights_0.json',
                              './0044.weights_2.json'],
                             ['./0044.biases_1.json',
                              './0095.biases_0.json',
                              './0044.biases_2.json'],
                             ),
            InvertedResidual(116, 116, 1, 1,
                             ['./0044.weights_3.json',
                              './0095.weights_1.json',
                              './0044.weights_4.json'],
                             ['./0044.biases_3.json',
                              './0095.biases_1.json',
                              './0044.biases_4.json'],
                             ),
            InvertedResidual(116, 116, 1, 1,
                             ['./0044.weights_5.json',
                              './0095.weights_2.json',
                              './0044.weights_6.json'],
                             ['./0044.biases_5.json',
                              './0095.biases_2.json',
                              './0044.biases_6.json'],
                             ),

            InvertedResidual(116, 232, 2, 2,
                             ['./0092.weights_0.json',
                              './0056.weights_0.json',
                              './0089.weights_0.json',
                              './0092.weights_1.json',
                              './0056.weights_1.json'],
                             ['./0092.biases_0.json',
                              './0056.biases_0.json',
                              './0089.biases_0.json',
                              './0092.biases_1.json',
                              './0056.biases_1.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0056.weights_2.json',
                              './0087.weights_0.json',
                              './0056.weights_3.json'],
                             ['./0056.biases_2.json',
                              './0087.biases_0.json',
                              './0056.biases_3.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0056.weights_4.json',
                              './0087.weights_1.json',
                              './0056.weights_5.json'],
                             ['./0056.biases_4.json',
                              './0087.biases_1.json',
                              './0056.biases_5.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0056.weights_6.json',
                              './0087.weights_2.json',
                              './0056.weights_7.json'],
                             ['./0056.biases_6.json',
                              './0087.biases_2.json',
                              './0056.biases_7.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0056.weights_8.json',
                              './0087.weights_3.json',
                              './0056.weights_9.json'],
                             ['./0056.biases_8.json',
                              './0087.biases_3.json',
                              './0056.biases_9.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0056.weights_10.json',
                              './0087.weights_4.json',
                              './0056.weights_11.json'],
                             ['./0056.biases_10.json',
                              './0087.biases_4.json',
                              './0056.biases_11.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0056.weights_12.json',
                              './0087.weights_5.json',
                              './0056.weights_13.json'],
                             ['./0056.biases_12.json',
                              './0087.biases_5.json',
                              './0056.biases_13.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0056.weights_14.json',
                              './0087.weights_6.json',
                              './0056.weights_15.json'],
                             ['./0056.biases_14.json',
                              './0087.biases_6.json',
                              './0056.biases_15.json'],
                             ),

            InvertedResidual(232, 464, 2, 2,
                             ['./0078.weights_0.json',
                              './0097.weights_0.json',
                              './0042.weights_0.json',
                              './0078.weights_1.json',
                              './0097.weights_1.json'],
                             ['./0078.biases_0.json',
                              './0097.biases_0.json',
                              './0042.biases_0.json',
                              './0078.biases_1.json',
                              './0097.biases_1.json'],
                             ),
            InvertedResidual(464, 464, 1, 1,
                             ['./0097.weights_2.json',
                              './0036.weights_0.json',
                              './0097.weights_3.json'],
                             ['./0097.biases_2.json',
                              './0036.biases_0.json',
                              './0097.biases_3.json'],
                             )
            ,
            InvertedResidual(464, 464, 1, 1,
                             ['./0097.weights_4.json',
                              './0036.weights_1.json',
                              './0097.weights_5.json'],
                             ['./0097.biases_4.json',
                              './0036.biases_1.json',
                              './0097.biases_5.json'],
                             ),
            InvertedResidual(464, 464, 1, 1,
                             ['./0097.weights_6.json',
                              './0036.weights_2.json',
                              './0097.weights_7.json'],
                             ['./0097.biases_6.json',
                              './0036.biases_2.json',
                              './0097.biases_7.json'],
                             ),
        ]

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1], './0040.weights_0.json', '0040.biases_0.json',)
        self.globalpool = nn.Sequential(nn.AvgPool2d(7))

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))
        set_weights(self.classifier[0], './0052.dense_weights_0.json')
        set_biases(self.classifier[0], './0052.biases_0.json')

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
    input_cat = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.7/shufflenetv2_tvm_O3/cat.bin"
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
    exit(0)