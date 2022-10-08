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
            var_path: str, gamma_path: str, mean_path: str, beta_path: str):
    blk = nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
    set_weights(blk[0], weights_path)
    if len(bias_path) > 0:
        set_biases(blk[0], bias_path)

    set_var(blk[1], var_path)
    set_bn_weights(blk[1], gamma_path)
    set_mean(blk[1], mean_path)
    set_biases(blk[1], beta_path)
    return blk


def conv_1x1_bn(inp, oup, weights_path: str, bias_path: str,
                var_path: str, gamma_path: str, mean_path: str, beta_path: str):
    blk = nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
    set_weights(blk[0], weights_path)
    if len(bias_path) > 0:
        set_biases(blk[0], bias_path)

    set_var(blk[1], var_path)
    set_bn_weights(blk[1], gamma_path)
    set_mean(blk[1], mean_path)
    set_biases(blk[1], beta_path)
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
                 var_path_list: list, gamma_path_list: list, mean_path_list: list, beta_path_list: list):
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
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=True),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            set_weights(self.banch2[0], weight_path_list[0])
            if len(bias_path_list) > 0:
                set_biases(self.banch2[0], bias_path_list[0])

            set_var(self.banch2[1], var_path_list[0])
            set_bn_weights(self.banch2[1], gamma_path_list[0])
            set_mean(self.banch2[1], mean_path_list[0])
            set_biases(self.banch2[1], beta_path_list[0])

            set_weights(self.banch2[3], weight_path_list[1])
            if len(bias_path_list) > 0:
                set_biases(self.banch2[3], bias_path_list[1])

            set_var(self.banch2[4], var_path_list[1])
            set_bn_weights(self.banch2[4], gamma_path_list[1])
            set_mean(self.banch2[4], mean_path_list[1])
            set_biases(self.banch2[4], beta_path_list[1])

            set_weights(self.banch2[5], weight_path_list[2])
            if len(bias_path_list) > 0:
                set_biases(self.banch2[5], bias_path_list[2])

            set_var(self.banch2[6], var_path_list[2])
            set_bn_weights(self.banch2[6], gamma_path_list[2])
            set_mean(self.banch2[6], mean_path_list[2])
            set_biases(self.banch2[6], beta_path_list[2])
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            set_weights(self.banch1[0], weight_path_list[0])
            if len(bias_path_list) > 0:
                set_biases(self.banch1[0], bias_path_list[0])

            set_var(self.banch1[1], var_path_list[0])
            set_bn_weights(self.banch1[1], gamma_path_list[0])
            set_mean(self.banch1[1], mean_path_list[0])
            set_biases(self.banch1[1], beta_path_list[0])

            set_weights(self.banch1[2], weight_path_list[1])
            if len(bias_path_list) > 0:
                set_biases(self.banch1[2], bias_path_list[1])

            set_var(self.banch1[3], var_path_list[1])
            set_bn_weights(self.banch1[3], gamma_path_list[1])
            set_mean(self.banch1[3], mean_path_list[1])
            set_biases(self.banch1[3], beta_path_list[1])

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=True),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            set_weights(self.banch2[0], weight_path_list[2])
            if len(bias_path_list) > 0:
                set_biases(self.banch2[0], bias_path_list[2])

            set_var(self.banch2[1], var_path_list[2])
            set_bn_weights(self.banch2[1], gamma_path_list[2])
            set_mean(self.banch2[1], mean_path_list[2])
            set_biases(self.banch2[1], beta_path_list[2])

            set_weights(self.banch2[3], weight_path_list[3])
            if len(bias_path_list) > 0:
                set_biases(self.banch2[3], bias_path_list[3])

            set_var(self.banch2[4], var_path_list[3])
            set_bn_weights(self.banch2[4], gamma_path_list[3])
            set_mean(self.banch2[4], mean_path_list[3])
            set_biases(self.banch2[4], beta_path_list[3])

            set_weights(self.banch2[5], weight_path_list[4])
            if len(bias_path_list) > 0:
                set_biases(self.banch2[5], bias_path_list[4])

            set_var(self.banch2[6], var_path_list[4])
            set_bn_weights(self.banch2[6], gamma_path_list[4])
            set_mean(self.banch2[6], mean_path_list[4])
            set_biases(self.banch2[6], beta_path_list[4])

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
        self.conv1 = conv_bn(3, input_channel, 2, './0080.weights_0.json', '',
                             '0053.var_0.json', '0204.gamma_0.json', '0131.mean_0.json', '0102.beta_0.json')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # building inverted residual blocks
        # for idxstage in range(len(self.stage_repeats)):
        #     numrepeat = self.stage_repeats[idxstage]
        #     output_channel = self.stage_out_channels[idxstage + 2]
        self.features = [
            InvertedResidual(24, 116, 2, 2,
                             ['./0064.weights_0.json',
                              './0098.weights_0.json',
                              './0145.weights_0.json',
                              './0028.weights_0.json',
                              './0084.weights_0.json'],
                             [],
                             ['0053.var_1.json',
                              '0139.var_0.json',
                              '0139.var_1.json',
                              '0139.var_2.json',
                              '0139.var_3.json'],
                             ['0204.gamma_2.json',
                              '0141.gamma_0.json',
                              '0141.gamma_2.json',
                              '0141.gamma_4.json',
                              '0141.gamma_6.json'],
                             ['0131.mean_1.json',
                              '0123.mean_0.json',
                              '0123.mean_1.json',
                              '0123.mean_2.json',
                              '0123.mean_3.json'],
                             ['0102.beta_1.json',
                              '0228.beta_0.json',
                              '0228.beta_1.json',
                              '0228.beta_2.json',
                              '0228.beta_3.json'],
                             ),
            InvertedResidual(116, 116, 1, 1,
                             ['./0084.weights_1.json',
                              './0224.weights_0.json',
                              './0084.weights_2.json'],
                             [],
                             ['0139.var_4.json',
                              '0139.var_5.json',
                              '0139.var_6.json'],
                             ['0141.gamma_8.json',
                              '0141.gamma_10.json',
                              '0141.gamma_12.json'],
                             ['0123.mean_4.json',
                              '0123.mean_5.json',
                              '0123.mean_6.json'],
                             ['0228.beta_4.json',
                              '0228.beta_5.json',
                              '0228.beta_6.json'],
                             ),
            InvertedResidual(116, 116, 1, 1,
                             ['./0084.weights_3.json',
                              './0224.weights_1.json',
                              './0084.weights_4.json'],
                             [],
                             ['0139.var_7.json',
                              '0139.var_8.json',
                              '0139.var_9.json'],
                             ['0141.gamma_14.json',
                              '0141.gamma_16.json',
                              '0141.gamma_18.json'],
                             ['0123.mean_7.json',
                              '0123.mean_8.json',
                              '0123.mean_9.json'],
                             ['0228.beta_7.json',
                              '0228.beta_8.json',
                              '0228.beta_9.json'],
                             ),
            InvertedResidual(116, 116, 1, 1,
                             ['./0084.weights_5.json',
                              './0224.weights_2.json',
                              './0084.weights_6.json'],
                             [],
                             ['0139.var_10.json',
                              '0139.var_11.json',
                              '0139.var_12.json'],
                             ['0141.gamma_20.json',
                              '0141.gamma_22.json',
                              '0141.gamma_24.json'],
                             ['0123.mean_10.json',
                              '0123.mean_11.json',
                              '0123.mean_12.json'],
                             ['0228.beta_10.json',
                              '0228.beta_11.json',
                              '0228.beta_12.json'],
                             ),

            InvertedResidual(116, 232, 2, 2,
                             ['./0109.weights_0.json',
                              './0074.weights_0.json',
                              './0068.weights_0.json',
                              './0109.weights_1.json',
                              './0074.weights_1.json'],
                             [],
                             ['0129.var_0.json',
                              '0129.var_1.json',
                              '0129.var_2.json',
                              '0129.var_3.json',
                              '0129.var_4.json'],
                             ['0180.gamma_0.json',
                              '0180.gamma_2.json',
                              '0180.gamma_4.json',
                              '0180.gamma_6.json',
                              '0180.gamma_8.json'],
                             ['0192.mean_0.json',
                              '0192.mean_1.json',
                              '0192.mean_2.json',
                              '0192.mean_3.json',
                              '0192.mean_4.json'],
                             ['0055.beta_0.json',
                              '0055.beta_1.json',
                              '0055.beta_2.json',
                              '0055.beta_3.json',
                              '0055.beta_4.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0074.weights_2.json',
                              './0041.weights_0.json',
                              './0074.weights_3.json'],
                             [],
                             ['0129.var_5.json',
                              '0129.var_6.json',
                              '0129.var_7.json'],
                             ['0180.gamma_10.json',
                              '0180.gamma_12.json',
                              '0180.gamma_14.json'],
                             ['0192.mean_5.json',
                              '0192.mean_6.json',
                              '0192.mean_7.json'],
                             ['0055.beta_5.json',
                              '0055.beta_6.json',
                              '0055.beta_7.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0074.weights_4.json',
                              './0041.weights_1.json',
                              './0074.weights_5.json'],
                             [],
                             ['0129.var_8.json',
                              '0129.var_9.json',
                              '0129.var_10.json'],
                             ['0180.gamma_16.json',
                              '0180.gamma_18.json',
                              '0180.gamma_20.json'],
                             ['0192.mean_8.json',
                              '0192.mean_9.json',
                              '0192.mean_10.json'],
                             ['0055.beta_8.json',
                              '0055.beta_9.json',
                              '0055.beta_10.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0074.weights_6.json',
                              './0041.weights_2.json',
                              './0074.weights_7.json'],
                             [],
                             ['0129.var_11.json',
                              '0129.var_12.json',
                              '0129.var_13.json'],
                             ['0180.gamma_22.json',
                              '0180.gamma_24.json',
                              '0180.gamma_26.json'],
                             ['0192.mean_11.json',
                              '0192.mean_12.json',
                              '0192.mean_13.json'],
                             ['0055.beta_11.json',
                              '0055.beta_12.json',
                              '0055.beta_13.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0074.weights_8.json',
                              './0041.weights_3.json',
                              './0074.weights_9.json'],
                             [],
                             ['0129.var_14.json',
                              '0129.var_15.json',
                              '0129.var_16.json'],
                             ['0180.gamma_28.json',
                              '0180.gamma_30.json',
                              '0180.gamma_32.json'],
                             ['0192.mean_14.json',
                              '0192.mean_15.json',
                              '0192.mean_16.json'],
                             ['0055.beta_14.json',
                              '0055.beta_15.json',
                              '0055.beta_16.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0074.weights_10.json',
                              './0041.weights_4.json',
                              './0074.weights_11.json'],
                             [],
                             ['0129.var_17.json',
                              '0129.var_18.json',
                              '0129.var_19.json'],
                             ['0180.gamma_34.json',
                              '0180.gamma_36.json',
                              '0180.gamma_38.json'],
                             ['0192.mean_17.json',
                              '0192.mean_18.json',
                              '0192.mean_19.json'],
                             ['0055.beta_17.json',
                              '0055.beta_18.json',
                              '0055.beta_19.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0074.weights_12.json',
                              './0041.weights_5.json',
                              './0074.weights_13.json'],
                             [],
                             ['0129.var_20.json',
                              '0129.var_21.json',
                              '0129.var_22.json'],
                             ['0180.gamma_40.json',
                              '0180.gamma_42.json',
                              '0180.gamma_44.json'],
                             ['0192.mean_20.json',
                              '0192.mean_21.json',
                              '0192.mean_22.json'],
                             ['0055.beta_20.json',
                              '0055.beta_21.json',
                              '0055.beta_22.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0074.weights_14.json',
                              './0041.weights_6.json',
                              './0074.weights_15.json'],
                             [],
                             ['0129.var_23.json',
                              '0129.var_24.json',
                              '0129.var_25.json'],
                             ['0180.gamma_46.json',
                              '0180.gamma_48.json',
                              '0180.gamma_50.json'],
                             ['0192.mean_23.json',
                              '0192.mean_24.json',
                              '0192.mean_25.json'],
                             ['0055.beta_23.json',
                              '0055.beta_24.json',
                              '0055.beta_25.json'],
                             ),

            InvertedResidual(232, 464, 2, 2,
                             ['./0160.weights_0.json',
                              './0168.weights_0.json',
                              './0059.weights_0.json',
                              './0160.weights_1.json',
                              './0168.weights_1.json'],
                             [],
                             ['0155.var_0.json',
                              '0155.var_1.json',
                              '0155.var_2.json',
                              '0155.var_3.json',
                              '0155.var_4.json'],
                             ['0135.gamma_0.json',
                              '0135.gamma_2.json',
                              '0135.gamma_4.json',
                              '0135.gamma_6.json',
                              '0135.gamma_8.json'],
                             ['0137.mean_0.json',
                              '0137.mean_1.json',
                              '0137.mean_2.json',
                              '0137.mean_3.json',
                              '0137.mean_4.json'],
                             ['0202.beta_0.json',
                              '0202.beta_1.json',
                              '0202.beta_2.json',
                              '0202.beta_3.json',
                              '0202.beta_4.json'],
                             ),
            InvertedResidual(464, 464, 1, 1,
                             ['./0168.weights_2.json',
                              './0219.weights_0.json',
                              './0168.weights_3.json'],
                             [],
                             ['0155.var_5.json',
                              '0155.var_6.json',
                              '0155.var_7.json'],
                             ['0135.gamma_10.json',
                              '0135.gamma_12.json',
                              '0135.gamma_14.json'],
                             ['0137.mean_5.json',
                              '0137.mean_6.json',
                              '0137.mean_7.json'],
                             ['0202.beta_5.json',
                              '0202.beta_6.json',
                              '0202.beta_7.json'],
                             )
            ,
            InvertedResidual(464, 464, 1, 1,
                             ['./0168.weights_4.json',
                              './0219.weights_1.json',
                              './0168.weights_5.json'],
                             [],
                             ['0155.var_8.json',
                              '0155.var_9.json',
                              '0155.var_10.json'],
                             ['0135.gamma_16.json',
                              '0135.gamma_18.json',
                              '0135.gamma_20.json'],
                             ['0137.mean_8.json',
                              '0137.mean_9.json',
                              '0137.mean_10.json'],
                             ['0202.beta_8.json',
                              '0202.beta_9.json',
                              '0202.beta_10.json'],),
            InvertedResidual(464, 464, 1, 1,
                             ['./0168.weights_6.json',
                              './0219.weights_2.json',
                              './0168.weights_7.json'],
                             [],
                             ['0155.var_11.json',
                              '0155.var_12.json',
                              '0155.var_13.json'],
                             ['0135.gamma_22.json',
                              '0135.gamma_24.json',
                              '0135.gamma_26.json'],
                             ['0137.mean_11.json',
                              '0137.mean_12.json',
                              '0137.mean_13.json'],
                             ['0202.beta_11.json',
                              '0202.beta_12.json',
                              '0202.beta_13.json'],
                             ),
        ]

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1], './0198.weights_0.json', '',
                                     '0153.var_0.json', '0157.gamma_0.json', '0038.mean_0.json', '0151.beta_0.json')
        self.globalpool = nn.Sequential(nn.AvgPool2d(7))

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))
        set_weights(self.classifier[0], './0174.dense_weights_0.json')
        set_biases(self.classifier[0], './0030.biases_0.json')

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
    input_cat = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.7/shufflenetv2_tvm_O0/cat.bin"
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