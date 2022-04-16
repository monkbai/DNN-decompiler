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
        self.conv1 = conv_bn(3, input_channel, 2, './0181.weights_0.json', '',
                             '0021.var_0.json', '0122.gamma_0.json', '0164.mean_0.json', '0024.beta_0.json')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # building inverted residual blocks
        # for idxstage in range(len(self.stage_repeats)):
        #     numrepeat = self.stage_repeats[idxstage]
        #     output_channel = self.stage_out_channels[idxstage + 2]
        self.features = [
            InvertedResidual(24, 116, 2, 2,
                             ['./0185.weights_0.json',
                              './0217.weights_0.json',
                              './0222.weights_0.json',
                              './0226.weights_0.json',
                              './0231.weights_0.json'],
                             [],
                             ['0021.var_1.json',
                              '0062.var_0.json',
                              '0062.var_1.json',
                              '0062.var_2.json',
                              '0062.var_3.json'],
                             ['0122.gamma_3.json',
                              '0143.gamma_0.json',
                              '0143.gamma_3.json',
                              '0143.gamma_6.json',
                              '0143.gamma_9.json'],
                             ['0164.mean_1.json',
                              '0167.mean_0.json',
                              '0167.mean_1.json',
                              '0167.mean_2.json',
                              '0167.mean_3.json'],
                             ['0024.beta_2.json',
                              '0065.beta_0.json',
                              '0065.beta_2.json',
                              '0065.beta_4.json',
                              '0065.beta_6.json'],
                             ),
            InvertedResidual(116, 116, 1, 1,
                             ['./0231.weights_1.json',
                              './0235.weights_0.json',
                              './0231.weights_2.json'],
                             [],
                             ['0062.var_4.json',
                              '0062.var_5.json',
                              '0062.var_6.json'],
                             ['0143.gamma_12.json',
                              '0143.gamma_15.json',
                              '0143.gamma_18.json'],
                             ['0167.mean_4.json',
                              '0167.mean_5.json',
                              '0167.mean_6.json'],
                             ['0065.beta_8.json',
                              '0065.beta_10.json',
                              '0065.beta_12.json'],
                             ),
            InvertedResidual(116, 116, 1, 1,
                             ['./0231.weights_3.json',
                              './0235.weights_1.json',
                              './0231.weights_4.json'],
                             [],
                             ['0062.var_7.json',
                              '0062.var_8.json',
                              '0062.var_9.json'],
                             ['0143.gamma_21.json',
                              '0143.gamma_24.json',
                              '0143.gamma_27.json'],
                             ['0167.mean_7.json',
                              '0167.mean_8.json',
                              '0167.mean_9.json'],
                             ['0065.beta_14.json',
                              '0065.beta_16.json',
                              '0065.beta_18.json'],
                             ),
            InvertedResidual(116, 116, 1, 1,
                             ['./0231.weights_5.json',
                              './0235.weights_2.json',
                              './0231.weights_6.json'],
                             [],
                             ['0062.var_10.json',
                              '0062.var_11.json',
                              '0062.var_12.json'],
                             ['0143.gamma_30.json',
                              '0143.gamma_33.json',
                              '0143.gamma_36.json'],
                             ['0167.mean_10.json',
                              '0167.mean_11.json',
                              '0167.mean_12.json'],
                             ['0065.beta_20.json',
                              '0065.beta_22.json',
                              '0065.beta_24.json'],
                             ),

            InvertedResidual(116, 232, 2, 2,
                             ['./0239.weights_0.json',
                              './0244.weights_0.json',
                              './0248.weights_0.json',
                              './0239.weights_1.json',
                              './0244.weights_1.json'],
                             [],
                             ['0074.var_0.json',
                              '0074.var_1.json',
                              '0074.var_2.json',
                              '0074.var_3.json',
                              '0074.var_4.json'],
                             ['0152.gamma_0.json',
                              '0152.gamma_3.json',
                              '0152.gamma_6.json',
                              '0152.gamma_9.json',
                              '0152.gamma_12.json'],
                             ['0170.mean_0.json',
                              '0170.mean_1.json',
                              '0170.mean_2.json',
                              '0170.mean_3.json',
                              '0170.mean_4.json'],
                             ['0077.beta_0.json',
                              '0077.beta_2.json',
                              '0077.beta_4.json',
                              '0077.beta_6.json',
                              '0077.beta_8.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0244.weights_2.json',
                              './0189.weights_0.json',
                              './0244.weights_3.json'],
                             [],
                             ['0074.var_5.json',
                              '0074.var_6.json',
                              '0074.var_7.json'],
                             ['0152.gamma_15.json',
                              '0152.gamma_18.json',
                              '0152.gamma_21.json'],
                             ['0170.mean_5.json',
                              '0170.mean_6.json',
                              '0170.mean_7.json'],
                             ['0077.beta_10.json',
                              '0077.beta_12.json',
                              '0077.beta_14.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0244.weights_4.json',
                              './0189.weights_1.json',
                              './0244.weights_5.json'],
                             [],
                             ['0074.var_8.json',
                              '0074.var_9.json',
                              '0074.var_10.json'],
                             ['0152.gamma_24.json',
                              '0152.gamma_27.json',
                              '0152.gamma_30.json'],
                             ['0170.mean_8.json',
                              '0170.mean_9.json',
                              '0170.mean_10.json'],
                             ['0077.beta_16.json',
                              '0077.beta_18.json',
                              '0077.beta_20.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0244.weights_6.json',
                              './0189.weights_2.json',
                              './0244.weights_7.json'],
                             [],
                             ['0074.var_11.json',
                              '0074.var_12.json',
                              '0074.var_13.json'],
                             ['0152.gamma_33.json',
                              '0152.gamma_36.json',
                              '0152.gamma_39.json'],
                             ['0170.mean_11.json',
                              '0170.mean_12.json',
                              '0170.mean_13.json'],
                             ['0077.beta_22.json',
                              '0077.beta_24.json',
                              '0077.beta_26.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0244.weights_8.json',
                              './0189.weights_3.json',
                              './0244.weights_9.json'],
                             [],
                             ['0074.var_14.json',
                              '0074.var_15.json',
                              '0074.var_16.json'],
                             ['0152.gamma_42.json',
                              '0152.gamma_45.json',
                              '0152.gamma_48.json'],
                             ['0170.mean_14.json',
                              '0170.mean_15.json',
                              '0170.mean_16.json'],
                             ['0077.beta_28.json',
                              '0077.beta_30.json',
                              '0077.beta_32.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0244.weights_10.json',
                              './0189.weights_4.json',
                              './0244.weights_11.json'],
                             [],
                             ['0074.var_17.json',
                              '0074.var_18.json',
                              '0074.var_19.json'],
                             ['0152.gamma_51.json',
                              '0152.gamma_54.json',
                              '0152.gamma_57.json'],
                             ['0170.mean_17.json',
                              '0170.mean_18.json',
                              '0170.mean_19.json'],
                             ['0077.beta_34.json',
                              '0077.beta_36.json',
                              '0077.beta_38.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0244.weights_12.json',
                              './0189.weights_5.json',
                              './0244.weights_13.json'],
                             [],
                             ['0074.var_20.json',
                              '0074.var_21.json',
                              '0074.var_22.json'],
                             ['0152.gamma_60.json',
                              '0152.gamma_63.json',
                              '0152.gamma_66.json'],
                             ['0170.mean_20.json',
                              '0170.mean_21.json',
                              '0170.mean_22.json'],
                             ['0077.beta_40.json',
                              '0077.beta_42.json',
                              '0077.beta_44.json'],
                             ),
            InvertedResidual(232, 232, 1, 1,
                             ['./0244.weights_14.json',
                              './0189.weights_6.json',
                              './0244.weights_15.json'],
                             [],
                             ['0074.var_23.json',
                              '0074.var_24.json',
                              '0074.var_25.json'],
                             ['0152.gamma_69.json',
                              '0152.gamma_72.json',
                              '0152.gamma_75.json'],
                             ['0170.mean_23.json',
                              '0170.mean_24.json',
                              '0170.mean_25.json'],
                             ['0077.beta_46.json',
                              '0077.beta_48.json',
                              '0077.beta_50.json'],
                             ),

            InvertedResidual(232, 464, 2, 2,
                             ['./0193.weights_0.json',
                              './0198.weights_0.json',
                              './0203.weights_0.json',
                              './0193.weights_1.json',
                              './0198.weights_1.json'],
                             [],
                             ['0033.var_0.json',
                              '0033.var_1.json',
                              '0033.var_2.json',
                              '0033.var_3.json',
                              '0033.var_4.json'],
                             ['0161.gamma_0.json',
                              '0161.gamma_3.json',
                              '0161.gamma_6.json',
                              '0161.gamma_9.json',
                              '0161.gamma_12.json'],
                             ['0173.mean_0.json',
                              '0173.mean_1.json',
                              '0173.mean_2.json',
                              '0173.mean_3.json',
                              '0173.mean_4.json'],
                             ['0036.beta_0.json',
                              '0036.beta_2.json',
                              '0036.beta_4.json',
                              '0036.beta_6.json',
                              '0036.beta_8.json'],
                             ),
            InvertedResidual(464, 464, 1, 1,
                             ['./0198.weights_2.json',
                              './0207.weights_0.json',
                              './0198.weights_3.json'],
                             [],
                             ['0033.var_5.json',
                              '0033.var_6.json',
                              '0033.var_7.json'],
                             ['0161.gamma_15.json',
                              '0161.gamma_18.json',
                              '0161.gamma_21.json'],
                             ['0173.mean_5.json',
                              '0173.mean_6.json',
                              '0173.mean_7.json'],
                             ['0036.beta_10.json',
                              '0036.beta_12.json',
                              '0036.beta_14.json'],
                             )
            ,
            InvertedResidual(464, 464, 1, 1,
                             ['./0198.weights_4.json',
                              './0207.weights_1.json',
                              './0198.weights_5.json'],
                             [],
                             ['0033.var_8.json',
                              '0033.var_9.json',
                              '0033.var_10.json'],
                             ['0161.gamma_24.json',
                              '0161.gamma_27.json',
                              '0161.gamma_30.json'],
                             ['0173.mean_8.json',
                              '0173.mean_9.json',
                              '0173.mean_10.json'],
                             ['0036.beta_16.json',
                              '0036.beta_18.json',
                              '0036.beta_20.json'],),
            InvertedResidual(464, 464, 1, 1,
                             ['./0198.weights_6.json',
                              './0207.weights_2.json',
                              './0198.weights_7.json'],
                             [],
                             ['0033.var_11.json',
                              '0033.var_12.json',
                              '0033.var_13.json'],
                             ['0161.gamma_33.json',
                              '0161.gamma_36.json',
                              '0161.gamma_39.json'],
                             ['0173.mean_11.json',
                              '0173.mean_12.json',
                              '0173.mean_13.json'],
                             ['0036.beta_22.json',
                              '0036.beta_24.json',
                              '0036.beta_26.json'],
                             ),
        ]
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
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1], './0212.weights_0.json', '',
                                     '0045.var_0.json', '0134.gamma_0.json', '0176.mean_0.json', '0048.beta_0.json')
        self.globalpool = nn.Sequential(nn.AvgPool2d(7))

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))
        set_weights(self.classifier[0], './0251.dense_weights_0.json')
        set_biases(self.classifier[0], './0053.biases_0.json')

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
    model = ShuffleNetV2()
    # print(model)
    # input = torch.randn(1, 3, 224, 224)
    with open("/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.9.dev/shufflenetv2_tvm_O0/cat.bin", 'br') as f:
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