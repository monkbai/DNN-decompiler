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


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        #input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        #print('input_channel:', input_channel)
        layers = [conv_3x3_bn(3, 32, 2)]
        set_weights(layers[0][0], '0133.weights_0.json')
        set_biases(layers[0][0], '0074.biases_0.json')

        # building inverted residual blocks
        # for t, c, n, s in self.cfgs:
        #     output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
        #     for i in range(n):
        #         layers.append(InvertedResidual(input_channel, output_channel, s if i == 0 else 1, t))
        #         print('InvertedResidual:', input_channel, output_channel, s if i == 0 else 1, t)
        #         input_channel = output_channel

        layers.append(InvertedResidual(32, 16, 1, 1, ['0137.weights_0.json', '0186.weights_0.json'],
                                                     ['0074.biases_1.json', '0077.biases_0.json']))

        layers.append(InvertedResidual(16, 24, 2, 6, ['0238.weights_0.json', '0242.weights_0.json', '0247.weights_0.json'],
                                                     ['0107.biases_0.json', '0110.biases_0.json', '0113.biases_0.json']))
        layers.append(InvertedResidual(24, 24, 1, 6, ['0252.weights_0.json', '0256.weights_0.json', '0261.weights_0.json'],
                                                     ['0116.biases_0.json', '0116.biases_1.json', '0113.biases_1.json']))

        layers.append(InvertedResidual(24, 32, 2, 6, ['0252.weights_1.json', '0265.weights_0.json', '0141.weights_0.json'],
                                                     ['0116.biases_2.json', '0119.biases_0.json', '0122.biases_0.json']))
        layers.append(InvertedResidual(32, 32, 1, 6, ['0145.weights_0.json', '0149.weights_0.json', '0153.weights_0.json'],
                                                     ['0125.biases_0.json', '0125.biases_1.json', '0122.biases_1.json']))
        layers.append(InvertedResidual(32, 32, 1, 6, ['0145.weights_1.json', '0149.weights_1.json', '0153.weights_1.json'],
                                                     ['0125.biases_2.json', '0125.biases_3.json', '0122.biases_2.json']))

        layers.append(InvertedResidual(32, 64, 2, 6, ['0145.weights_2.json', '0157.weights_0.json', '0162.weights_0.json'],
                                                     ['0125.biases_4.json', '0128.biases_0.json', '0080.biases_0.json']))
        layers.append(InvertedResidual(64, 64, 1, 6, ['0167.weights_0.json', '0171.weights_0.json', '0176.weights_0.json'],
                                                     ['0083.biases_0.json', '0083.biases_1.json', '0080.biases_1.json']))
        layers.append(InvertedResidual(64, 64, 1, 6, ['0167.weights_1.json', '0171.weights_1.json', '0176.weights_1.json'],
                                                     ['0083.biases_2.json', '0083.biases_3.json', '0080.biases_2.json']))
        layers.append(InvertedResidual(64, 64, 1, 6, ['0167.weights_2.json', '0171.weights_2.json', '0176.weights_2.json'],
                                                     ['0083.biases_4.json', '0083.biases_5.json', '0080.biases_3.json']))

        layers.append(InvertedResidual(64, 96, 1, 6, ['0167.weights_3.json', '0171.weights_3.json', '0181.weights_0.json'],
                                                     ['0083.biases_6.json', '0083.biases_7.json', '0086.biases_0.json']))
        layers.append(InvertedResidual(96, 96, 1, 6, ['0191.weights_0.json', '0195.weights_0.json', '0200.weights_0.json'],
                                                     ['0089.biases_0.json', '0089.biases_1.json', '0086.biases_1.json']))
        layers.append(InvertedResidual(96, 96, 1, 6, ['0191.weights_1.json', '0195.weights_1.json', '0200.weights_1.json'],
                                                     ['0089.biases_2.json', '0089.biases_3.json', '0086.biases_2.json']))

        layers.append(InvertedResidual(96, 160, 2, 6, ['0191.weights_2.json', '0204.weights_0.json', '0209.weights_0.json'],
                                                      ['0089.biases_4.json', '0092.biases_0.json', '0095.biases_0.json']))
        layers.append(InvertedResidual(160, 160, 1, 6, ['0214.weights_0.json', '0218.weights_0.json', '0223.weights_0.json'],
                                                       ['0098.biases_0.json', '0098.biases_1.json', '0095.biases_1.json']))
        layers.append(InvertedResidual(160, 160, 1, 6, ['0214.weights_1.json', '0218.weights_1.json', '0223.weights_1.json'],
                                                       ['0098.biases_2.json', '0098.biases_3.json', '0095.biases_2.json']))

        layers.append(InvertedResidual(160, 320, 1, 6, ['0214.weights_2.json', '0218.weights_2.json', '0228.weights_0.json'],
                                                       ['0098.biases_4.json', '0098.biases_5.json', '0101.biases_0.json']))

        self.features = nn.Sequential(*layers)
        # building last several layers
        #output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(320, 1280)
        set_weights(self.conv[0], '0233.weights_0.json')
        set_biases(self.conv[0], '0104.biases_0.json')

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)
        set_weights(self.classifier, '0268.dense_weights_0.json')
        set_biases(self.classifier, '0035.biases_0.json')

        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    model = MobileNetV2()
    model.eval()
    # print(model)
    # exit(0)

    # input = torch.randn(1, 3, 224, 224)
    with open("/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.9.dev/mobilenetv2_tvm_O0/cat.bin", 'br') as f:
            bin_data = f.read()
            np_arr = np.frombuffer(bin_data, dtype=np.float32)
            print(np_arr.shape)
            np_arr = np_arr.reshape(3, 224, 224)
            np_arr = np_arr.reshape((1, 3, 224, 224))

            # new_np_arr = np.transpose(np_arr, (0, 2, 3, 1))
            # print(new_np_arr.shape)
            # with open("/home/lifter/Documents/DL_compiler/BTD_DATA/Glow-2022/efficientnet/cat_transpose.bin", "wb") as fp:
            #     fp.write(new_np_arr.astype(np.float32).tobytes())

            x = torch.Tensor(np_arr)
            print(x.shape)
    input = x
    out = model(input)

    max_index = np.argmax(out.detach().numpy())
    print(max_index)
    # print(out)
    print(out.detach().numpy()[0, max_index])
    exit(0)
