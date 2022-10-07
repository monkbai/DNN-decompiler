import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
import math
import torch.functional as F

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


class MBConvBlock(nn.Module):
    def __init__(self, inp, final_oup, k, s, expand_ratio, weights_path_list: list, bias_path_list: list):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.id_skip = True  # skip connection and drop connect

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=True)
            # self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, padding=(k - 1) // 2, stride=s, bias=True)
        # self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Output phase
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=True)
        # self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._momentum, eps=self._epsilon)
        self._relu = nn.ReLU6(inplace=True)

        index = 0
        if self.expand_ratio != 1:
            set_weights(self._expand_conv, weights_path_list[index])
            set_biases(self._expand_conv, bias_path_list[index])
            index += 1
        set_weights(self._depthwise_conv, weights_path_list[index])
        set_biases(self._depthwise_conv, bias_path_list[index])
        index += 1
        set_weights(self._project_conv, weights_path_list[index])
        set_biases(self._project_conv, bias_path_list[index])
        index += 1
        assert index == len(weights_path_list) == len(bias_path_list)

    def forward(self, x):
        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            # x = self._relu(self._bn0(self._expand_conv(x)))
            x = self._relu(self._expand_conv(x))
        # x = self._relu(self._bn1(self._depthwise_conv(x)))
        x = self._relu(self._depthwise_conv(x))

        # x = self._bn2(self._project_conv(x))
        x = self._project_conv(x)

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.input_filters == self.output_filters:
            x += identity  # skip connection
        return x


class EfficientNetLite(nn.Module):
    def __init__(self, widthi_multiplier, depth_multiplier, num_classes):
        super(EfficientNetLite, self).__init__()

        # Batch norm parameters
        momentum = 0.01
        epsilon = 1e-3

        mb_block_settings = [
            # repeat|kernal_size|stride|expand|input|output|se_ratio
            [1, 3, 1, 1, 32, 16, 0.25],
            [2, 3, 2, 6, 16, 24, 0.25],
            [2, 5, 2, 6, 24, 40, 0.25],
            [3, 3, 2, 6, 40, 80, 0.25],
            [3, 5, 1, 6, 80, 112, 0.25],
            [4, 5, 2, 6, 112, 192, 0.25],
            [1, 3, 1, 6, 192, 320, 0.25]
        ]

        # Stem
        out_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )
        set_weights(self.stem[0], '0009.weights_0.json')
        set_biases(self.stem[0], '0009.biases_0.json')

        # Build blocks
        self.blocks = nn.ModuleList([])

        # 0
        self.blocks.append(nn.ModuleList([
            MBConvBlock(32, 24, 3, 1, 1, ['0010.weights_0.json', '0011.weights_0.json'], ['0010.biases_0.json', '0011.biases_0.json'])
        ]))

        # 1
        self.blocks.append(nn.ModuleList([
            MBConvBlock(24, 32, 3, 2, 6, ['0012.weights_0.json', '0013.weights_0.json', '0014.weights_0.json'], ['0012.biases_0.json', '0013.biases_0.json', '0014.biases_0.json']),
            MBConvBlock(32, 32, 3, 1, 6, ['0015.weights_0.json', '0016.weights_0.json', '0017.weights_0.json'], ['0015.biases_0.json', '0016.biases_0.json', '0017.biases_0.json']),
            MBConvBlock(32, 32, 3, 1, 6, ['0015.weights_1.json', '0016.weights_1.json', '0017.weights_1.json'], ['0015.biases_1.json', '0016.biases_1.json', '0017.biases_1.json']),
            MBConvBlock(32, 32, 3, 1, 6, ['0015.weights_2.json', '0016.weights_2.json', '0017.weights_2.json'], ['0015.biases_2.json', '0016.biases_2.json', '0017.biases_2.json'])
        ]))

        # 2
        self.blocks.append(nn.ModuleList([
            MBConvBlock(32, 56, 5, 2, 6, ['0015.weights_3.json', '0021.weights_0.json', '0022.weights_0.json'], ['0015.biases_3.json', '0021.biases_0.json', '0022.biases_0.json']),
            MBConvBlock(56, 56, 5, 1, 6, ['0023.weights_0.json', '0024.weights_0.json', '0025.weights_0.json'], ['0023.biases_0.json', '0024.biases_0.json', '0025.biases_0.json']),
            MBConvBlock(56, 56, 5, 1, 6, ['0023.weights_1.json', '0024.weights_1.json', '0025.weights_1.json'], ['0023.biases_1.json', '0024.biases_1.json', '0025.biases_1.json']),
            MBConvBlock(56, 56, 5, 1, 6, ['0023.weights_2.json', '0024.weights_2.json', '0025.weights_2.json'], ['0023.biases_2.json', '0024.biases_2.json', '0025.biases_2.json'])
        ]))

        # 3
        self.blocks.append(nn.ModuleList([
            MBConvBlock(56, 112, 3, 2, 6, ['0023.weights_3.json', '0029.weights_0.json', '0030.weights_0.json'], ['0023.biases_3.json', '0029.biases_0.json', '0030.biases_0.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0031.weights_0.json', '0032.weights_0.json', '0033.weights_0.json'], ['0031.biases_0.json', '0032.biases_0.json', '0033.biases_0.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0031.weights_1.json', '0032.weights_1.json', '0033.weights_1.json'], ['0031.biases_1.json', '0032.biases_1.json', '0033.biases_1.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0031.weights_2.json', '0032.weights_2.json', '0033.weights_2.json'], ['0031.biases_2.json', '0032.biases_2.json', '0033.biases_2.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0031.weights_3.json', '0032.weights_3.json', '0033.weights_3.json'], ['0031.biases_3.json', '0032.biases_3.json', '0033.biases_3.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0031.weights_4.json', '0032.weights_4.json', '0033.weights_4.json'], ['0031.biases_4.json', '0032.biases_4.json', '0033.biases_4.json']),
        ]))

        # 4
        self.blocks.append(nn.ModuleList([
            MBConvBlock(112, 160, 5, 1, 6, ['0031.weights_5.json', '0039.weights_0.json', '0040.weights_0.json'], ['0031.biases_5.json', '0039.biases_0.json', '0040.biases_0.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0041.weights_0.json', '0042.weights_0.json', '0043.weights_0.json'], ['0041.biases_0.json', '0042.biases_0.json', '0043.biases_0.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0041.weights_1.json', '0042.weights_1.json', '0043.weights_1.json'], ['0041.biases_1.json', '0042.biases_1.json', '0043.biases_1.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0041.weights_2.json', '0042.weights_2.json', '0043.weights_2.json'], ['0041.biases_2.json', '0042.biases_2.json', '0043.biases_2.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0041.weights_3.json', '0042.weights_3.json', '0043.weights_3.json'], ['0041.biases_3.json', '0042.biases_3.json', '0043.biases_3.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0041.weights_4.json', '0042.weights_4.json', '0043.weights_4.json'], ['0041.biases_4.json', '0042.biases_4.json', '0043.biases_4.json'])
        ]))

        # 5
        self.blocks.append(nn.ModuleList([
            MBConvBlock(160, 272, 5, 2, 6, ['0041.weights_5.json', '0049.weights_0.json', '0050.weights_0.json'], ['0041.biases_5.json', '0049.biases_0.json', '0050.biases_0.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0051.weights_0.json', '0052.weights_0.json', '0053.weights_0.json'], ['0051.biases_0.json', '0052.biases_0.json', '0053.biases_0.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0051.weights_1.json', '0052.weights_1.json', '0053.weights_1.json'], ['0051.biases_1.json', '0052.biases_1.json', '0053.biases_1.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0051.weights_2.json', '0052.weights_2.json', '0053.weights_2.json'], ['0051.biases_2.json', '0052.biases_2.json', '0053.biases_2.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0051.weights_3.json', '0052.weights_3.json', '0053.weights_3.json'], ['0051.biases_3.json', '0052.biases_3.json', '0053.biases_3.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0051.weights_4.json', '0052.weights_4.json', '0053.weights_4.json'], ['0051.biases_4.json', '0052.biases_4.json', '0053.biases_4.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0051.weights_5.json', '0052.weights_5.json', '0053.weights_5.json'], ['0051.biases_5.json', '0052.biases_5.json', '0053.biases_5.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0051.weights_6.json', '0052.weights_6.json', '0053.weights_6.json'], ['0051.biases_6.json', '0052.biases_6.json', '0053.biases_6.json'])
        ]))

        # 6
        self.blocks.append(nn.ModuleList([
            MBConvBlock(272, 448, 3, 1, 6, ['0051.weights_7.json', '0061.weights_0.json', '0062.weights_0.json'], ['0051.biases_7.json', '0061.biases_0.json', '0062.biases_0.json'])
        ]))

        # Head
        in_channels = 448
        out_channels = 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )
        set_weights(self.head[0], '0063.weights_0.json')
        set_biases(self.head[0], '0063.biases_0.json')

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(out_channels, num_classes)
        set_weights(self.fc, '0065.params_0.json')
        set_biases(self.fc, '0065.biases_0.json')

        self.softmax = nn.Softmax()
        # self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        idx = 0
        for stage in self.blocks:
            for block in stage:
                x = block(x)
                idx += 1
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return self.softmax(x)

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
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 1.0 / float(n))
    #             m.bias.data.zero_()
    #
    # def load_pretrain(self, path):
    #     state_dict = torch.load(path)
    #     self.load_state_dict(state_dict, strict=True)


if __name__ == '__main__':
    input_cat = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2021/efficientnet/cat.bin"
    new_cat = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2021/efficientnet/cat_transpose.bin"
    if len(sys.argv) == 2:
        input_cat = sys.argv[1]
        cat_dir = os.path.dirname(input_cat)
        new_cat = os.path.join(cat_dir, "cat_transpose.bin")
        # print(new_cat)
        
    model_name = 'efficientnet_lite4'
    width_coefficient, depth_coefficient, image_size, dropout_rate = 1.4, 1.8, 300, 0.3
    num_classes = 1000
    model = EfficientNetLite(width_coefficient, depth_coefficient, num_classes)
    model.eval()
    # print(model)
    # exit(0)

    # input = torch.randn(1, 3, 224, 224)
    with open(input_cat, 'br') as f:
            bin_data = f.read()
            np_arr = np.frombuffer(bin_data, dtype=np.float32)
            # print(np_arr.shape)
            np_arr = np_arr.reshape(3, 224, 224)
            np_arr = np_arr.reshape((1, 3, 224, 224))

            new_np_arr = np.transpose(np_arr, (0, 2, 3, 1))
            # print(new_np_arr.shape)
            with open(new_cat, "wb") as fp:
                fp.write(new_np_arr.astype(np.float32).tobytes())

            x = torch.Tensor(np_arr)
            # print(x.shape)
    input = x
    out = model(input)

    max_index = np.argmax(out.detach().numpy())
    print("Result:", max_index)
    # print(out)
    print("Confidence:", out.detach().numpy()[0, max_index])
    #print(total_size)
    exit(0)
