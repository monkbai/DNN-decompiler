import os
import sys
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
    if len(json_path) == 0:
        module.bias.data.zero_()
    else:
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
    def __init__(self, inp, final_oup, k, s, expand_ratio, weights_path_list: list, bias_path_list: list, with_bn=False, bn_init_list=[]):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.id_skip = True  # skip connection and drop connect
        self.with_bn = with_bn

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=True)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, padding=(k - 1) // 2, stride=s, bias=True)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Output phase
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=True)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._momentum, eps=self._epsilon)
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

        if with_bn:
            set_var(self._bn0, bn_init_list[0])
            set_bn_weights(self._bn0, bn_init_list[1])
            set_mean(self._bn0, bn_init_list[2])
            set_biases(self._bn0, bn_init_list[3])

    def forward(self, x):
        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            if self.with_bn:
                x = self._relu(self._bn0(self._expand_conv(x)))
            else:
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

        # Stem
        out_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=True),
            # nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )
        set_weights(self.stem[0], '0352.weights_0.json')
        set_biases(self.stem[0], '0270.biases_0.json')

        # Build blocks
        self.blocks = nn.ModuleList([])

        # 0
        self.blocks.append(nn.ModuleList([
            MBConvBlock(32, 24, 3, 1, 1, ['0167.weights_0.json', '0160.weights_0.json'], ['0270.biases_1.json', '0146.biases_0.json'])
        ]))

        # 1
        self.blocks.append(nn.ModuleList([
            MBConvBlock(24, 32, 3, 2, 6, ['0254.weights_0.json', '0338.weights_0.json', '0077.weights_0.json'], ['', '0268.biases_0.json', '0295.biases_0.json'],
                        with_bn=True, bn_init_list=['0317.var_0.json', '0346.gamma_0.json', '0200.mean_0.json', '0038.beta_0.json']),
            MBConvBlock(32, 32, 3, 1, 6, ['0301.weights_0.json', '0355.weights_0.json', '0156.weights_0.json'], ['', '0185.biases_0.json', '0295.biases_1.json'],
                        with_bn=True, bn_init_list=['0119.var_0.json', '0085.gamma_0.json', '0303.mean_0.json', '0026.beta_0.json']),
            MBConvBlock(32, 32, 3, 1, 6, ['0301.weights_1.json', '0355.weights_1.json', '0156.weights_1.json'], ['0185.biases_1.json', '0185.biases_2.json', '0295.biases_2.json']),
            MBConvBlock(32, 32, 3, 1, 6, ['0301.weights_2.json', '0355.weights_2.json', '0156.weights_2.json'], ['0185.biases_3.json', '0185.biases_4.json', '0295.biases_3.json'])
        ]))

        # 2
        self.blocks.append(nn.ModuleList([
            MBConvBlock(32, 56, 5, 2, 6, ['0301.weights_3.json', '0022.weights_0.json', '0264.weights_0.json'], ['0185.biases_5.json', '0242.biases_0.json', '0260.biases_0.json']),
            MBConvBlock(56, 56, 5, 1, 6, ['0286.weights_0.json', '0104.weights_0.json', '0044.weights_0.json'], ['', '0202.biases_0.json', '0260.biases_1.json'],
                        with_bn=True, bn_init_list=['0258.var_0.json', '0272.gamma_0.json', '0148.mean_0.json', '0144.beta_0.json']),
            MBConvBlock(56, 56, 5, 1, 6, ['0286.weights_1.json', '0104.weights_1.json', '0044.weights_1.json'], ['0202.biases_1.json', '0202.biases_2.json', '0260.biases_2.json']),
            MBConvBlock(56, 56, 5, 1, 6, ['0286.weights_2.json', '0104.weights_2.json', '0044.weights_2.json'], ['0202.biases_3.json', '0202.biases_4.json', '0260.biases_3.json'])
        ]))

        # 3
        self.blocks.append(nn.ModuleList([
            MBConvBlock(56, 112, 3, 2, 6, ['0286.weights_3.json', '0096.weights_0.json', '0209.weights_0.json'], ['0202.biases_5.json', '0050.biases_0.json', '0019.biases_0.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0034.weights_0.json', '0214.weights_0.json', '0181.weights_0.json'], ['', '0216.biases_0.json', '0019.biases_1.json'],
                        with_bn=True, bn_init_list=['0059.var_0.json', '0226.gamma_0.json', '0325.mean_0.json', '0236.beta_0.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0034.weights_1.json', '0214.weights_1.json', '0181.weights_1.json'], ['0216.biases_1.json', '0216.biases_2.json', '0019.biases_2.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0034.weights_2.json', '0214.weights_2.json', '0181.weights_2.json'], ['0216.biases_3.json', '0216.biases_4.json', '0019.biases_3.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0034.weights_3.json', '0214.weights_3.json', '0181.weights_3.json'], ['0216.biases_5.json', '0216.biases_6.json', '0019.biases_4.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0034.weights_4.json', '0214.weights_4.json', '0181.weights_4.json'], ['0216.biases_7.json', '0216.biases_8.json', '0019.biases_5.json']),
        ]))

        # 4
        self.blocks.append(nn.ModuleList([
            MBConvBlock(112, 160, 5, 1, 6, ['0034.weights_5.json', '0134.weights_0.json', '0222.weights_0.json'], ['0216.biases_9.json', '0216.biases_10.json', '0125.biases_0.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0240.weights_0.json', '0205.weights_0.json', '0321.weights_0.json'], ['', '0052.biases_0.json', '0125.biases_1.json'],
                        with_bn=True, bn_init_list=['0342.var_0.json', '0244.gamma_0.json', '0106.mean_0.json', '0087.beta_0.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0240.weights_1.json', '0205.weights_1.json', '0321.weights_1.json'], ['0052.biases_1.json', '0052.biases_2.json', '0125.biases_2.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0240.weights_2.json', '0205.weights_2.json', '0321.weights_2.json'], ['0052.biases_3.json', '0052.biases_4.json', '0125.biases_3.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0240.weights_3.json', '0205.weights_3.json', '0321.weights_3.json'], ['0052.biases_5.json', '0052.biases_6.json', '0125.biases_4.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0240.weights_4.json', '0205.weights_4.json', '0321.weights_4.json'], ['0052.biases_7.json', '0052.biases_8.json', '0125.biases_5.json'])
        ]))

        # 5
        self.blocks.append(nn.ModuleList([
            MBConvBlock(160, 272, 5, 2, 6, ['0240.weights_5.json', '0190.weights_0.json', '0091.weights_0.json'], ['0052.biases_5.json', '0309.biases_0.json', '0152.biases_0.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0083.weights_0.json', '0101.weights_0.json', '0291.weights_0.json'], ['', '0115.biases_0.json', '0152.biases_1.json'],
                        with_bn=True, bn_init_list=['0335.var_0.json', '0192.gamma_0.json', '0098.mean_0.json', '0329.beta_0.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0083.weights_1.json', '0101.weights_1.json', '0291.weights_1.json'], ['0115.biases_1.json', '0115.biases_2.json', '0152.biases_2.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0083.weights_2.json', '0101.weights_2.json', '0291.weights_2.json'], ['0115.biases_3.json', '0115.biases_4.json', '0152.biases_3.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0083.weights_3.json', '0101.weights_3.json', '0291.weights_3.json'], ['0115.biases_5.json', '0115.biases_6.json', '0152.biases_4.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0083.weights_4.json', '0101.weights_4.json', '0291.weights_4.json'], ['0115.biases_7.json', '0115.biases_8.json', '0152.biases_5.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0083.weights_5.json', '0101.weights_5.json', '0291.weights_5.json'], ['0115.biases_9.json', '0115.biases_10.json', '0152.biases_6.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0083.weights_6.json', '0101.weights_6.json', '0291.weights_6.json'], ['0115.biases_11.json', '0115.biases_12.json', '0152.biases_7.json'])
        ]))

        # 6
        self.blocks.append(nn.ModuleList([
            MBConvBlock(272, 448, 3, 1, 6, ['0083.weights_7.json', '0111.weights_0.json', '0142.weights_0.json'], ['0115.biases_13.json', '0115.biases_14.json', '0162.biases_0.json'])
        ]))

        # Head
        in_channels = 448
        out_channels = 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )
        set_weights(self.head[0], '0065.weights_0.json')
        # set_biases(self.head[0], '0063.biases_0.json')
        set_var(self.head[1], '0282.var_0.json')
        set_bn_weights(self.head[1], '0218.gamma_0.json')
        set_mean(self.head[1], '0315.mean_0.json')
        set_biases(self.head[1], '0363.beta_0.json')

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(out_channels, num_classes)
        set_weights(self.fc, '0187.dense_weights_0.json')
        set_biases(self.fc, '0293.biases_0.json')

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
    input_cat = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.7/efficientnet_tvm_O0/cat.bin"
    new_cat = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.7/efficientnet_tvm_O0/cat_transpose.bin"
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
    exit(0)
