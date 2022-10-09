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
        set_weights(self.stem[0], '0305.weights_0.json')
        set_biases(self.stem[0], '0252.biases_0.json')

        # Build blocks
        self.blocks = nn.ModuleList([])

        # 0
        self.blocks.append(nn.ModuleList([
            MBConvBlock(32, 24, 3, 1, 1, ['0309.weights_0.json', '0360.weights_0.json'], ['0252.biases_1.json', '0255.biases_0.json'])
        ]))

        # 1
        self.blocks.append(nn.ModuleList([
            MBConvBlock(24, 32, 3, 2, 6, ['0411.weights_0.json', '0425.weights_0.json', '0430.weights_0.json'], ['', '0279.biases_0.json', '0282.biases_0.json'],
                        with_bn=True, bn_init_list=['0021.var_0.json', '0185.gamma_0.json', '0227.mean_0.json', '0024.beta_0.json']),
            MBConvBlock(32, 32, 3, 1, 6, ['0435.weights_0.json', '0439.weights_0.json', '0444.weights_0.json'], ['', '0285.biases_0.json', '0282.biases_1.json'],
                        with_bn=True, bn_init_list=['0080.var_0.json', '0203.gamma_0.json', '0230.mean_0.json', '0083.beta_0.json']),
            MBConvBlock(32, 32, 3, 1, 6, ['0435.weights_1.json', '0439.weights_1.json', '0444.weights_1.json'], ['0285.biases_1.json', '0285.biases_2.json', '0282.biases_2.json']),
            MBConvBlock(32, 32, 3, 1, 6, ['0435.weights_2.json', '0439.weights_2.json', '0444.weights_2.json'], ['0285.biases_3.json', '0285.biases_4.json', '0282.biases_3.json'])
        ]))

        # 2
        self.blocks.append(nn.ModuleList([
            MBConvBlock(32, 56, 5, 2, 6, ['0435.weights_3.json', '0448.weights_0.json', '0314.weights_0.json'], ['0285.biases_5.json', '0288.biases_0.json', '0291.biases_0.json']),
            MBConvBlock(56, 56, 5, 1, 6, ['0319.weights_0.json', '0323.weights_0.json', '0328.weights_0.json'], ['', '0294.biases_0.json', '0291.biases_1.json'],
                        with_bn=True, bn_init_list=['0092.var_0.json', '0209.gamma_0.json', '0233.mean_0.json', '0095.beta_0.json']),
            MBConvBlock(56, 56, 5, 1, 6, ['0319.weights_1.json', '0323.weights_1.json', '0328.weights_1.json'], ['0294.biases_1.json', '0294.biases_2.json', '0291.biases_2.json']),
            MBConvBlock(56, 56, 5, 1, 6, ['0319.weights_2.json', '0323.weights_2.json', '0328.weights_2.json'], ['0294.biases_3.json', '0294.biases_4.json', '0291.biases_3.json'])
        ]))

        # 3
        self.blocks.append(nn.ModuleList([
            MBConvBlock(56, 112, 3, 2, 6, ['0319.weights_3.json', '0332.weights_0.json', '0337.weights_0.json'], ['0294.biases_5.json', '0297.biases_0.json', '0300.biases_0.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0342.weights_0.json', '0346.weights_0.json', '0351.weights_0.json'], ['', '0258.biases_0.json', '0300.biases_1.json'],
                        with_bn=True, bn_init_list=['0030.var_0.json', '0215.gamma_0.json', '0236.mean_0.json', '0033.beta_0.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0342.weights_1.json', '0346.weights_1.json', '0351.weights_1.json'], ['0258.biases_1.json', '0258.biases_2.json', '0300.biases_2.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0342.weights_2.json', '0346.weights_2.json', '0351.weights_2.json'], ['0258.biases_3.json', '0258.biases_4.json', '0300.biases_3.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0342.weights_3.json', '0346.weights_3.json', '0351.weights_3.json'], ['0258.biases_5.json', '0258.biases_6.json', '0300.biases_4.json']),
            MBConvBlock(112, 112, 3, 1, 6, ['0342.weights_4.json', '0346.weights_4.json', '0351.weights_4.json'], ['0258.biases_7.json', '0258.biases_8.json', '0300.biases_5.json']),
        ]))

        # 4
        self.blocks.append(nn.ModuleList([
            MBConvBlock(112, 160, 5, 1, 6, ['0342.weights_5.json', '0355.weights_0.json', '0365.weights_0.json'], ['0258.biases_9.json', '0258.biases_10.json', '0261.biases_0.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0370.weights_0.json', '0374.weights_0.json', '0379.weights_0.json'], ['', '0264.biases_0.json', '0261.biases_1.json'],
                        with_bn=True, bn_init_list=['0042.var_0.json', '0221.gamma_0.json', '0239.mean_0.json', '0045.beta_0.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0370.weights_1.json', '0374.weights_1.json', '0379.weights_1.json'], ['0264.biases_1.json', '0264.biases_2.json', '0261.biases_2.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0370.weights_2.json', '0374.weights_2.json', '0379.weights_2.json'], ['0264.biases_3.json', '0264.biases_4.json', '0261.biases_3.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0370.weights_3.json', '0374.weights_3.json', '0379.weights_3.json'], ['0264.biases_5.json', '0264.biases_6.json', '0261.biases_4.json']),
            MBConvBlock(160, 160, 5, 1, 6, ['0370.weights_4.json', '0374.weights_4.json', '0379.weights_4.json'], ['0264.biases_7.json', '0264.biases_8.json', '0261.biases_5.json'])
        ]))

        # 5
        self.blocks.append(nn.ModuleList([
            MBConvBlock(160, 272, 5, 2, 6, ['0370.weights_5.json', '0383.weights_0.json', '0388.weights_0.json'], ['0264.biases_5.json', '0267.biases_0.json', '0270.biases_0.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0393.weights_0.json', '0397.weights_0.json', '0402.weights_0.json'], ['', '0273.biases_0.json', '0270.biases_1.json'],
                        with_bn=True, bn_init_list=['0054.var_0.json', '0191.gamma_0.json', '0242.mean_0.json', '0060.beta_0.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0393.weights_1.json', '0397.weights_1.json', '0402.weights_1.json'], ['0273.biases_1.json', '0273.biases_2.json', '0270.biases_2.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0393.weights_2.json', '0397.weights_2.json', '0402.weights_2.json'], ['0273.biases_3.json', '0273.biases_4.json', '0270.biases_3.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0393.weights_3.json', '0397.weights_3.json', '0402.weights_3.json'], ['0273.biases_5.json', '0273.biases_6.json', '0270.biases_4.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0393.weights_4.json', '0397.weights_4.json', '0402.weights_4.json'], ['0273.biases_7.json', '0273.biases_8.json', '0270.biases_5.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0393.weights_5.json', '0397.weights_5.json', '0402.weights_5.json'], ['0273.biases_9.json', '0273.biases_10.json', '0270.biases_6.json']),
            MBConvBlock(272, 272, 5, 1, 6, ['0393.weights_6.json', '0397.weights_6.json', '0402.weights_6.json'], ['0273.biases_11.json', '0273.biases_12.json', '0270.biases_7.json'])
        ]))

        # 6
        self.blocks.append(nn.ModuleList([
            MBConvBlock(272, 448, 3, 1, 6, ['0393.weights_7.json', '0406.weights_0.json', '0416.weights_0.json'], ['0273.biases_13.json', '0273.biases_14.json', '0276.biases_0.json'])
        ]))

        # Head
        in_channels = 448
        out_channels = 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )
        set_weights(self.head[0], '0421.weights_0.json')
        # set_biases(self.head[0], '0063.biases_0.json')
        set_var(self.head[1], '0069.var_0.json')
        set_bn_weights(self.head[1], '0197.gamma_0.json')
        set_mean(self.head[1], '0245.mean_0.json')
        set_biases(self.head[1], '0072.beta_0.json')

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(out_channels, num_classes)
        set_weights(self.fc, '0451.dense_weights_0.json')
        set_biases(self.fc, '0077.biases_0.json')

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
    input_cat = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.8/efficientnet_tvm_O0/cat.bin"
    new_cat = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/TVM-v0.8/efficientnet_tvm_O0/cat_transpose.bin"
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
