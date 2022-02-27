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


class MyResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyResNet, self).__init__()
        net = []
        
        net.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))
        set_weights(net[-1], './0143.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=64, affine=False, track_running_stats=False))
        set_var(net[-1], './0021.var_0.json')
        set_bn_weights(net[-1], './0098.gamma_0.json')
        set_mean(net[-1], './0125.mean_0.json')
        set_biases(net[-1], './0024.beta_0.json')

        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0149.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=64, track_running_stats=False))
        set_var(net[-1], './0021.var_1.json')
        set_bn_weights(net[-1], './0098.gamma_3.json')
        set_mean(net[-1], './0125.mean_1.json')
        set_biases(net[-1], './0024.beta_2.json')
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0149.weights_1.json')
        
        net.append(nn.BatchNorm2d(num_features=64, track_running_stats=False))
        set_var(net[-1], './0021.var_2.json')
        set_bn_weights(net[-1], './0098.gamma_6.json')
        set_mean(net[-1], './0125.mean_2.json')
        set_biases(net[-1], './0024.beta_4.json')
        
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0149.weights_2.json')
        
        net.append(nn.BatchNorm2d(num_features=64, track_running_stats=False))
        set_var(net[-1], './0021.var_3.json')
        set_bn_weights(net[-1], './0098.gamma_9.json')
        set_mean(net[-1], './0125.mean_3.json')
        set_biases(net[-1], './0024.beta_6.json')
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0149.weights_3.json')
        
        net.append(nn.BatchNorm2d(num_features=64, track_running_stats=False))
        set_var(net[-1], './0021.var_4.json')
        set_bn_weights(net[-1], './0098.gamma_12.json')
        set_mean(net[-1], './0125.mean_4.json')
        set_biases(net[-1], './0024.beta_8.json')
        
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0))
        set_weights(net[-1], './0160.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=128, track_running_stats=False))
        set_var(net[-1], './0059.var_0.json')
        set_bn_weights(net[-1], './0107.gamma_0.json')
        set_mean(net[-1], './0128.mean_0.json')
        set_biases(net[-1], './0062.beta_0.json')

        # net.append(nn.Conv2d(in_channels=16, out_channels=128, kernel_size=6, stride=2, padding=1))  # wrong shape
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))  # correct the shape
        set_weights(net[-1], './0166.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=128, track_running_stats=False))
        set_var(net[-1], './0059.var_1.json')
        set_bn_weights(net[-1], './0107.gamma_3.json')
        set_mean(net[-1], './0128.mean_1.json')
        set_biases(net[-1], './0062.beta_2.json')
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0172.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=128, track_running_stats=False))
        set_var(net[-1], './0059.var_2.json')
        set_bn_weights(net[-1], './0107.gamma_6.json')
        set_mean(net[-1], './0128.mean_2.json')
        set_biases(net[-1], './0062.beta_4.json')
        
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0172.weights_1.json')
        
        net.append(nn.BatchNorm2d(num_features=128, track_running_stats=False))
        set_var(net[-1], './0059.var_3.json')
        set_bn_weights(net[-1], './0107.gamma_9.json')
        set_mean(net[-1], './0128.mean_3.json')
        set_biases(net[-1], './0062.beta_6.json')
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0172.weights_2.json')
        
        net.append(nn.BatchNorm2d(num_features=128, track_running_stats=False))
        set_var(net[-1], './0059.var_4.json')
        set_bn_weights(net[-1], './0107.gamma_12.json')
        set_mean(net[-1], './0128.mean_4.json')
        set_biases(net[-1], './0062.beta_8.json')
        
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0))
        set_weights(net[-1], './0177.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=256, track_running_stats=False))
        set_var(net[-1], './0071.var_0.json')
        set_bn_weights(net[-1], './0113.gamma_0.json')
        set_mean(net[-1], './0131.mean_0.json')
        set_biases(net[-1], './0027.beta_0.json')
        
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1))
        set_weights(net[-1], './0183.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=256, track_running_stats=False))
        set_var(net[-1], './0071.var_1.json')
        set_bn_weights(net[-1], './0113.gamma_3.json')
        set_mean(net[-1], './0131.mean_1.json')
        set_biases(net[-1], './0027.beta_2.json')
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0189.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=256, track_running_stats=False))
        set_var(net[-1], './0071.var_2.json')
        set_bn_weights(net[-1], './0113.gamma_6.json')
        set_mean(net[-1], './0131.mean_2.json')
        set_biases(net[-1], './0027.beta_4.json')
        
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0189.weights_1.json')
        
        net.append(nn.BatchNorm2d(num_features=256, track_running_stats=False))
        set_var(net[-1], './0071.var_3.json')
        set_bn_weights(net[-1], './0113.gamma_9.json')
        set_mean(net[-1], './0131.mean_3.json')
        set_biases(net[-1], './0027.beta_6.json')
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0189.weights_2.json')
        
        net.append(nn.BatchNorm2d(num_features=256, track_running_stats=False))
        set_var(net[-1], './0071.var_4.json')
        set_bn_weights(net[-1], './0113.gamma_12.json')
        set_mean(net[-1], './0131.mean_4.json')
        set_biases(net[-1], './0027.beta_8.json')
        
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0))
        set_weights(net[-1], './0194.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=512, track_running_stats=False))
        set_var(net[-1], './0036.var_0.json')
        set_bn_weights(net[-1], './0119.gamma_0.json')
        set_mean(net[-1], './0134.mean_0.json')
        set_biases(net[-1], './0039.beta_0.json')
        
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1))
        set_weights(net[-1], './0200.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=512, track_running_stats=False))
        set_var(net[-1], './0036.var_1.json')
        set_bn_weights(net[-1], './0119.gamma_3.json')
        set_mean(net[-1], './0134.mean_1.json')
        set_biases(net[-1], './0039.beta_2.json')
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0155.weights_0.json')
        
        net.append(nn.BatchNorm2d(num_features=512, track_running_stats=False))
        set_var(net[-1], './0036.var_2.json')
        set_bn_weights(net[-1], './0119.gamma_6.json')
        set_mean(net[-1], './0134.mean_2.json')
        set_biases(net[-1], './0039.beta_4.json')
        
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0155.weights_1.json')
        
        net.append(nn.BatchNorm2d(num_features=512, track_running_stats=False))
        set_var(net[-1], './0036.var_3.json')
        set_bn_weights(net[-1], './0119.gamma_9.json')
        set_mean(net[-1], './0134.mean_3.json')
        set_biases(net[-1], './0039.beta_6.json')
        net.append(nn.ReLU())
        
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        set_weights(net[-1], './0155.weights_2.json')
        
        net.append(nn.BatchNorm2d(num_features=512, track_running_stats=False))
        set_var(net[-1], './0036.var_4.json')
        set_bn_weights(net[-1], './0119.gamma_12.json')
        set_mean(net[-1], './0134.mean_4.json')
        set_biases(net[-1], './0039.beta_8.json')
        
        net.append(nn.ReLU())
        net.append(nn.AvgPool2d(kernel_size=7, stride=1))
        net.append(nn.Flatten())
        
        net.append(nn.Linear(in_features=512, out_features=1000))
        set_weights(net[-1], './0203.dense_weights_0.json')
        set_biases(net[-1], '0047.bias_add_0.json')
        self.net = nn.ModuleList(net)

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
    with open("/home/lifter/Documents/DL_compiler/BTD_DATA/TVM-v0.8/resnet18_tvm_O0/cat.bin", 'br') as f:
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

    exit(0)

    # Input to the model
    vgg.eval()
    batch_size = 1
    torch_out = vgg(x)

    # Export the model
    torch.onnx.export(vgg,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "resnet18_tvmO0_rebuild.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      training=True,   # to disable all optimization and keep the BN layers
                      do_constant_folding=False,
                      # do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      )

