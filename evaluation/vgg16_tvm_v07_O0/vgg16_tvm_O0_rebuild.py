import sys
import torch.nn as nn
import torch
import json
import numpy as np
import time

def read_json(json_path: str):
    with open(json_path, 'r') as f:
        j_txt = f.read()
        list_obj = json.loads(s=j_txt)
        arr_obj = np.array(list_obj, dtype=np.float32)
        tensor_obj = torch.from_numpy(arr_obj)
        return tensor_obj


def set_weights(module: nn.modules, json_path: str):
    # https://stackoverflow.com/a/59468760
    w = read_json(json_path)
    module.weight = torch.nn.Parameter(w)


def set_biases(module: nn.modules, json_path: str):
    # https://stackoverflow.com/a/59468760
    w = read_json(json_path)
    w = w.reshape(w.shape[1])
    module.bias = torch.nn.Parameter(w)


class SE_VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        net = []

        # block 1
        net.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1))  # 0
        set_weights(net[0], './0064.weights_0.json')
        set_biases(net[0], './0019.biases_0.json')
        net.append(nn.ReLU())  # 1
        # the input channels is the output channels of previous conv2d layer
        net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1))  # 2
        set_weights(net[2], './0069.weights_0.json')
        set_biases(net[2], './0019.biases_1.json')
        net.append(nn.ReLU())  # 3
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 4

        # block 2
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))  # 5
        set_weights(net[5], './0081.weights_0.json')
        set_biases(net[5], './0050.biases_0.json')
        net.append(nn.ReLU())  # 6
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))  # 7
        set_weights(net[7], './0103.weights_0.json')
        set_biases(net[7], './0050.biases_1.json')
        net.append(nn.ReLU())  # 8
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 9

        # block 3
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))  # 10
        set_weights(net[10], './0110.weights_0.json')
        set_biases(net[10], './0105.biases_0.json')
        net.append(nn.ReLU())  # 11
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))  # 12
        set_weights(net[12], './0094.weights_0.json')
        set_biases(net[12], './0105.biases_1.json')
        net.append(nn.ReLU())  # 13
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))  # 14
        set_weights(net[14], './0094.weights_1.json')
        set_biases(net[14], './0105.biases_2.json')
        net.append(nn.ReLU())  # 15
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 16

        # block 4
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))  # 17
        set_weights(net[17], './0030.weights_0.json')
        set_biases(net[17], './0036.biases_0.json')
        net.append(nn.ReLU())  # 18
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 19
        set_weights(net[19], './0057.weights_0.json')
        set_biases(net[19], './0036.biases_1.json')
        net.append(nn.ReLU())  # 20
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 21
        set_weights(net[21], './0057.weights_1.json')
        set_biases(net[21], './0036.biases_2.json')
        net.append(nn.ReLU())  # 22
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 23

        # block 5
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 24
        set_weights(net[24], './0074.weights_0.json')
        set_biases(net[24], './0025.biases_0.json')
        net.append(nn.ReLU())  # 25
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 26
        set_weights(net[26], './0074.weights_1.json')
        set_biases(net[26], './0025.biases_1.json')
        net.append(nn.ReLU())  # 27
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 28
        set_weights(net[28], './0074.weights_2.json')
        set_biases(net[28], './0025.biases_2.json')
        net.append(nn.ReLU())  # 29
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 30

        # add net into class property
        self.extract_feature = nn.Sequential(*net)

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=512*7*7, out_features=4096))
        #print(type(classifier[0].weight))
        #print(classifier[0].weight.shape)
        #print(classifier[0].bias.shape)
        set_weights(classifier[-1], './0040.dense_weights_0.json')
        set_biases(classifier[-1], './0046.biases_0.json')
        #print(classifier[0].bias.shape)
        #print(classifier[0].bias)
        classifier.append(nn.ReLU())
        # classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=4096))
        set_weights(classifier[-1], './0038.dense_weights_0.json')
        set_biases(classifier[-1], './0046.biases_1.json')
        classifier.append(nn.ReLU())
        # classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=self.num_classes))
        set_weights(classifier[-1], './0042.dense_weights_0.json')
        set_biases(classifier[-1], './0034.biases_0.json')

        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        feature = self.extract_feature(x)
        feature = feature.view(x.size(0), -1)
        classify_result = self.classifier(feature)
        return classify_result


if __name__ == "__main__":
    # x = torch.rand(size=(1, 3, 224, 224))
    input_cat = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/cat.bin"
    if len(sys.argv) == 2:
        input_cat = sys.argv[1]
        
    with open(input_cat, 'br') as f:
        bin_data = f.read()
        np_arr = np.frombuffer(bin_data, dtype=np.float32)
        # print(np_arr.shape)
        np_arr = np_arr.reshape(3, 224, 224)
        np_arr = np_arr.reshape((1, 3, 224, 224))
        x = torch.Tensor(np_arr)
        # print(x.shape)

    time1 = time.time()
    # print('building the model:', end=' ')
    vgg = SE_VGG(num_classes=1000)
    time2 = time.time()
    # print('{}s'.format(time2 - time1))

    # print('predicting the label:', end=' ')
    out = vgg(x)
    time3 = time.time()
    # print('{}s'.format(time3 - time2))

    # print(out.size())
    # print(type(out))
    max_index = np.argmax(out.detach().numpy())
    print("Result:", max_index)
    # print(out)
    print("Confidence:", out.detach().numpy()[0, max_index])

    exit(0)
    # Input to the model
    vgg.eval()
    batch_size = 1
    torch_out = vgg(x)

    # Export the model
    torch.onnx.export(vgg,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "vgg16_tvmO0_rebuild.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      # do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      )
    exit(0)