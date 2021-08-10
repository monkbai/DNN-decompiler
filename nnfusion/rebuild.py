#!/usr/bin/python3
#!/usr/bin/python3
# =========================================================
# Code usde to run nnfusion executbale 
import os
import time
import json
import numpy as np
import subprocess


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def cmd(commandline):
    with cd(project_dir):
        # print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


project_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/nnfusion/"  # linrary and Constant in this file

# =========================================================

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
    if len(w.shape) == 2:
        w = np.transpose(w, (1, 0))
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
        set_weights(net[-1], './0068.sub_406A50.weights_0.json')
        set_biases(net[-1], './0055.sub_404C00.params_0.json')
        net.append(nn.ReLU())  # 1
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 2

        # block 2
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))  # 3
        set_weights(net[-1], './0070.sub_406Ec0.weights_0.json')
        set_biases(net[-1], './0054.sub_404A90.params_0.json')
        net.append(nn.ReLU())  # 4
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 5

        # block 3
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))  # 10
        set_weights(net[-1], './0046.sub_4040f0.weights_0.json')
        set_biases(net[-1], './0052.sub_4047b0.params_0.json')
        net.append(nn.ReLU())  # 11
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))  # 12
        set_weights(net[-1], './0057.sub_404Ee0.weights_0.json')
        set_biases(net[-1], './0052.sub_4047b0.params_1.json')
        net.append(nn.ReLU())  # 13
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 16

        # block 4
        net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))  # 17
        set_weights(net[-1], './0081.sub_408650.weights_0.json')
        set_biases(net[-1], './0066.sub_406630.params_0.json')
        net.append(nn.ReLU())  # 18
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 19
        set_weights(net[-1], './0051.sub_4045e0.weights_0.json')
        set_biases(net[-1], './0066.sub_406630.params_1.json')
        net.append(nn.ReLU())  # 20
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 23

        # block 5
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 24
        set_weights(net[-1], './0044.sub_403Ed0.weights_0.json')
        set_biases(net[-1], './0056.sub_404D70.params_0.json')
        net.append(nn.ReLU())  # 25
        net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))  # 26
        set_weights(net[-1], './0044.sub_403Ed0.weights_1.json')
        set_biases(net[-1], './0056.sub_404D70.params_1.json')
        net.append(nn.ReLU())  # 27
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 30

        # add net into class property
        self.extract_feature = nn.Sequential(*net)
        self.net = net

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=512 * 7 * 7, out_features=4096))
        set_weights(classifier[-1], './0050.sub_404590.params_0.json')
        #print(classifier[-1].weight.shape)
        set_biases(classifier[-1], './0049.sub_404460.params_0.json')
        classifier.append(nn.ReLU())

        classifier.append(nn.Linear(in_features=4096, out_features=4096))
        set_weights(classifier[-1], './0045.sub_4040A0.params_0.json')
        set_biases(classifier[-1], './0049.sub_404460.params_1.json')
        classifier.append(nn.ReLU())

        classifier.append(nn.Linear(in_features=4096, out_features=self.num_classes))
        #print(classifier[-1].weight.shape)
        set_weights(classifier[-1], './0043.sub_403E80.params_0.json')
        #print(classifier[-1].weight.shape)
        set_biases(classifier[-1], './0074.sub_407760.params_0.json')

        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        # for debug
        #out0 = self.net[0](x)
        #print(out0)
        feature = self.extract_feature(x)
        feature = feature.view(x.size(0), -1)
        classify_result = self.classifier(feature)
        return classify_result


def validate():
    vgg = SE_VGG(num_classes=1001)

    input_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/imagenet_part/"
    files = os.listdir(input_dir)
    same_count = 0
    not_same_count = 0
    for f in files:
        if f.endswith(".BIN"):
            print(f)
            f = os.path.join(input_dir, f)
            status, output = cmd("./vgg_nnfusion_strip {}".format(f))
            index_1 = output[output.find("Result: ")+8:]
            index_1 = index_1[:index_1.find("\n")].strip()
            max_index_1 = int(index_1)
            val_1 = output[output.find("Confidence: ")+12:]
            val_1 = val_1[:val_1.find("\n")].strip()
            max_val_1 = float(val_1)
            # print(max_index_1, max_val_1)

            with open(f, 'br') as f:
                bin_data = f.read()
                np_arr = np.frombuffer(bin_data, dtype=np.float32)
                print(np_arr.shape)
                np_arr = np_arr.reshape(224, 224, 3)
                np_arr = np.transpose(np_arr, (2, 0, 1))
                np_arr = np_arr.reshape((1, 3, 224, 224))
                x = torch.Tensor(np_arr)
                out = vgg(x)
                max_index_2 = np.argmax(out.detach().numpy())
                max_val_2 = out.detach().numpy()[0, max_index_2]
                # print(max_index_2, max_val_2)
                if abs(max_val_1-max_val_2) < 0.00001:
                    print('same')
                    same_count += 1
                else:
                    print('not same')
                    not_same_count += 1 
    print('same count', same_count)
    print('not same count', not_same_count)
    exit(0)



if __name__ == "__main__":
    validate()

    # x = torch.rand(size=(1, 3, 224, 224))
    with open("cat2.bin", 'br') as f:
        bin_data = f.read()
        np_arr = np.frombuffer(bin_data, dtype=np.float32)
        print(np_arr.shape)
        np_arr = np_arr.reshape(224, 224, 3)
        np_arr = np.transpose(np_arr, (2, 0, 1))
        np_arr = np_arr.reshape((1, 3, 224, 224))
        x = torch.Tensor(np_arr)
        print(x.shape)
    #print(x)
    time1 = time.time()
    print('building the model:', end=' ')
    vgg = SE_VGG(num_classes=1001)
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

    '''
    # Input to the model
    vgg.eval()
    torch_out = vgg(x)

    # Export the model
    torch.onnx.export(vgg,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "vgg11_nnfusion_rebuild.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      # do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      )
    '''
    
    '''
    # Export to ONNX, using freezer provided by nnfusion 
    from pytorch_freezer import IODescription, ModelDescription, PTFreezer
    import torch
    import torchvision
    # define pytorch module
    model = vgg
    # define the pytorch model input/output description
    input_desc = [IODescription("input", [1, 3, 224, 224], torch.float32)]
    output_desc = [IODescription("output", [1, 1000], torch.float32)]
    model_desc = ModelDescription(input_desc, output_desc)
    # save the onnx model somewhere
    freezer = PTFreezer(model, model_desc)
    freezer.execute("./vgg11_nnfusion_rebuild_2.onnx")
    '''

