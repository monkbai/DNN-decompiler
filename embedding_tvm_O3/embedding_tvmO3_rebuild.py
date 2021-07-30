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


class NET(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        net = []
        net.append(nn.Embedding(25006, 100))
        set_weights(net[-1], '0019.sub_401DA0.params_0.json')

        net.append(nn.AvgPool2d((seq_len, 1)))

        net.append(nn.Linear(in_features=100, out_features=1))
        set_weights(net[-1], '0021.sub_402490.params_0.json')

        net.append(nn.Linear(in_features=1, out_features=1))
        set_biases(net[-1], '0027.sub_403030.params_0.json')

        self.net = net
        self.seq = nn.Sequential(*net)

    def forward(self, x):
        # out = self.seq(x)
        embedded = self.net[0](x)
        embedded = embedded.reshape(embedded.shape[0], 1, embedded.shape[1], embedded.shape[2])
        pooled = self.net[1](embedded)
        # pooled = pooled.squeeze(1)
        out = self.net[2](pooled)
        return out


if __name__ == "__main__":
    # x = torch.randint(0, 25000, size=(7, 1))
    # x = torch.tensor([[70, 24, 9, 676, 285, 816, 6514]])  # This film is terrible
    x = torch.tensor([[70, 24, 9, 113, 285, 816, 2382]])  # This film is great


    time1 = time.time()
    print('building the model:', end=' ')
    vgg = NET(seq_len=7)
    time2 = time.time()
    print('{}s'.format(time2 - time1))

    print('predicting the label:', end=' ')
    out = vgg(x)
    time3 = time.time()
    print('{}s'.format(time3 - time2))

    print(out.size())
    print(type(out))
    print(out)

