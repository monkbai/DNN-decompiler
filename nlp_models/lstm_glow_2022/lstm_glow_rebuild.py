import torch
import torch.nn as nn
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
    # if module.bias is not None:
    #     torch.nn.init.zeros_(module.bias)


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


class ElementwiseOp(nn.Module):
    def __init__(self, arith='Add'):
        super(ElementwiseOp, self).__init__()
        self.arith = arith

    def forward(self, x1, x2):
        if self.arith == 'Add':
            out = x1 + x2
        elif self.arith == 'Sub':
            out = x1 - x2
        elif self.arith == 'Mul':
            out = x1 * x2
        elif self.arith == 'Concat':
            if len(x1.shape) == 1:
                x1 = x1.reshape((1, x1.shape[0]))
            x2 = x2.reshape((1, x2.shape[0]))

            out = torch.cat((x1, x2), 0)
            print(out.shape)
        return out


class LSTM(nn.Module):
    def __init__(self, ):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(9, 6)
        set_weights(self.embed, '0008.params_0.json')

        self.matmuls = []
        for i in range(36):
            self.matmuls.append(nn.Linear(in_features=6, out_features=6))
            set_weights(self.matmuls[-1], './0010.params_{}.json'.format(i))
            set_biases(self.matmuls[-1], './0010.biases_{}.json'.format(i))

        self.last_matmul = nn.Linear(in_features=6, out_features=3)
        set_weights(self.last_matmul, './0081.params_0.json')
        set_biases(self.last_matmul, './0081.biases_0.json')

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.add = ElementwiseOp(arith='Add')
        self.sub = ElementwiseOp(arith='Sub')
        self.mul = ElementwiseOp(arith='Mul')
        self.cat = ElementwiseOp(arith='Concat')
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.h1 = torch.tensor([0.725997269, 0.401943088, 0.368505448, 0.420361012, 0.288543254, 0.271911263])  # 0x409880
        self.h2 = torch.tensor([0.613749325, -0.407810897, 0.633142233, 0.100949034, 0.502200663, 0.209129766])  # 0x409ac0
        self.h3 = torch.tensor([-0.16099745, -0.193747699, -0.00115226966, 0.287639111, -0.0413968861, -0.378149092])  # 0x409640
        self.h4 = torch.tensor([0, 0, 0, 0, 0, 0])  # 0x4091c0
        self.h5 = torch.tensor([0.259007722, 0.782112896, 0.629726291, -0.199307173, 0.0231584515, -0.0966704041])  # 0x409400

    def forward(self, x):
        x = self.embed(x)

        x0 = x[0]
        x1 = self.matmuls[0](x0)
        x2 = self.add(x1, self.h1)
        x3 = self.sigmoid(x2)
        x4 = self.matmuls[1](x0)
        x5 = self.add(x4, self.h2)
        x6 = self.tanh(x5)
        x7 = self.mul(x3, x6)
        x8 = self.matmuls[2](x0)
        x9 = self.add(x8, self.h3)
        x10 = self.sigmoid(x9)
        x11 = self.mul(x10, self.h4)
        x12 = self.add(x11, x7)
        x13 = self.tanh(x12)
        x14 = self.matmuls[3](x0)
        x15 = self.add(x14, self.h5)
        x16 = self.sigmoid(x15)
        x17 = self.mul(x16, x13)
        x18 = self.matmuls[4](x17)

        x19 = x[1]
        x20 = self.matmuls[5](x19)
        x21 = self.add(x20, x18)
        x22 = self.sigmoid(x21)
        x23 = self.mul(x22, x12)
        x24 = self.matmuls[6](x17)
        x25 = self.matmuls[7](x19)
        x26 = self.add(x25, x24)
        x27 = self.sigmoid(x26)
        x28 = self.matmuls[8](x17)
        x29 = self.matmuls[9](x19)
        x30 = self.add(x29, x28)
        x31 = self.tanh(x30)
        x32 = self.mul(x27, x31)
        x33 = self.add(x23, x32)
        x34 = self.tanh(x33)
        x35 = self.matmuls[10](x17)
        x36 = self.matmuls[11](x19)
        x37 = self.add(x36, x35)
        x38 = self.sigmoid(x37)
        x39 = self.mul(x38, x34)
        x40 = self.matmuls[12](x39)

        x41 = x[2]
        x42 = self.matmuls[13](x41)
        x43 = self.add(x42, x40)
        x44 = self.sigmoid(x43)
        x45 = self.mul(x44, x33)
        x46 = self.matmuls[14](x39)
        x47 = self.matmuls[15](x41)
        x48 = self.add(x47, x46)
        x49 = self.sigmoid(x48)
        x50 = self.matmuls[16](x39)
        x51 = self.matmuls[17](x41)
        x52 = self.add(x51, x50)
        x53 = self.tanh(x52)
        x54 = self.mul(x49, x53)
        x55 = self.add(x45, x54)
        x56 = self.tanh(x55)
        x57 = self.matmuls[18](x39)
        x58 = self.matmuls[19](x41)
        x59 = self.add(x58, x57)
        x60 = self.sigmoid(x59)
        x61 = self.mul(x60, x56)
        x62 = self.matmuls[20](x61)

        x63 = x[3]
        x64 = self.matmuls[21](x63)
        x65 = self.add(x64, x62)
        x66 = self.sigmoid(x65)
        x67 = self.mul(x66, x55)
        x68 = self.matmuls[22](x61)
        x69 = self.matmuls[23](x63)
        x70 = self.add(x69, x68)
        x71 = self.sigmoid(x70)
        x72 = self.matmuls[24](x61)
        x73 = self.matmuls[25](x63)
        x74 = self.add(x73, x72)
        x75 = self.tanh(x74)
        x76 = self.mul(x71, x75)
        x77 = self.add(x67, x76)
        x78 = self.tanh(x77)
        x79 = self.matmuls[26](x61)
        x80 = self.matmuls[27](x63)
        x81 = self.add(x80, x79)
        x82 = self.sigmoid(x81)
        x83 = self.mul(x82, x78)
        x84 = self.matmuls[28](x83)

        x85 = x[4]
        x86 = self.matmuls[29](x85)
        x87 = self.add(x86, x84)
        x88 = self.sigmoid(x87)
        x89 = self.mul(x88, x77)
        x90 = self.matmuls[30](x83)
        x91 = self.matmuls[31](x85)
        x92 = self.add(x91, x90)
        x93 = self.sigmoid(x92)
        x94 = self.matmuls[32](x83)
        x95 = self.matmuls[33](x85)
        x96 = self.add(x95, x94)
        x97 = self.tanh(x96)
        x98 = self.mul(x93, x97)
        x99 = self.add(x89, x98)
        x100 = self.tanh(x99)
        x101 = self.matmuls[34](x83)
        x102 = self.matmuls[35](x85)
        x103 = self.add(x102, x101)
        x104 = self.sigmoid(x103)
        x105 = x17  # x105 = self.cat(x101, x17)  # x101 is covered by x17
        x106 = self.cat(x105, x39)
        x107 = self.cat(x106, x61)
        x108 = self.cat(x107, x83)
        x109 = self.mul(x104, x100)
        x110 = self.last_matmul(x108)

        out = self.logsoftmax(x110)
        return out


model = LSTM()
# print(model)

x = torch.zeros(5).long()
for i in range(5):
    x[i] = i

out = model(x)

print(out)
exit(0)
