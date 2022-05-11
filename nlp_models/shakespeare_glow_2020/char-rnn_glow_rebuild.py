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
        return out


class CharRNN(nn.Module):
    def __init__(self, ):
        super(CharRNN, self).__init__()
        self.embed = nn.Embedding(100, 50)
        set_weights(self.embed, '0008.params_0.json')

        self.matmul1 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul1, './0009.params_0.json')
        set_biases(self.matmul1, './0010.params_0.json')
        self.matmul2 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul2, './0009.params_1.json')
        set_biases(self.matmul2, './0010.params_1.json')
        self.matmul3 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul3, './0009.params_2.json')
        set_biases(self.matmul3, './0010.params_2.json')
        self.matmul4 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul4, './0009.params_3.json')
        set_biases(self.matmul4, './0010.params_3.json')
        self.matmul5 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul5, './0009.params_4.json')
        set_biases(self.matmul5, './0010.params_4.json')
        self.matmul6 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul6, './0009.params_5.json')
        set_biases(self.matmul6, './0010.params_5.json')
        self.matmul7 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul7, './0009.params_6.json')
        set_biases(self.matmul7, './0010.params_6.json')
        self.matmul8 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul8, './0009.params_7.json')
        set_biases(self.matmul8, './0010.params_7.json')
        self.matmul9 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul9, './0009.params_8.json')
        set_biases(self.matmul9, './0010.params_8.json')
        self.matmul10 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul10, './0009.params_9.json')
        set_biases(self.matmul10, './0010.params_9.json')
        self.matmul11 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul11, './0009.params_10.json')
        set_biases(self.matmul11, './0010.params_10.json')
        self.matmul12 = nn.Linear(in_features=50, out_features=50)
        set_weights(self.matmul12, './0009.params_11.json')
        set_biases(self.matmul12, './0010.params_11.json')
        self.matmul13 = nn.Linear(in_features=50, out_features=100)
        set_weights(self.matmul13, './0020.params_0.json')
        set_biases(self.matmul13, './0021.params_0.json')

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.add = ElementwiseOp(arith='Add')
        self.sub = ElementwiseOp(arith='Sub')
        self.mul = ElementwiseOp(arith='Mul')

        # self.linear = nn.Linear(in_features=50, out_features=100)
        # set_weights(self.linear, './0110.params_0.json')
        # set_biases(self.linear, './0110.biases_0.json')

    def forward(self, x, hidden_state):
        x = self.embed(x)

        # First GRU
        o0 = self.matmul1(x)
        o1 = self.matmul2(hidden_state)
        o2 = self.add(o0, o1)
        o3 = self.sigmoid(o2)

        o4 = self.matmul3(hidden_state)
        o5 = self.mul(o3, o4)
        o6 = self.matmul4(x)
        o7 = self.add(o6, o5)
        o8 = self.tanh(o7)

        o9 = self.matmul5(x)
        o10 = self.matmul6(hidden_state)
        o11 = self.add(o9, o10)
        o12 = self.sigmoid(o11)
        o13 = self.mul(o12, o8)  # mul_s
        o14 = self.sub(o8, o13)
        o15 = self.mul(o12, hidden_state)
        o16 = self.add(o14, o15)

        # Second GRU
        o17 = self.matmul7(o16)
        o18 = self.matmul8(hidden_state)
        o19 = self.add(o17, o18)
        o20 = self.sigmoid(o19)

        o21 = self.matmul9(hidden_state)
        o22 = self.mul(o20, o21)
        o23 = self.matmul10(o16)
        o24 = self.add(o23, o22)
        o25 = self.tanh(o24)

        o26 = self.matmul11(o16)
        o27 = self.matmul12(hidden_state)
        o28 = self.add(o26, o27)
        o29 = self.sigmoid(o28)
        o30 = self.mul(o29, o25)  # mul_s
        o31 = self.sub(o25, o30)
        o32 = self.mul(o29, hidden_state)
        o33 = self.add(o31, o32)

        o34 = self.matmul13(o33)
        return o34


model = CharRNN()
# print(model)
hidden_state = torch.zeros(1, 50)
x = torch.zeros(1).long()
x[0] = 36
out = model(x, hidden_state)

print(out)
exit(0)
