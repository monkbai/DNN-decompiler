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


# Inception module
class Block(nn.Module):
    def __init__(self, in_channels, out_chanel_1, out_channel_3_reduce, out_channel_3,
                 out_channel_5_reduce, out_channel_5, out_channel_pool):
        super(Block, self).__init__()

        block = []
        self.block1 = nn.Conv2d(in_channels=in_channels, out_channels=out_chanel_1, kernel_size=1)
        block.append(self.block1)
        
        self.block2_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel_3_reduce, kernel_size=1)
        self.relu2_1 = nn.ReLU()
        self.block2 = nn.Conv2d(in_channels=out_channel_3_reduce, out_channels=out_channel_3, kernel_size=3, padding=1)
        block.append(self.block2)
        
        self.block3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel_5_reduce, kernel_size=1)
        self.relu3_1 = nn.ReLU()
        self.block3 = nn.Conv2d(in_channels=out_channel_5_reduce, out_channels=out_channel_5, kernel_size=3, padding=2, stride=1)
        block.append(self.block3)
        
        self.block4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.block4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel_pool, kernel_size=1)
        block.append(self.block4)

        self.relu = nn.ReLU()
        # self.incep = nn.Sequential(*block)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(self.relu2_1(self.block2_1(x)))
        out3 = self.block3(self.relu3_1(self.block3_1(x)))
        out4 = self.block4(self.block4_1(x))
        # print(out1.shape, out2.shape, out3.shape, out4.shape)  # debug
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.relu(out)
        # print(out.shape)  # debug
        return out


def set_block(blk: Block, w1: str, b1: str, w2_1: str, b2_1: str, w2: str, b2: str, w3_1: str, b3_1: str, w3: str, b3: str, w4: str, b4: str):
    set_weights(blk.block1, w1)
    set_biases(blk.block1, b1)

    set_weights(blk.block2_1, w2_1)
    set_biases(blk.block2_1, b2_1)
    set_weights(blk.block2, w2)
    set_biases(blk.block2, b2)

    set_weights(blk.block3_1, w3_1)
    set_biases(blk.block3_1, b3_1)
    set_weights(blk.block3, w3)
    set_biases(blk.block3, b3)

    set_weights(blk.block4, w4)
    set_biases(blk.block4, b4)


# class InceptionClassifiction(nn.Module):
#     def __init__(self, in_channels,out_channels):
#         super(InceptionClassifiction, self).__init__()

#         self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
#         self.linear1 = nn.Linear(in_features=128 * 4 * 4, out_features=1024)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=0.7)
#         self.linear2 = nn.Linear(in_features=1024, out_features=out_channels)

#     def forward(self, x):
#         x = self.conv1(self.avgpool(x))
#         x = x.view(x.size(0), -1)
#         x= self.relu(self.linear1(x))
#         out = self.linear2(self.dropout(x))
#         return out

class InceptionV1(nn.Module):
    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV1, self).__init__()
        self.stage = stage

        self.blockA = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2, padding=0),
            nn.LocalResponseNorm(5),

        )
        set_weights(self.blockA[0], './0010.weights_0.json')
        set_biases(self.blockA[0], '0010.biases_0.json')
        
        self.blockB = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        set_weights(self.blockB[0], './0013.weights_0.json')
        set_biases(self.blockB[0], '0013.biases_0.json')
        set_weights(self.blockB[2], './0014.weights_0.json')
        set_biases(self.blockB[2], '0014.biases_0.json')

        self.blockC = nn.Sequential(
            Block(in_channels=192,out_chanel_1=64, out_channel_3_reduce=96, out_channel_3=128,
                  out_channel_5_reduce = 16, out_channel_5=32, out_channel_pool=32), 
            Block(in_channels=256, out_chanel_1=128, out_channel_3_reduce=128, out_channel_3=192,
                  out_channel_5_reduce=32, out_channel_5=96, out_channel_pool=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        set_block(self.blockC[0], 
                  '0021.weights_0.json', '0021.biases_0.json', 
                  '0022.weights_0.json', '0022.biases_0.json', 
                  '0023.weights_0.json', '0023.biases_0.json', 
                  '0017.weights_0.json', '0017.biases_0.json',
                  '0018.weights_0.json', '0018.biases_0.json',
                  '0020.weights_0.json', '0020.biases_0.json')
        set_block(self.blockC[1],  
                  '0032.weights_0.json', '0032.biases_0.json', 
                  '0033.weights_0.json', '0033.biases_0.json', 
                  '0034.weights_0.json', '0034.biases_0.json',
                  '0030.weights_0.json', '0030.biases_0.json',
                  '0031.weights_0.json', '0031.biases_0.json',
                  '0029.weights_0.json', '0029.biases_0.json')

        self.blockD_1 = Block(in_channels=480, out_chanel_1=192, out_channel_3_reduce=96, out_channel_3=208,
                              out_channel_5_reduce=16, out_channel_5=48, out_channel_pool=64)
        set_block(self.blockD_1,  
                  '0045.weights_0.json', '0045.biases_0.json', 
                  '0046.weights_0.json', '0046.biases_0.json', 
                  '0047.weights_0.json', '0047.biases_0.json',
                  '0041.weights_0.json', '0041.biases_0.json',
                  '0042.weights_0.json', '0042.biases_0.json',
                  '0044.weights_0.json', '0044.biases_0.json')
        # if self.stage == 'train':
        #     self.Classifiction_logits1 = InceptionClassifiction(in_channels=512,out_channels=num_classes)

        self.blockD_2 = nn.Sequential(
            Block(in_channels=512, out_chanel_1=160, out_channel_3_reduce=112, out_channel_3=224,
                              out_channel_5_reduce=24, out_channel_5=64, out_channel_pool=64),
            Block(in_channels=512, out_chanel_1=128, out_channel_3_reduce=128, out_channel_3=256,
                              out_channel_5_reduce=24, out_channel_5=64, out_channel_pool=64),
            Block(in_channels=512, out_chanel_1=112, out_channel_3_reduce=144, out_channel_3=288,
                              out_channel_5_reduce=32, out_channel_5=64, out_channel_pool=64),
        )
        set_block(self.blockD_2[0],  
                  '0057.weights_0.json', '0057.biases_0.json', 
                  '0058.weights_0.json', '0058.biases_0.json', 
                  '0059.weights_0.json', '0059.biases_0.json',
                  '0053.weights_0.json', '0053.biases_0.json',
                  '0054.weights_0.json', '0054.biases_0.json',
                  '0056.weights_0.json', '0056.biases_0.json')
        set_block(self.blockD_2[1],  
                  '0063.weights_0.json', '0063.biases_0.json', 
                  '0064.weights_0.json', '0064.biases_0.json', 
                  '0065.weights_0.json', '0065.biases_0.json',
                  '0053.weights_1.json', '0053.biases_1.json',
                  '0054.weights_1.json', '0054.biases_1.json',
                  '0056.weights_1.json', '0056.biases_1.json')
        set_block(self.blockD_2[2],  
                  '0071.weights_0.json', '0071.biases_0.json', 
                  '0072.weights_0.json', '0072.biases_0.json', 
                  '0073.weights_0.json', '0073.biases_0.json',
                  '0069.weights_0.json', '0069.biases_0.json',
                  '0070.weights_0.json', '0070.biases_0.json',
                  '0056.weights_2.json', '0056.biases_2.json')

        # if self.stage == 'train':
        #     self.Classifiction_logits2 = InceptionClassifiction(in_channels=528,out_channels=num_classes)

        self.blockD_3 = nn.Sequential(
            Block(in_channels=528, out_chanel_1=256, out_channel_3_reduce=160, out_channel_3=320,
                              out_channel_5_reduce=32, out_channel_5=128, out_channel_pool=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        set_block(self.blockD_3[0],  
                  '0082.weights_0.json', '0082.biases_0.json', 
                  '0083.weights_0.json', '0083.biases_0.json', 
                  '0084.weights_0.json', '0084.biases_0.json',
                  '0078.weights_0.json', '0078.biases_0.json',
                  '0079.weights_0.json', '0079.biases_0.json',
                  '0081.weights_0.json', '0081.biases_0.json')

        self.blockE = nn.Sequential(
            Block(in_channels=832, out_chanel_1=256, out_channel_3_reduce=160, out_channel_3=320,
                  out_channel_5_reduce=32, out_channel_5=128, out_channel_pool=128),
            Block(in_channels=832, out_chanel_1=384, out_channel_3_reduce=192, out_channel_3=384,
                  out_channel_5_reduce=48, out_channel_5=128, out_channel_pool=128),
        )
        set_block(self.blockE[0],  
                  '0094.weights_0.json', '0094.biases_0.json', 
                  '0095.weights_0.json', '0095.biases_0.json', 
                  '0096.weights_0.json', '0096.biases_0.json',
                  '0090.weights_0.json', '0090.biases_0.json',
                  '0091.weights_0.json', '0091.biases_0.json',
                  '0093.weights_0.json', '0093.biases_0.json')
        set_block(self.blockE[1],  
                  '0103.weights_0.json', '0103.biases_0.json', 
                  '0104.weights_0.json', '0104.biases_0.json', 
                  '0105.weights_0.json', '0105.biases_0.json',
                  '0101.weights_0.json', '0101.biases_0.json',
                  '0102.weights_0.json', '0102.biases_0.json',
                  '0093.weights_1.json', '0093.biases_1.json')

        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        # self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=1024,out_features=num_classes)
        set_weights(self.linear, './0110.params_0.json')
        set_biases(self.linear, './0110.biases_0.json')
        
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.blockA(x)
        x = self.blockB(x)
        x = self.blockC(x)
        Classifiction1 = x = self.blockD_1(x)
        Classifiction2 = x = self.blockD_2(x)
        x = self.blockD_3(x)
        # print(x[0])  # debug
        out = self.blockE(x)
        out = self.avgpool(out)
        # out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.stage == 'train':
            # Classifiction1 = self.Classifiction_logits1(Classifiction1)
            # Classifiction2 = self.Classifiction_logits2(Classifiction2)
            # return Classifiction1, Classifiction2, out
            return out
        else:
            return self.softmax(out)


model = InceptionV1(num_classes=1000, stage='test')
# print(model)

# input = torch.randn(1, 3, 224, 224)
with open("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2022/inception_v1/cat.bin", 'br') as f:
        bin_data = f.read()
        np_arr = np.frombuffer(bin_data, dtype=np.float32)
        print(np_arr.shape)
        np_arr = np_arr.reshape(3, 224, 224)
        np_arr = np_arr.reshape((1, 3, 224, 224))
        x = torch.Tensor(np_arr)
        print(x.shape)
input = x
out = model(input)

max_index = np.argmax(out.detach().numpy())
print(max_index)
# print(out)
print(out.detach().numpy()[0, max_index])
exit(0)
