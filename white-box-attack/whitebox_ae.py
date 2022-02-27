import os
import sys
import time
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()
        #self.init_dataset()

    def init_param(self):
        self.gpus = torch.cuda.device_count()
        # self.transform = transforms.Compose([
        #                         transforms.Resize(self.args.image_size),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize((0.5,), (0.5,)),
        #                    ])

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle
                        )
        return data_loader

class TorchImageNetDatasetCov(Dataset):
    def __init__(self,
                 args,
                 image_dir='/root/data/IMAGE-NET/ILSVRC/Data/CLS-LOC/',
                 # label_file='SelectedLabel-100K.json',
                 label2index_file='ImageNetLabel2Index.json',
                 split='train'):
        super(TorchImageNetDatasetCov).__init__()
        self.args = args
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        # self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_list = []
        
        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        self.cat_list = sorted(os.listdir(self.image_dir))[:args.num_cat]

        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        # label = self.cat_list.index(label)
        index = self.label2index[label]
        assert int(index) < self.args.num_cat
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

if __name__ == '__main__':
    import argparse
    import torchattacks
    # from data_loader import DataLoader

    # import classifier as cifar10_models

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='PGD',
                            choices=['PGD', 'CW', 'FGSM'])
    parser.add_argument('--is_train', type=int, default=0,
                            choices=[0, 1])
    args = parser.parse_args()
    args.batch_size = 128
    args.num_workers = 4

    os.environ['TORCH_HOME'] = '/root/collection/ImageNet/'
    
    model = torchvision.models.__dict__['resnet18'](pretrained=True).cuda()
    model.eval()

    args.z_dim = 120
    args.image_size = 128
    args.num_cat = 1000

    train_data = TorchImageNetDatasetCov(args, split='train')
    test_data = TorchImageNetDatasetCov(args, split='val')

    loader = DataLoader(args)
    train_loader = loader.get_loader(train_data, False)
    test_loader = loader.get_loader(test_data, False)

    if args.alg == 'PGD':
        atk = torchattacks.PGD(model)
    elif args.alg == 'CW':
        atk = torchattacks.CW(model)
    elif args.alg == 'FGSM':
        atk = torchattacks.FGSM(model)

    name = '%s-%s-%s-%s.pt' % (args.alg, args.model, args.dataset, ('train' if args.is_train else 'test'))
    start_time = time.time()
    if args.is_train:
        atk.save(train_loader, save_path="/root/data/adversarial_examples/" + name, verbose=True)
    else:
        atk.save(test_loader, save_path="/root/data/adversarial_examples/" + name, verbose=True)
    print('Cost time: %f' % (time.time() - start_time))
