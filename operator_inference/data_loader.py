import os
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import utils

def word_sort_key(word):
    return -len(word)


class OPSeqDataset(Dataset):
    def __init__(self, sent_dir, vocab_path, bpe_path, index_path, label_path, setting, pad_length=500):
        super(OPSeqDataset).__init__()
        self.max_length = -1

        self.bpe = utils.BPE(bpe_path)
        with open(vocab_path, 'r') as f:
            self.word_dict = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        with open(index_path, 'r') as f:
            self.label_index_dict = json.load(f)

        self.word_index_dict = {}
        for word in self.word_dict.keys():
            index = self.word_dict[word]
            self.word_index_dict[index] = word

        num_labels = len(self.label_index_dict.keys())
        # print('num_labels: ', num_labels)

        self.sent_list = []
        self.label_list = []
        self.name_list = []
        cat_list = sorted(os.listdir(sent_dir))
        for cat in cat_list:
            if not (setting in cat):
                continue
            label = [0] * num_labels
            for c in self.label_dict[cat]:
                label[self.label_index_dict[c]] = 1
            self.name_list.append(cat)
            with open(sent_dir + cat, 'r') as f:
                content = f.read()
                self.sent_list.append(content.strip().split(' ')) # append(['word1', 'word2', ...])
                self.label_list.append(label)

        self.indexes_list = [self.sent_to_indexes(
                                sent=sent,
                                sent_pad_length=pad_length) for sent in self.sent_list]

        print('Vocab Size: %d' % len(self.word_dict.keys()))
        print('Max Sentence Length: %d' % self.max_length)
        print('Number of Sentences: %d' % len(self.sent_list))

    def word_to_index(self, word):
        if word not in self.word_dict.keys():
            return self.word_dict['<UNK>']
        return self.word_dict[word]

    def index_to_word(self, index):
        return self.word_index_dict[index]

    def sent_to_indexes(self, sent, sent_pad_length, word_pad_length=3):
        word_list = self.bpe.apply(sent) # ['w o r d 1', ...]
        indexes = []
        # print(word_list)
        for word in word_list:
            sub_word_list = [w for w in word.strip().split(' ')]
            if len(sub_word_list) < word_pad_length:
                sub_word_list += ['UNK'] * (word_pad_length - len(sub_word_list))
            else:
                sub_word_list = sorted(sub_word_list, key=word_sort_key)[:word_pad_length]
            sub_word_list = [self.word_to_index(w) for w in sub_word_list]
            indexes.append(sub_word_list)
        self.max_length = max(self.max_length, len(indexes))
        if len(indexes) < sent_pad_length:
            indexes += [[self.word_to_index('<UNK>')] * word_pad_length] * \
                       (sent_pad_length - len(indexes))
        else:
            indexes = indexes[:sent_pad_length]
        return indexes

    def __len__(self):
        return len(self.sent_list)

    def __getitem__(self, idx):
        indexes = self.indexes_list[idx]
        indexes = np.array(indexes)
        indexes = torch.from_numpy(indexes).type(torch.LongTensor)
        label = self.label_list[idx]
        label = np.array(label)
        label = torch.from_numpy(label).type(torch.FloatTensor)
        name = self.name_list[idx]
        return indexes, label, name

class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()

    def init_param(self):
        self.gpus = torch.cuda.device_count()

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle
                        )
        return data_loader

if __name__ == '__main__':
    pass