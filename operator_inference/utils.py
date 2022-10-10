import torch
import os
import random
import json
import torchvision
import fastBPE
import numpy as np


class BPE:
    def __init__(self, codes_path):
        self.bpe = fastBPE.fastBPE(codes_path)

    def apply(self, sent_list):
        out = self.bpe.apply(sent_list)
        out = [o.replace('@@', '') for o in out]
        return out


def accuracy(scores, targets, k=1):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() / batch_size


def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def build_vocab(vocab_path, token_list):
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            token_vocab = vocab['token']
            index_vocab = vocab['index']
    else:
        token_vocab = {}
        index_vocab = {}
        count = {}
        cur_index = 0
        for token in token_list:
            for t in token:
                if t in count.keys():
                    count[t] += 1
                    if count[t] == 10:
                        token_vocab[t] = cur_index
                        index_vocab[cur_index] = t
                        cur_index += 1
                else:
                    count[t] = 1
        for t in ['<UNK>', '<START>', '<END>']:
            token_vocab[t] = cur_index
            index_vocab[cur_index] = t
            cur_index += 1
        with open(vocab_path, 'w') as f:
            json.dump({'token': token_vocab,
                       'index': index_vocab}, f)
    return token_vocab, index_vocab


class Record(object):
    def __init__(self):
        self.loss = 0
        self.count = 0

    def add(self, value):
        self.loss += value
        self.count += 1

    def mean(self):
        return self.loss / self.count
