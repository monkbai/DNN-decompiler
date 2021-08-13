import torch
import math
irange = range
import os
import random
import shutil
import json
import progressbar
import torchvision
import fastBPE
import numpy as np
import progressbar

class BPE:
    def __init__(self, codes_path):
        self.bpe = fastBPE.fastBPE(codes_path)

    def apply(self, sent_list):
        out = self.bpe.apply(sent_list)
        out = [o.replace('@@', '') for o in out]
        return out

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def accuracy(scores, targets, k=1):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    #print('bs:', batch_size)
    _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
    #print('ind: ', ind.shape)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    #print('correct: ', correct.shape)
    #print(correct)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() / batch_size


def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_params(json_file):
    with open(json_file) as f:
        return json.load(f)

def get_batch(data_loader):
    while True:
        for batch in data_loader:
            yield batch

class Record(object):
    def __init__(self):
        self.loss = 0
        self.count = 0

    def add(self, value):
        self.loss += value
        self.count += 1

    def mean(self):
        return self.loss / self.count
