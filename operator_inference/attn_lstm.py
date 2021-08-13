import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import models

class Text(object):
    def __init__(self, args, idx2word, word2idx, label_list):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.bce = nn.BCELoss().cuda()
        self.ce = nn.CrossEntropyLoss().cuda()
        self.last_output = None
        self.idx2word = idx2word # function
        self.word2idx = word2idx
        self.topk = 1
        self.label_list = label_list
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        self.model = models.__dict__['AttentionModel'](
                        bpe=True,
                        output_size=self.args.num_labels,
                        hidden_size=self.args.hidden_dim,
                        vocab_size=self.args.vocab_size,
                        embedding_length=self.args.embed_dim,
                        dropout=self.args.dropout
                        )
        self.model = torch.nn.DataParallel(self.model).cuda()    

        self.optim = torch.optim.Adam(
                        self.model.module.parameters(),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'model': self.model.module.state_dict(),
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.model.module.load_state_dict(ckpt['model'])

    def zero_grad(self):
        self.model.zero_grad()

    def set_train(self):
        self.model.train()
    
    def set_eval(self):
        self.model.eval()

    def save_output(self, scores, labels, name_list, path):
        with open(path, 'w') as f:
            assert len(scores) == len(labels)
            for i in range(len(scores)):
                name = name_list[i]
                score = scores[i]
                label = labels[i]
                assert len(score) == len(label)
                f.write('%s Pred:' % name)
                for j in range(len(score)):
                    if score[j] > 0.5:
                        f.write(' %s' % self.label_list[j])
                f.write('\n')
                f.write('%s Label:' % name)
                for j in range(len(label)):
                    if label[j]:
                        f.write(' %s' % self.label_list[j])
                f.write('\n\n')

    def is_correct(self, preds, y, n_class):
        preds = (preds > 0.5)
        correct = (preds == y).float()
        total_correct = torch.sum(correct, dim=1) / n_class
        return total_correct.sum() / len(total_correct)

    def train(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.set_train()
            record = utils.Record()
            record_acc = utils.Record()
            start_time = time.time()
            for i, (indexes, labels, name) in enumerate(tqdm(data_loader)):
                self.zero_grad()
                indexes = indexes.cuda()
                labels = labels.cuda()
                scores = self.model(indexes)
                # loss = self.ce(scores, labels)
                loss = self.bce(scores, labels)
                loss.backward()
                self.optim.step()
                record.add(loss.detach().item())
                acc = self.is_correct(scores.detach(), labels.detach(), self.args.num_labels)
                record_acc.add(float(acc))
            print('----------------------------------------')
            print('Epoch: %d' % self.epoch)
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('Top %d Acc: %f' % (self.topk, record_acc.mean()))

    def test(self, data_loader):
        self.set_eval()
        record = utils.Record()
        record_acc = utils.Record()
        start_time = time.time()
        with torch.no_grad():
            for i, (indexes, labels, name_list) in enumerate(tqdm(data_loader)):
                indexes = indexes.cuda()
                labels = labels.cuda()
                scores = self.model(indexes)
                loss = self.bce(scores, labels)
                record.add(loss.detach().item())
                acc = self.is_correct(scores.detach(), labels.detach(), self.args.num_labels)
                record_acc.add(float(acc))
                self.save_output(
                    scores=scores.data,
                    labels=labels.data,
                    name_list=name_list,
                    path=((self.args.text_root+'test_%03d-%03d.txt') % (self.epoch, i))
                    )
            print('----------------------------------------')
            print('Test.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('Top %d Acc: %f' % (self.topk, record_acc.mean()))

if __name__ == '__main__':
    import argparse

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    from data_loader import *

    compiler = 'TVM'
    setting = 'O3'

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=('%s-%s' % (compiler, setting)))

    parser.add_argument('--output_root', type=str, default='/root/output/decompiler/')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--test_freq', type=int, default=5)

    parser.add_argument('--embed_dim', type=int, default=128)
    # parser.add_argument('--num_labels', type=int, default=87)
    parser.add_argument('--num_labels', type=int, default=-1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dense_layer_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--vocab_size', type=int, default=-1)

    args = parser.parse_args()

    print(args.exp_name)

    utils.make_path(args.output_root)
    utils.make_path(args.output_root + args.exp_name)

    args.ckpt_root = args.output_root + args.exp_name + '/ckpt/'
    args.text_root = args.output_root + args.exp_name + '/text/'

    utils.make_path(args.ckpt_root)
    utils.make_path(args.text_root)

    with open(args.output_root + args.exp_name + '/args.json', 'w') as f:
        json.dump(args.__dict__, f)

    loader = DataLoader(args)


    train_dataset = OPSepDataset(
                    sent_dir=('/path_to_data/%s/train/' % compiler),
                    bpe_path=('/path_to_bpe/%s_BPE' % compiler),
                    vocab_path=('%s_vocab_BPE.json' % compiler),
                    label_dict_path=('%s_index.json' % compiler),
                    transform_dict_path=('%s_label.json' % compiler),
                    pad_length=500, # TVM: 500, GLOW: 1500
                    setting=setting
                )

    test_dataset = OPSepDataset(
                    sent_dir=('/path_to_data/%s/test/' % compiler),
                    bpe_path=('/path_to_bpe/%s_BPE' % compiler),
                    vocab_path=('%s_vocab_BPE.json' % compiler),
                    label_dict_path=('%s_index.json' % compiler),
                    transform_dict_path=('%s_label.json' % compiler),
                    pad_length=500,
                    setting=setting
                )

    args.vocab_size = len(train_dataset.word_dict.keys())
    args.num_labels = len(train_dataset.label_dict.keys())

    model = Text(args, 
            idx2word=train_dataset.index_to_word,
            word2idx=train_dataset.word_to_index,
            label_list=list(train_dataset.label_dict.keys())
        ) 

    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset, False)

    for i in range(args.num_epoch):
        model.train(train_loader)
        if i % args.test_freq == 0:
            model.test(test_loader)
            model.save_model((args.ckpt_root + '%03d.pth') % (i + 1))
    model.save_model(args.ckpt_root + 'final.pth')

