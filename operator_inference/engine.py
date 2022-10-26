import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import utils
import models


class Engine(object):
    def __init__(self, args, idx2word, word2idx, label_list):
        self.args = args
        self.epoch = 0
        self.max_acc = -1
        self.max_epoch = -1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mse = nn.MSELoss().to(self.device)
        self.l1 = nn.L1Loss().to(self.device)
        self.bce = nn.BCELoss().to(self.device)
        self.ce = nn.CrossEntropyLoss().to(self.device)
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
        
        self.model = self.model.to(self.device)    

        self.optim = torch.optim.Adam(
                        self.model.parameters(),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'model': self.model.state_dict(),
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model'])

    def zero_grad(self):
        self.model.zero_grad()

    def set_train(self):
        self.model.train()
    
    def set_eval(self):
        self.model.eval()

    def save_output(self, scores, labels, name_list, f):
        # with open(path, 'w') as f:
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
            for i, (indexes, labels, name_list) in enumerate(tqdm(data_loader)):
                self.zero_grad()
                indexes = indexes.to(self.device)
                labels = labels.to(self.device)
                
                scores = self.model(indexes)
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
            print('Acc: %f' % (record_acc.mean()))

    def test(self, data_loader):
        self.set_eval()
        record = utils.Record()
        record_acc = utils.Record()
        start_time = time.time()
        with torch.no_grad():
            path = ((self.args.text_dir+'test_%03d.txt') % (self.epoch))
            with open(path, 'w') as f:    
                for i, (indexes, labels, name_list) in enumerate(tqdm(data_loader)):
                    indexes = indexes.to(self.device)
                    labels = labels.to(self.device)
                    scores = self.model(indexes)
                    loss = self.bce(scores, labels)
                    record.add(loss.detach().item())
                    acc = self.is_correct(scores.detach(), labels.detach(), self.args.num_labels)
                    record_acc.add(float(acc))

                    self.save_output(
                        scores=scores.data,
                        labels=labels.data,
                        name_list=name_list,
                        f=f)

            print('----------------------------------------')
            print('Test.')
            print('Costs Time: %.2f s' % (time.time() - start_time))
            print('Recons Loss: %f' % (record.mean()))
            print('Acc: %f' % (record_acc.mean()))
            print('Inference results are written in %s' % path)

    def id_to_text(self, scores):
        result_list = []
        for i in range(len(scores)):
            score = scores[i]
            result = []
            for j in range(len(score)):
                if score[j] > 0.5:
                    result.append(self.label_list[j])
            result_list.append(result)
        return result_list

    def inference(self, indexes):
        self.set_eval()
        indexes = indexes.to(self.device)
        scores = self.model(indexes)
        return self.id_to_text(scores.data)

if __name__ == '__main__':
    pass