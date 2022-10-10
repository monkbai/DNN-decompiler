import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from engine import Engine
from data_loader import OPSeqDataset, DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str, default='artifact-demo')

parser.add_argument('--output_dir', type=str, default='./output/')
parser.add_argument('--sent_dir', type=str, default='./data/dataset/')
parser.add_argument('--bpe_dir', type=str, default='./data/bpe/')
parser.add_argument('--vocab_dir', type=str, default='./data/vocab/')
parser.add_argument('--index_dir', type=str, default='./data/index/')
parser.add_argument('--label_dir', type=str, default='./data/label/')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--test_freq', type=int, default=5)

parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--num_labels', type=int, default=-1) # set automatically
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--dense_layer_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--vocab_size', type=int, default=-1)

parser.add_argument('--compiler', type=str, default='TVM', choices=['TVM', 'GLOW'])
parser.add_argument('--setting', type=str, default='TVM_v0.7_O0')
'''
version & opt level
for TVM, choices=['TVM_v0.7_O0', 'TVM_v0.8_O0', 'TVM_v0.9.dev_O0',
                  'TVM_v0.7_O3', 'TVM_v0.8_O3', 'TVM_v0.9.dev_O3']
for GLOW, choices=['GLOW_2020', 'GLOW_2021', 'GLOW_2022']
'''
parser.add_argument('--pad_length', type=int, default=2000)
parser.add_argument('--training', type=int, default=0, choices=[0, 1])

args = parser.parse_args()

assert args.compiler in args.setting

print(args.exp_name)

utils.make_path(args.output_dir)
utils.make_path(args.output_dir + args.exp_name)

args.ckpt_dir = args.output_dir + args.exp_name + '/ckpt/'
args.text_dir = args.output_dir + args.exp_name + '/text/'

utils.make_path(args.ckpt_dir)
utils.make_path(args.text_dir)

with open(args.output_dir + args.exp_name + '/args.json', 'w') as f:
    json.dump(args.__dict__, f)

loader = DataLoader(args)

train_dataset = OPSeqDataset(
                sent_dir=args.sent_dir + ('%s/train/' % args.compiler),
                bpe_path=args.bpe_dir + ('%s_BPE_200' % args.compiler),
                vocab_path=args.vocab_dir + ('%s_vocab_BPE_200.json' % args.compiler),
                index_path=args.index_dir + ('%s_index_final.json' % args.compiler),
                label_path=args.label_dir + ('%s_label_final.json' % args.compiler),
                pad_length=args.pad_length,
                setting=args.setting
            )

test_dataset = OPSeqDataset(
                sent_dir=args.sent_dir + ('%s/test/' % args.compiler),
                bpe_path=args.bpe_dir + ('%s_BPE_200' % args.compiler),
                vocab_path=args.vocab_dir + ('%s_vocab_BPE_200.json' % args.compiler),
                index_path=args.index_dir + ('%s_index_final.json' % args.compiler),
                label_path=args.label_dir + ('%s_label_final.json' % args.compiler),
                pad_length=args.pad_length,
                setting=args.setting
            )

args.vocab_size = len(train_dataset.word_dict.keys())
args.num_labels = len(train_dataset.label_index_dict.keys())

print('vocab_size: ', args.vocab_size)
print('num_labels: ', args.num_labels)

engine = Engine(args, 
        idx2word=train_dataset.index_to_word,
        word2idx=train_dataset.word_to_index,
        label_list=list(train_dataset.label_index_dict.keys())
    ) 

train_loader = loader.get_loader(train_dataset)
test_loader = loader.get_loader(test_dataset, False)


if args.training:
    for i in range(args.num_epoch):
        engine.train(train_loader)
        if i % args.test_freq == 0:
            engine.test(test_loader)
            engine.save_model((args.ckpt_dir + '%03d.pth') % (i + 1))
    engine.save_model(args.ckpt_dir + 'final.pth')
else:
    engine.load_model(args.ckpt_dir + 'final2.pth')
    engine.test(test_loader)