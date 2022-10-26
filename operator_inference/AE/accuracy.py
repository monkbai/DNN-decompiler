import json
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from engine import Engine
from data_loader import OPSeqDataset, DataLoader

parser = argparse.ArgumentParser()

# parser.add_argument('--exp_name', type=str, default='artifact-demo')

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

parser.add_argument('--model', type=str, default='', choices=['',
                    'resnet18', 'vgg16', 'inception_v1',
                    'shufflenet_v2', 'mobilenet', 'efficientnet'])

args = parser.parse_args()

args.exp_name = args.setting

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
                setting=args.setting+(('-' + args.model) if len(args.model) else '')
            )

test_dataset = OPSeqDataset(
                sent_dir=args.sent_dir + ('%s/test/' % args.compiler),
                bpe_path=args.bpe_dir + ('%s_BPE_200' % args.compiler),
                vocab_path=args.vocab_dir + ('%s_vocab_BPE_200.json' % args.compiler),
                index_path=args.index_dir + ('%s_index_final.json' % args.compiler),
                label_path=args.label_dir + ('%s_label_final.json' % args.compiler),
                pad_length=args.pad_length,
                setting=args.setting+(('-' + args.model) if len(args.model) else '')
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

# train_loader = loader.get_loader(train_dataset)
test_loader = loader.get_loader(test_dataset, False)

with open('transform.json', 'r') as f:
    TABLE = json.load(f)

LABEL_INDEX = train_dataset.label_index_dict

def accuracy(pred_list, GT_list, setting, f=None):
    # some operators have different (name) but equivalent implementations;
    # fix the naming issue here
    assert len(pred_list) == len(GT_list)
    acc_list = []
    for i in range(len(pred_list)):
        if 'O3' in setting:
            pred = ' '.join(pred_list[i])
            GT = ' '.join(GT_list[i])
            for k, v in TABLE['TVM-O3'].items():
                if k in pred: pred = pred.replace(k, v)
                if k in GT: GT = GT.replace(k, v)
            pred = [x.strip() for x in pred.split(' ')]
            GT = [x.strip() for x in GT.split(' ')]
        else:
            pred = pred_list[i]
            GT = GT_list[i]
        if 'TVM' in setting:
            for j in range(len(pred)):
                pred[j] = TABLE['TVM-O0'].get(pred[j], pred[j])
            for j in range(len(GT)):
                GT[j] = TABLE['TVM-O0'].get(GT[j], GT[j])
        else:
            for j in range(len(pred)):
                pred[j] = TABLE['GLOW'].get(pred[j], pred[j])
            for j in range(len(GT)):
                GT[j] = TABLE['GLOW'].get(GT[j], GT[j])
        if not len(''.join(GT)) or not len(''.join(pred)):
            continue
        pred_id = np.array([0] * args.num_labels)
        GT_id = np.array([0] * args.num_labels)
        for j in range(len(pred)):
            if len(pred[j]):
                pred_id[LABEL_INDEX.get(pred[j], 0)] = 1
        for j in range(len(GT)):
            if len(GT[j]):
                GT_id[LABEL_INDEX.get(GT[j], 0)] = 1
        acc = (pred_id == GT_id).sum() / len(pred_id)
        if acc < 1:
            print('\twrong prediction: ', pred)
            print('\tground truth: ', GT)
            if f is not None:
                print('\twrong prediction: ', pred, file=f)
                print('\tground truth: ', GT, file=f)
        acc_list.append(acc)
    return np.mean(acc_list)

# utils.make_path('log')
# f = open('log/%s.txt' % (args.setting+('-'+args.model if len(args.model) else '')), 'w')
f_acc = open('acc.txt', 'a')
f_log = open('log.txt', 'a')

if f_log is not None:
    print(args.setting+(('-' + args.model) if len(args.model) else ''), file=f_log)

engine.load_model(args.ckpt_dir + ('final.pth' if args.compiler == 'TVM' else 'final2.pth'))
engine.set_eval()
record_acc = utils.Record()
with torch.no_grad():
    total_acc_list = []
    for i, (indexes, labels, name_list) in enumerate(tqdm(test_loader)):
        indexes = indexes.to(engine.device)
        labels = labels.to(engine.device)

        pred_list = engine.inference(indexes)
        GT_list = engine.id_to_text(labels)

        record_acc.add(accuracy(pred_list, GT_list, args.setting, f_log))

    # print('%s Acc: %f' % ( (args.setting+('-'+args.model if len(args.model) else '')),
    #                        record_acc.mean() ))
    # print('%s Acc: %f' % ( (args.setting+('-'+args.model if len(args.model) else '')),
    #                        record_acc.mean() ), file=f_acc)
    print('%s Acc: %.2f%%' % ( (args.setting+('-'+args.model if len(args.model) else '')),
                           record_acc.mean() * 100 ))
    print('%s Acc: %.2f%%' % ( (args.setting+('-'+args.model if len(args.model) else '')),
                           record_acc.mean() * 100 ), file=f_acc)

f_acc.close()
f_log.close()