# -*- coding: utf-8 -*-
# @Time    : 18-8-3 下午8:42
# @Author  : WangZhen
# @FileName: main.py
# @Software: PyCharm Community Edition
import model_LSTM
import DP as dp
import argparse
import train
from word2vec import out_word_vec

parser = argparse.ArgumentParser(description='LSTM text classification')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=50, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-init-weight-decay', type = float, default=1e-8, help='l2 the parameters of weight decay')
parser.add_argument('-batch-normalizations', type = bool, default=True, help='batch_normalizations')
parser.add_argument('-bath-norm-momentum', type = float, default=0.1, help='bath_norm_momentum')
parser.add_argument('-batch-norm-affine', type = bool, default=True, help='batch_norm_affine')
parser.add_argument('-wide-conv', type = bool, default=True, help='wide_conv')
parser.add_argument('-seed-num', type = int , default=233, help='in order to chang the par of k_dim')
parser.add_argument('-out-word-v', type= bool, default=True, help='whether to add out-word-v')
parser.add_argument('-word-embed', type=list, default=[], help='worc-v')
parser.add_argument('-lstm-hidden-dim', type=int, default=300, help='lstm_hidden_dim')
parser.add_argument('-lstm-num-layers', type=int, default=1, help='lstm_num_layers')


# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')

args = parser.parse_args()

def loaddata():

    print('loading the data.....')
    # 标签建词典
    lab_v = []
    lab_v.append('negative')
    lab_v.append('positive')

    # 处理训练数据 包括建数据词典等
    t_data_list_node, data_v = dp.datadeal('data/raw.clean.train', is_traindata=True)
    # 处理dev数据
    d_data_list_node = dp.datadeal('data/raw.clean.dev', is_traindata=False)
    # 处理test数据
    test_data_node = dp.datadeal('data/raw.clean.test', is_traindata=False)

    if args.out_word_v:
        args.word_embed = out_word_vec.add_word_v(data_v)

    args.embed_num = len(data_v)
    args.class_num = len(lab_v)

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr == 'word_embed':
            continue
        print("\t{}={}".format(attr.upper(), value))

    return t_data_list_node, data_v, d_data_list_node, lab_v

if __name__ == "__main__":

    train_data_list_node, data_voc, dev_data_list_node, lab_voc = loaddata()
    lstm = model_LSTM.LSTM(args)

    try:
        train.train_lstm(train_data_list_node, data_voc, dev_data_list_node, lab_voc, lstm, args)
    except KeyboardInterrupt:
        print('\nstop by human!!!')

