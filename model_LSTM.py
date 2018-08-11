# -*- coding: utf-8 -*-
# @Time    : 18-8-3 下午8:42
# @Author  : WangZhen
# @FileName: model_LSTM.py
# @Software: PyCharm Community Edition

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init

torch.manual_seed(300)
random.seed(300)
"""
Neural Networks model : LSTM
"""


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D)

        if args.out_word_v is True:
            self.embed.weight.data.copy_(torch.from_numpy(np.array(args.word_embed)))


        # lstm
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers)

        # if args.init_weight:
        #     print("Initing W .......")
        #     # n = self.lstm.input_size * self.lstm
        #     init.xavier_normal(self.lstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
        #     init.xavier_normal(self.lstm.all_weights[0][1], gain=np.sqrt(args.init_weight_value))

        # linear
        self.hidden2label = nn.Linear(self.hidden_dim, C)
        # dropout
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout_embed(embed)
        x = embed.view(len(x), embed.size(1), -1)

        # lstm
        lstm_out, _ = self.lstm(x)

        # lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # print(lstm_out)

        # pooling
        lstm_out = F.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = F.tanh(lstm_out)

        lstm_out = self.dropout(lstm_out)
        # linear
        logit = self.hidden2label(lstm_out)
        return logit


