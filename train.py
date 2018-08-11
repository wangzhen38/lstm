# -*- coding: utf-8 -*-
# @Time    : 18-8-6 下午7:49
# @Author  : WangZhen
# @FileName: train.py
# @Software: PyCharm Community Edition
import DP
import random
import data_node
import torch.nn.functional as F
import torch
import os
import sys
from torch.autograd import Variable
import torch.nn.utils as utils


def train_lstm(train_data_list_node, data_voc, dev_data_list_node, lab_voc, model, args):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
    steps = 0
    best_acc = 0
    model.train()
    sentence_num = len(train_data_list_node)
    for epoch in range(args.epochs):  # 训练次数
        if args.shuffle:  # 是否打乱数据
            random.shuffle(train_data_list_node)
        for batch in range(0, sentence_num, args.batch_size):  # 模拟迭代器
            max_len = 0  # 最大句子长度
            l = min(batch + args.batch_size, sentence_num)    # 考虑会有越界情况
            for i in range(batch, l):
                max_len = max(max_len, train_data_list_node[i].train_data_word_num)
            feature = []
            target = []
            for i in range(batch, l):  # 处理每个batch
                n_list = []
                # print('len(train_data_list_node[i].train_data)=',len(train_data_list_node[i].train_data))
                for j in range(len(train_data_list_node[i].train_data)):
                    word = train_data_list_node[i].train_data[j].lower()
                    if word in data_voc:
                        n_list.append(data_voc.index(word))
                    else:  # 正常情况下只有测试会出现此情况
                        print('error')
                        n_list.append(0)
                for j in range(max_len - len(train_data_list_node[i].train_data)):
                    n_list.append(1)
                feature.append(n_list)
                target.append(lab_voc.index(train_data_list_node[i].train_data_label))
            feature, target = Variable(torch.LongTensor(feature)), Variable(torch.LongTensor(target))

            optimizer.zero_grad()
            logit = model(feature)

            loss = F.cross_entropy(logit, target)
            loss.backward()

            if args.max_norm is not None:
                utils.clip_grad_norm(model.parameters(), max_norm=args.max_norm)

            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / args.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects,
                                                                             args.batch_size))

            if steps % args.test_interval == 0:
                dev_acc = eval(data_voc, dev_data_list_node, lab_voc, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                    else :
                        if steps - last_step >= args.early_stop:
                            print('early stop by {} steps.'.format(args.early_stop))


def eval(data_voc, dev_data_list_node, lab_voc, model, args):
    model.eval()
    corrects, avg_loss = 0, 0

    if args.shuffle:  # 是否打乱数据
        random.shuffle(dev_data_list_node)
    sentence_num = len(dev_data_list_node)

    for batch in range(0, sentence_num, args.batch_size):  # 模拟迭代器
        max_len = 0  # 最大句子长度
        l = min(batch + args.batch_size, sentence_num)    # 考虑会有越界情况
        for i in range(batch, l):
            max_len = max(max_len, dev_data_list_node[i].train_data_word_num)
        feature = []
        target = []
        for i in range(batch, l):  # 处理每个batch
            n_list = []
            for j in range(len(dev_data_list_node[i].train_data)):
                word = dev_data_list_node[i].train_data[j].lower()
                if word in data_voc:
                    n_list.append(data_voc.index(word))
                else:  # 测试会出现此情况
                    n_list.append(data_voc.index('<unk>'))

            for j in range(max_len - len(dev_data_list_node[i].train_data)):
                n_list.append(data_voc.index('<pad>'))

            feature.append(n_list)
            target.append(lab_voc.index(dev_data_list_node[i].train_data_label))

        feature, target = Variable(torch.LongTensor(feature)), Variable(torch.LongTensor(target))

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = sentence_num
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.txt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)










